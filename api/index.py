"""HTTP layer for the message-intelligence system.

Deployment target is a Vercel Python Function, so this module is deliberately
thin: it parses input, calls `mint`, and serialises the result. All of the
actual logic lives in the package so that the CLI, the tests and the web app
exercise exactly the same code path.

Data-handling properties of this service:

* Nothing is written to disk and nothing is cached between requests. An
  uploaded CSV lives in memory for the duration of one request and is then
  garbage collected.
* Responses contain masked text only, because `mint.pipeline` masks before
  anything else runs. There is no redaction step here to forget to call.
* There are no outbound network calls. No message text is sent anywhere.
"""

from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from mint import __version__, pipeline  # noqa: E402
from mint import taxonomy as T  # noqa: E402
from mint.classifier import classify as classify_message  # noqa: E402
from mint.extract import extract as extract_item  # noqa: E402
from mint.model import load_default  # noqa: E402
from mint.sensitive import HUMAN_LABEL, RECOMMENDED_ACTION, RISK_LEVEL, scan  # noqa: E402

MAX_UPLOAD_BYTES = 8 * 1024 * 1024   # 8 MB
MAX_ROWS = 20_000

api = FastAPI(
    title="Local Message Intelligence",
    version=__version__,
    description=(
        "Classifies messages, extracts tasks and events, and detects and masks "
        "sensitive information. Runs entirely locally -- no external AI services."
    ),
    docs_url="/v1/docs",
    openapi_url="/v1/openapi.json",
)

WEB_DIR = ROOT / "web"
SAMPLE_CSV = ROOT / "sample_data" / "sample_messages.csv"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def serialise(result: pipeline.PipelineResult) -> dict:
    return {
        "summary": result.summary(),
        "messages": [
            {
                **m.to_row(),
                "secondary_category": m.classification.secondary_category,
                "evidence": m.classification.to_dict()["evidence"],
                "item_id": m.item.item_id if m.item else None,
                "sensitivity_type": m.scan_result.primary_type,
                "recommended_action": m.scan_result.overall_action,
            }
            for m in result.messages
        ],
        "items": [i.to_dict() for i in result.items],
        "sensitive": result.sensitive_report(),
        "taxonomy": {
            "categories": T.CATEGORIES,
            "descriptions": T.DESCRIPTIONS,
        },
    }


def _read_upload(raw: bytes) -> str:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB")
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(400, "could not decode the file as text")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@api.get("/v1/health")
def health() -> dict:
    model = load_default()
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": model is not None,
        "runtime": "pure-python inference (no scikit-learn at serve time)",
        "sends_data_externally": False,
    }


@api.get("/v1/model")
def model_info() -> dict:
    model = load_default()
    if model is None:
        return {"trained": False,
                "note": "run scripts/train.py to produce models/classifier.json"}
    return {
        "trained": True,
        "classes": model.classes,
        "vocabulary_size": len(model.vocabulary),
        "ngram_max": model.ngram_max,
        "metadata": model.metadata,
        "taxonomy": {
            "categories": T.CATEGORIES,
            "descriptions": T.DESCRIPTIONS,
            "model_categories": T.MODEL_CATEGORIES,
            "note": (
                "sensitive_information is decided by a deterministic detector, "
                "not by the statistical model."
            ),
        },
        "sensitivity_types": [
            {
                "type": t,
                "label": HUMAN_LABEL[t],
                "risk": RISK_LEVEL[t],
                "recommended_action": RECOMMENDED_ACTION[t],
            }
            for t in RISK_LEVEL
        ],
    }


@api.get("/v1/analyze/sample")
def analyze_sample() -> dict:
    """Run the bundled synthetic sample.

    This file is written from scratch for the public demo. The assessment
    dataset is never committed to this repository, so the hosted demo has
    nothing confidential to expose -- upload the real CSV to analyse it.
    """
    if not SAMPLE_CSV.exists():
        raise HTTPException(500, "bundled sample data is missing")
    result = pipeline.run(SAMPLE_CSV)
    payload = serialise(result)
    payload["source"] = {
        "name": "sample_messages.csv",
        "kind": "bundled synthetic sample",
        "note": "Invented for this demo. Not the assessment dataset.",
    }
    return payload


@api.post("/v1/analyze")
async def analyze_upload(file: UploadFile = File(...)) -> dict:
    """Analyse an uploaded CSV. Held in memory for this request only."""
    text = _read_upload(await file.read())
    if text.count("\n") > MAX_ROWS:
        raise HTTPException(413, f"file exceeds {MAX_ROWS} rows")
    try:
        result = pipeline.run(text)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    payload = serialise(result)
    payload["source"] = {
        "name": file.filename or "upload.csv",
        "kind": "uploaded file",
        "note": "Processed in memory and discarded; nothing was written to disk.",
    }
    return payload


class MessageIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    sender: str = Field("", max_length=120)
    timestamp: str = Field("", max_length=40)
    message_id: str = Field("LIVE_001", max_length=64)


@api.post("/v1/classify")
def classify_one(payload: MessageIn) -> dict:
    """Classify a single message typed into the UI, end to end."""
    scan_result = scan(payload.message)
    masked = scan_result.masked_text

    cls = classify_message(
        message_id=payload.message_id,
        masked_text=masked,
        sender=payload.sender,
        scan_result=scan_result,
    )
    # No epoch default: a single typed message has no send time unless the
    # caller supplies one, and inventing 1970-01-01 would make every deadline
    # look ~20,000 days away. An empty timestamp leaves the send date
    # unresolved, so priority falls back to its stated-but-unanchored reason.
    item = extract_item(
        payload.message_id, masked, payload.sender,
        payload.timestamp, cls.category, 1, (),
    )

    return {
        # The raw input is never echoed back -- only the masked form.
        "masked_text": masked,
        "classification": cls.to_dict(),
        "extracted_item": item.to_dict() if item else None,
        "sensitive": {
            "detected": scan_result.is_sensitive,
            "types": scan_result.types,
            "risk": scan_result.overall_risk,
            "recommended_action": scan_result.overall_action,
            "findings": [
                {
                    "sensitivity_type": f.sensitivity_type,
                    "label": HUMAN_LABEL.get(f.sensitivity_type, ""),
                    "risk": f.risk,
                    "recommended_action": f.recommended_action,
                    "reason": f.reason,
                }
                for f in scan_result.findings
            ],
        },
    }


@api.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    path = WEB_DIR / "index.html"
    if not path.exists():
        return HTMLResponse("<h1>UI not found</h1>", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@api.exception_handler(404)
def not_found(_request, _exc):
    return JSONResponse({"error": "not found"}, status_code=404)


class RestoreOriginalPath:
    """Put the real request path back before FastAPI routes the request.

    Vercel's catch-all rewrite (`/(.*)` -> `/api/index`) is a genuine rewrite:
    the function receives `/api/index` as its path for every URL, and no header
    carries the original. So `vercel.json` smuggles the original path through a
    `__path` query parameter, and this wrapper restores it -- and removes the
    parameter again, so handlers never see it.

    Locally there is no `__path`, so this is a transparent pass-through and the
    dev server behaves identically to production.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            params = urllib.parse.parse_qs(scope.get("query_string", b"").decode())
            original = params.pop("__path", [None])[0]
            if original:
                scope = dict(scope)
                scope["path"] = original
                scope["raw_path"] = original.encode()
                scope["query_string"] = urllib.parse.urlencode(
                    params, doseq=True
                ).encode()
        await self.app(scope, receive, send)


#: The ASGI callable Vercel and uvicorn both look for.
app = RestoreOriginalPath(api)
