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
from fastapi.responses import HTMLResponse, JSONResponse, Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from mint import __version__, pipeline  # noqa: E402
from mint import groups as GRP  # noqa: E402
from mint import ledger as LEDGER  # noqa: E402
from mint import priority as PRIO  # noqa: E402
from mint import routing as ROUTE  # noqa: E402
from mint.assistant import Assistant  # noqa: E402
from mint.embed import load_default as load_semantic  # noqa: E402
from mint.retrieve import ExactLexicalIndex, HybridIndex  # noqa: E402
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
SAMPLE_L2 = ROOT / "sample_data" / "sample_l2_messages.csv"
SAMPLE_L2_DEMO = ROOT / "sample_data" / "sample_l2_demo.csv"


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


@api.get("/app.css")
def stylesheet():
    """One stylesheet, shared by the L1 and L2 pages.

    Both views are the same product, so they use the same palette and the same
    components; serving the CSS once is how that stays true rather than
    drifting into two lookalikes.
    """
    path = WEB_DIR / "app.css"
    if not path.exists():
        raise HTTPException(404, "stylesheet not found")
    return Response(path.read_text(encoding="utf-8"), media_type="text/css")


@api.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    path = WEB_DIR / "index.html"
    if not path.exists():
        return HTMLResponse("<h1>UI not found</h1>", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@api.get("/l2", response_class=HTMLResponse)
def l2_index() -> HTMLResponse:
    """The L2 view: priority, subject groups, the assistant, privacy, benchmark."""
    path = WEB_DIR / "l2.html"
    if not path.exists():
        return HTMLResponse("<h1>L2 UI not found</h1>", status_code=500)
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


# ===========================================================================
# L2: the ledger, the assistant, privacy routing and the live benchmark
# ===========================================================================
"""
Session model
-------------
L2 answers questions *about a corpus*, so a query needs the analysed corpus to
be there when it arrives. Vercel functions are stateless between cold starts,
so something has to hold it.

What is held is deliberately narrow:

* **The raw CSV text is never retained.** Masking runs during ingest and the
  ledger that survives contains masked text only, exactly like every other
  output. There is nothing in the cache that a request could not already see.
* **The cache is memory only, bounded, and content-addressed.** The key is a
  hash of the uploaded bytes, so re-uploading the same file reuses the same
  session instead of accumulating copies. Nothing is written to disk.
* **A cache miss is not an error the user has to understand.** A cold instance
  returns 409 and the browser re-uploads the file it still holds.
* **Adding a batch means re-analysing from scratch.** There is no incremental
  append path that would require keeping the earlier raw text around. Re-doing
  1,100 messages costs ~200 ms, which is a good price for not storing them.
"""

import hashlib  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

#: At most this many analysed corpora are held at once, oldest evicted first.
MAX_SESSIONS = 3
MAX_FILES = 6

_SESSIONS: "OrderedDict[str, LEDGER.Ledger]" = OrderedDict()


def _remember(key: str, ledger: LEDGER.Ledger) -> None:
    _SESSIONS[key] = ledger
    _SESSIONS.move_to_end(key)
    while len(_SESSIONS) > MAX_SESSIONS:
        _SESSIONS.popitem(last=False)


def _session(key: str) -> LEDGER.Ledger:
    ledger = _SESSIONS.get(key)
    if ledger is None:
        raise HTTPException(
            409,
            "this analysis is no longer in memory (the server may have "
            "restarted). Re-upload the file to continue.")
    _SESSIONS.move_to_end(key)
    return ledger


def serialise_ledger(ledger: LEDGER.Ledger, session_id: str,
                     source: dict) -> dict:
    """The L2 payload. Masked text only, because that is all the ledger holds.

    Per-item priority records are sent in a compact form and the full signal
    detail is sent once per group. Every item extracted from one subject
    carries that subject's priority, so shipping 543 copies of the same
    reasoning would be several hundred kilobytes of duplication.
    """
    groups = []
    for group in ledger.groups:
        payload = group.to_dict()
        decision = ledger.priorities.get(group.group_id)
        payload["priority"] = decision.to_dict() if decision else None
        payload["priority_history"] = [
            c.to_dict() for c in ledger.priority_history.get(group.group_id, [])
        ]
        groups.append(payload)

    items = []
    for rec in ledger.records:
        if rec.item is None:
            continue
        decision = ledger.priorities.get(rec.group_id or "")
        items.append({
            "item_id": rec.item.item_id,
            "message_id": rec.message_id,
            "group_id": rec.group_id,
            "type": rec.item.type,
            "title": rec.item.title,
            "own_deadline": rec.item.date,
            "own_time": rec.item.time,
            "unresolved_fields": rec.item.unresolved_fields,
            "priority": decision.priority if decision else None,
            "confidence": decision.confidence if decision else None,
        })

    return {
        "session_id": session_id,
        "source": source,
        "summary": ledger.summary(),
        "messages": [r.to_dict() for r in ledger.records],
        "groups": groups,
        "items": items,
        "routing": {
            "policy_table": ROUTE.policy_table(),
            "counts": ledger.summary()["route_counts"],
            "decisions": ledger.routing_records(),
        },
        "snapshots": [s.to_dict() for s in ledger.snapshots],
        "vocabulary": {
            "statuses": GRP.STATUSES,
            "status_labels": GRP.STATUS_LABEL,
            "priorities": PRIO.BANDS,
            "routes": ROUTE.ROUTES,
            "route_labels": ROUTE.ROUTE_LABEL,
            "signals": {name: {"family": spec.family, "weight": spec.weight,
                               "explain": spec.explain}
                        for name, spec in PRIO.SIGNALS.items()},
        },
    }


def _analyse(files: List[tuple], source: dict) -> dict:
    """Run the ledger over (name, text) pairs and cache the masked result."""
    digest = hashlib.sha256()
    for name, text in files:
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    session_id = digest.hexdigest()[:24]

    cached = _SESSIONS.get(session_id)
    if cached is not None:
        _SESSIONS.move_to_end(session_id)
        return serialise_ledger(cached, session_id,
                                {**source, "cached": True})

    started = time.perf_counter()
    try:
        ledger = LEDGER.run([(name, text) for name, text in files])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # The raw text goes out of scope here and is never stored.
    _remember(session_id, ledger)
    payload = serialise_ledger(ledger, session_id, source)
    payload["source"]["ingest_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return payload


@api.get("/v2/analyze/sample")
def l2_sample() -> dict:
    """Run the bundled synthetic L2 sample.

    Written from scratch for this demo. The assessment datasets are never
    committed to this repository, so the hosted demo has nothing confidential
    to expose -- upload the real CSVs to analyse them.
    """
    if not SAMPLE_L2.exists():
        raise HTTPException(500, "bundled L2 sample data is missing")
    files = [(SAMPLE_L2.name, SAMPLE_L2.read_text(encoding="utf-8-sig"))]
    if SAMPLE_L2_DEMO.exists():
        files.append((SAMPLE_L2_DEMO.name,
                      SAMPLE_L2_DEMO.read_text(encoding="utf-8-sig")))
    return _analyse(files, {
        "name": " + ".join(n for n, _ in files),
        "kind": "bundled synthetic sample",
        "note": "Invented for this demo. Not the assessment dataset.",
    })


@api.post("/v2/analyze")
async def l2_analyze(files: List[UploadFile] = File(...)) -> dict:
    """Analyse one or more CSV batches, processed in the order supplied.

    Order is the point: the brief requires the L2 messages to be processed
    after the L1 messages. Upload them in that order and the ledger will
    replay them in it, whatever the timestamps say.
    """
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"at most {MAX_FILES} files")
    payload_files = []
    for upload in files:
        text = _read_upload(await upload.read())
        if text.count("\n") > MAX_ROWS:
            raise HTTPException(413, f"{upload.filename} exceeds {MAX_ROWS} rows")
        payload_files.append((upload.filename or "batch.csv", text))
    return _analyse(payload_files, {
        "name": " + ".join(n for n, _ in payload_files),
        "kind": "uploaded batches, processed in order",
        "note": ("Masked in memory for this request; the raw text is never "
                 "stored and never leaves the process."),
    })


class AskIn(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=64)
    query: str = Field(..., min_length=1, max_length=500)
    confirmed: bool = False


@api.post("/v2/ask")
def l2_ask(payload: AskIn) -> dict:
    """Answer one question against an analysed corpus."""
    ledger = _session(payload.session_id)
    started = time.perf_counter()
    answer = Assistant(ledger).answer(payload.query, confirmed=payload.confirmed)
    out = answer.to_dict()
    out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return out


@api.get("/v2/policy")
def l2_policy() -> dict:
    """The privacy-routing policy table, independent of any uploaded data."""
    return {
        "policies": ROUTE.policy_table(),
        "operations": ROUTE.OPERATIONS,
        "note": ("send_to_external_service is denied for every message, "
                 "including messages with nothing sensitive in them. That is "
                 "not the policy's doing: this package contains no network "
                 "client and no outbound call, so the policy records the fact "
                 "rather than enforcing it."),
    }


class BenchIn(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=64)
    repeats: int = Field(15, ge=3, le=60)
    #: Corpus size at which to run the second measurement. An inverted index
    #: has nothing to prune on a few dozen documents, so a small corpus makes
    #: the optimised index look slower -- which is true, and is exactly why the
    #: comparison is run at two sizes instead of one.
    scale_to: int = Field(2000, ge=0, le=20000)


@api.post("/v2/benchmark")
def l2_benchmark(payload: BenchIn) -> dict:
    """Measure the baseline index against the optimised one, live.

    Same documents, same queries, same process, warm-up first. The full
    offline report (`scripts/benchmark.py`) adds quality metrics and a scaling
    run; this endpoint exists so the comparison can be reproduced on whatever
    corpus is actually loaded rather than quoted from a file.
    """
    from mint.retrieve import Document

    ledger = _session(payload.session_id)
    base = []
    for rec in ledger.records:
        group = ledger.group(rec.group_id) if rec.group_id else None
        text = rec.masked_text + (f" {group.title}" if group else "")
        base.append(Document(doc_id=rec.message_id, text=text,
                             indexable=rec.route.indexable))

    queries = [g.title for g in ledger.groups[:20]] or ["update"]
    model = load_semantic()

    def measure(docs: list) -> dict:
        out = {}
        for name, factory in (
            ("v1-exact-lexical", lambda: ExactLexicalIndex(docs)),
            ("v2-hybrid-pruned", lambda: HybridIndex(docs, model)),
        ):
            start = time.perf_counter()
            index = factory()
            build_ms = (time.perf_counter() - start) * 1000
            for q in queries[:3]:
                index.search(q, 10)
            samples = []
            for q in queries:
                for _ in range(payload.repeats):
                    t = time.perf_counter()
                    index.search(q, 10)
                    samples.append((time.perf_counter() - t) * 1000)
            samples.sort()
            out[name] = {
                "build_ms": round(build_ms, 2),
                "median_ms": round(statistics.median(samples), 4),
                "p95_ms": round(
                    samples[min(len(samples) - 1, int(0.95 * len(samples)))], 4),
                "in_memory_kb": round(index.memory_bytes() / 1024, 1),
                "documents": index.size,
                "excluded_by_routing": len(index.excluded),
            }
        return out

    results = measure(base)
    v1, v2 = results["v1-exact-lexical"], results["v2-hybrid-pruned"]

    scaled = None
    copies = 0
    if payload.scale_to and len(base) and payload.scale_to > len(base):
        copies = max(2, -(-payload.scale_to // len(base)))
        big = [Document(doc_id=(d.doc_id if c == 0 else f"C{c}_{d.doc_id}"),
                        text=d.text, indexable=d.indexable)
               for c in range(copies) for d in base]
        scaled = measure(big)
        scaled["documents"] = len(big)
        scaled["copies"] = copies

    reference = ExactLexicalIndex(base)
    optimised = HybridIndex(base, model)
    overlap = []
    for q in queries[:8]:
        ref = [h.doc_id for h in reference.search(q, 10)]
        got = [h.doc_id for h in optimised.search(q, 10)]
        if ref:
            overlap.append(len(set(ref) & set(got)) / len(ref))

    speedup = round(v1["median_ms"] / max(v2["median_ms"], 1e-9), 2)
    note = ("An inverted index has nothing to prune on a small corpus, so on a "
            "few dozen documents the optimised index can be slower than the "
            "full scan it replaces. That is why the same measurement is "
            "repeated on a larger corpus below."
            if speedup < 1.2 else
            "Candidate generation over posting lists replaces the full scan, "
            "so the advantage grows with the number of documents.")

    return {
        "documents": len(base),
        "indexed": v2["documents"],
        "excluded_by_privacy_routing": v2["excluded_by_routing"],
        "queries": len(queries),
        "repeats": payload.repeats,
        "results": results,
        "speedup": speedup,
        "memory_ratio": round(v2["in_memory_kb"] / max(v1["in_memory_kb"], 1e-9), 2),
        "ranking_fidelity_at_10": round(sum(overlap) / max(len(overlap), 1), 4),
        "scaled": scaled,
        "speedup_scaled": (round(
            scaled["v1-exact-lexical"]["median_ms"]
            / max(scaled["v2-hybrid-pruned"]["median_ms"], 1e-9), 2)
            if scaled else None),
        "note": note,
        "method": ("same documents, same queries, one process, warm-up pass "
                   "discarded, median and p95 over every timed call. The "
                   "larger corpus is this one replicated with distinct IDs, "
                   "which measures how the two designs diverge with size and "
                   "is not used for any quality number"),
    }


@api.delete("/v2/session/{session_id}")
def l2_forget(session_id: str) -> dict:
    """Drop an analysed corpus from memory immediately."""
    existed = _SESSIONS.pop(session_id, None) is not None
    return {"forgotten": existed, "sessions_held": len(_SESSIONS)}


@api.get("/v2/health")
def l2_health() -> dict:
    model = load_semantic()
    return {
        "status": "ok",
        "version": __version__,
        "semantic_model_loaded": model is not None,
        "semantic_dimensions": model.k if model else 0,
        "semantic_metadata": model.metadata if model else {},
        "sessions_held": len(_SESSIONS),
        "max_sessions": MAX_SESSIONS,
        "stores_raw_message_text": False,
        "sends_data_externally": False,
    }
