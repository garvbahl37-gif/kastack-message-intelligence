"""End-to-end orchestration.

The ordering here is the security design of the whole system:

    read row  ->  scan & mask  ->  drop raw text  ->  classify  ->  extract

Masking happens *before* any other component runs, and the raw message is never
stored on the record that the rest of the program passes around. That is why no
downstream stage -- classifier, extractor, JSON writer, HTTP response, log
line -- needs its own redaction logic: by the time it runs, there is nothing
left to redact. `tests/test_no_leakage.py` asserts the property end to end.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from . import taxonomy as T
from .classifier import Classification, classify as classify_message
from .extract import ExtractedItem, extract as extract_item
from .model import TfidfLogisticModel, load_default
from .sensitive import HUMAN_LABEL, ScanResult, scan

REQUIRED_COLUMNS = ("message_id", "timestamp", "sender", "message")


@dataclass
class ProcessedMessage:
    """One message after masking. Deliberately has no `raw` field."""

    message_id: str
    timestamp: str
    sender: str
    masked_text: str
    classification: Classification
    scan_result: ScanResult
    item: Optional[ExtractedItem] = None

    def to_row(self) -> dict:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "masked_text": self.masked_text,
            "category": self.classification.category,
            "confidence": self.classification.confidence,
            "reason": self.classification.reason,
            "needs_review": self.classification.needs_review,
            "is_sensitive": self.scan_result.is_sensitive,
            "risk": self.scan_result.overall_risk,
        }


@dataclass
class PipelineResult:
    messages: List[ProcessedMessage] = field(default_factory=list)
    items: List[ExtractedItem] = field(default_factory=list)
    roster: List[str] = field(default_factory=list)

    # -- Part 1 -------------------------------------------------------------
    def classifications(self) -> List[dict]:
        return [m.classification.to_dict() for m in self.messages]

    def category_counts(self) -> Dict[str, int]:
        counts = {c: 0 for c in T.CATEGORIES}
        for m in self.messages:
            counts[m.classification.category] = counts.get(
                m.classification.category, 0
            ) + 1
        return counts

    # -- Part 2 -------------------------------------------------------------
    def tasks(self) -> List[ExtractedItem]:
        return [i for i in self.items if i.type == "task"]

    def events(self) -> List[ExtractedItem]:
        return [i for i in self.items if i.type == "event"]

    # -- Part 3 -------------------------------------------------------------
    def sensitive_report(self) -> List[dict]:
        """The Part-3 record. Contains masked text only, by construction."""
        out = []
        for m in self.messages:
            sr = m.scan_result
            if not sr.is_sensitive:
                continue
            out.append(
                {
                    "message_id": m.message_id,
                    "sensitivity_type": sr.primary_type,
                    "sensitivity_types": sr.types,
                    "sensitivity_label": HUMAN_LABEL.get(sr.primary_type or "", ""),
                    "risk": sr.overall_risk,
                    "masked_text": sr.masked_text,
                    "recommended_action": sr.overall_action,
                    "reason": "; ".join(
                        dict.fromkeys(f.reason for f in sr.findings)
                    ),
                    "detector_count": len(sr.findings),
                    "timestamp": m.timestamp,
                    "sender": m.sender,
                }
            )
        return out

    # -- summary ------------------------------------------------------------
    def summary(self) -> dict:
        sens = self.sensitive_report()
        risk_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for s in sens:
            risk_counts[s["risk"]] = risk_counts.get(s["risk"], 0) + 1
            type_counts[s["sensitivity_type"]] = (
                type_counts.get(s["sensitivity_type"], 0) + 1
            )
        confidences = [m.classification.confidence for m in self.messages]
        return {
            "total_messages": len(self.messages),
            "date_range": {
                "from": self.messages[0].timestamp if self.messages else None,
                "to": self.messages[-1].timestamp if self.messages else None,
            },
            "categories": self.category_counts(),
            "tasks_extracted": len(self.tasks()),
            "events_extracted": len(self.events()),
            "items_with_unresolved_fields": sum(
                1 for i in self.items if i.unresolved_fields
            ),
            "sensitive_messages": len(sens),
            "sensitive_by_risk": risk_counts,
            "sensitive_by_type": type_counts,
            "flagged_for_review": sum(
                1 for m in self.messages if m.classification.needs_review
            ),
            "mean_confidence": (
                round(sum(confidences) / len(confidences), 4) if confidences else 0.0
            ),
            "senders": sorted(self.roster),
        }


def read_rows(source: str | Path | io.StringIO | Iterable[str]) -> List[dict]:
    """Read a message CSV and return rows sorted chronologically.

    Chronological order is a requirement of the brief, so it is enforced here
    rather than inherited from whatever order the file happened to be in.
    """
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8-sig")
    elif isinstance(source, str):
        # A str is CSV *content* unless it plausibly names a file. The length
        # and newline guards matter: without them this asks the filesystem
        # whether a multi-kilobyte CSV payload is a filename, which raises
        # ENAMETOOLONG on Linux as soon as the content exceeds NAME_MAX.
        looks_like_path = "\n" not in source and len(source) < 255
        if looks_like_path and Path(source).exists():
            text = Path(source).read_text(encoding="utf-8-sig")
        else:
            text = source
    else:
        text = "".join(source)

    # Strip a UTF-8 BOM however the text arrived: Excel writes one, and it
    # otherwise turns the first header into "﻿message_id".
    rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))

    if not rows:
        raise ValueError("the CSV contains no data rows")

    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(
            f"missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(REQUIRED_COLUMNS)}"
        )

    rows.sort(key=lambda r: ((r.get("timestamp") or ""), r.get("message_id") or ""))
    return rows


def run(
    source: str | Path | io.StringIO | Iterable[str],
    model: Optional[TfidfLogisticModel] = None,
    roster: Optional[Sequence[str]] = None,
) -> PipelineResult:
    """Run the full pipeline over a message CSV."""
    rows = read_rows(source)
    if model is None:
        model = load_default()

    # The participant roster is derived from the senders present in the file,
    # so person-matching adapts to whatever dataset is loaded.
    people = sorted({(r.get("sender") or "").strip() for r in rows if r.get("sender")})
    known = list(roster) if roster is not None else [
        p for p in people if p.lower() not in {"promotions", "hr team", "project lead"}
    ]

    result = PipelineResult(roster=people)
    task_n = event_n = 0

    for row in rows:
        # --- mask first; the raw text goes no further than this line -------
        scan_result = scan(row["message"])
        masked = scan_result.masked_text

        cls = classify_message(
            message_id=row["message_id"],
            masked_text=masked,
            sender=row.get("sender", ""),
            scan_result=scan_result,
            model=model,
        )

        item = None
        if cls.category == T.MEETING_EVENT:
            event_n += 1
            item = extract_item(
                row["message_id"], masked, row.get("sender", ""),
                row.get("timestamp", ""), cls.category, event_n, known,
            )
        elif cls.category == T.ACTION_REQUIRED:
            task_n += 1
            item = extract_item(
                row["message_id"], masked, row.get("sender", ""),
                row.get("timestamp", ""), cls.category, task_n, known,
            )

        pm = ProcessedMessage(
            message_id=row["message_id"],
            timestamp=row.get("timestamp", ""),
            sender=row.get("sender", ""),
            masked_text=masked,
            classification=cls,
            scan_result=scan_result,
            item=item,
        )
        result.messages.append(pm)
        if item is not None:
            result.items.append(item)

    return result


def write_outputs(result: PipelineResult, out_dir: str | Path) -> List[Path]:
    """Write the four structured output files. All contain masked text only."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    files = {
        "classified_messages.json": result.classifications(),
        "extracted_items.json": [i.to_dict() for i in result.items],
        "sensitive_report.json": result.sensitive_report(),
        "summary.json": result.summary(),
    }
    for name, payload in files.items():
        path = out / name
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        written.append(path)

    # A flat CSV as well -- easier to eyeball and to show on screen.
    csv_path = out / "classified_messages.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        rows = [m.to_row() for m in result.messages]
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    written.append(csv_path)

    return written
