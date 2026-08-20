"""The L2 ledger: one chronological pass that produces every L2 artifact.

L1 was a pipeline -- read a row, mask it, classify it, extract from it, move
on. L2 cannot be, because half its questions are about *relationships between*
messages: which ones share a subject, which one superseded which, what the
current deadline is. So L2 keeps a ledger: a running account of every subject,
updated message by message in time order.

The ordering below is the whole design:

    read batch -> mask -> classify (L1) -> extract (L1) -> detect act ->
    assign to subject group -> route for privacy -> snapshot priorities ->
    (all batches done) -> build the retrieval index

Four properties are worth stating because each one is a decision:

**Batches are processed in the order given, not merged by timestamp.**
The brief says the L2 messages are processed after the L1 messages. Sorting the
combined file by timestamp would usually produce the same order, but "usually"
is not a guarantee, and a single out-of-order timestamp would silently rewrite
history. So rows are sorted within a batch and batches are concatenated.

**Priority is snapshotted at every batch boundary.**
That is what makes "the system should update the priority when a later message
changes the deadline, urgency or status" observable rather than merely true.
Each snapshot records what changed since the last one and whether a message
caused it or time simply passed.

**Masking still happens first.**
Unchanged from L1, and now load-bearing for more consumers: the act detector,
the grouper, the summariser and the retrieval index all receive masked text
because there is nothing else in the record to receive.

**Routing decides what enters the index, before the index is built.**
A blocked message is never a document. It cannot be retrieved by a query it
was never a candidate for.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import acts as A
from . import groups as G
from . import priority as P
from . import routing as R
from . import taxonomy as T
from .classifier import Classification, classify as classify_message
from .embed import SemanticModel, load_default as load_semantic
from .extract import ExtractedItem, extract as extract_item
from .model import TfidfLogisticModel, load_default as load_classifier
from .pipeline import REQUIRED_COLUMNS
from .retrieve import Document, HybridIndex
from .sensitive import HUMAN_LABEL, ScanResult, scan


@dataclass
class Batch:
    """One input file, kept identifiable so provenance survives the merge."""

    name: str
    kind: str                    # "l1" | "l2" | "demo"
    message_ids: List[str] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind,
                "messages": len(self.message_ids),
                "from": self.first_seen, "to": self.last_seen}


@dataclass
class LedgerRecord:
    """One message, fully processed. Deliberately has no raw-text field."""

    message_id: str
    timestamp: str
    sender: str
    batch: str
    batch_kind: str
    masked_text: str
    classification: Classification
    scan_result: ScanResult
    verdict: A.ActVerdict
    route: R.RouteDecision
    item: Optional[ExtractedItem] = None
    group_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "batch": self.batch,
            "batch_kind": self.batch_kind,
            "masked_text": self.masked_text,
            "category": self.classification.category,
            "confidence": self.classification.confidence,
            "needs_review": self.classification.needs_review,
            "act": self.verdict.act,
            "act_label": A.HUMAN_ACT.get(self.verdict.act, self.verdict.act),
            "act_frame": self.verdict.frame,
            "act_hedged": self.verdict.hedged,
            "subject_phrase": self.verdict.subject_phrase,
            "group_id": self.group_id,
            "item_id": self.item.item_id if self.item else None,
            "route": self.route.route,
            "risk": self.scan_result.overall_risk,
            "sensitivity_type": self.scan_result.primary_type,
            "indexable": self.route.indexable,
        }


@dataclass
class Snapshot:
    """Priorities as they stood at the end of one batch."""

    batch: str
    as_of: str
    decisions: Dict[str, P.PriorityDecision] = field(default_factory=dict)
    changes: List[Tuple[str, P.PriorityChange]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "batch": self.batch,
            "as_of": self.as_of,
            "counts": {band: sum(1 for d in self.decisions.values()
                                 if d.priority == band) for band in P.BANDS},
            "changes": [{"group_id": gid, **change.to_dict()}
                        for gid, change in self.changes],
        }


@dataclass
class Ledger:
    """Everything L2 produces, in one object."""

    records: List[LedgerRecord] = field(default_factory=list)
    groups: List[G.MessageGroup] = field(default_factory=list)
    items: List[ExtractedItem] = field(default_factory=list)
    batches: List[Batch] = field(default_factory=list)
    snapshots: List[Snapshot] = field(default_factory=list)
    priorities: Dict[str, P.PriorityDecision] = field(default_factory=dict)
    priority_history: Dict[str, List[P.PriorityChange]] = field(default_factory=dict)
    index: Optional[HybridIndex] = None
    #: The subject vocabulary the grouper used. Kept so the assistant can
    #: resolve a phrase in a question ("the project report") to a group with
    #: the same matcher, rather than a second, subtly different one.
    space: Optional[G.SubjectSpace] = None
    as_of: str = ""

    # -- lookups ------------------------------------------------------------
    def __post_init__(self) -> None:
        self._by_id: Dict[str, LedgerRecord] = {}
        self._group_by_id: Dict[str, G.MessageGroup] = {}
        self._item_by_id: Dict[str, ExtractedItem] = {}

    def reindex(self) -> None:
        self._by_id = {r.message_id: r for r in self.records}
        self._group_by_id = {g.group_id: g for g in self.groups}
        self._item_by_id = {i.item_id: i for i in self.items}

    def record(self, message_id: str) -> Optional[LedgerRecord]:
        return self._by_id.get(message_id)

    def group(self, group_id: str) -> Optional[G.MessageGroup]:
        return self._group_by_id.get(group_id)

    def resolve_subject(self, phrase: str):
        """Match a phrase from a question to a group, using the grouper's own
        matcher. Returns (group, score, shared_tokens) or None."""
        if not phrase or self.space is None:
            return None
        probe = G.signature(phrase)
        if not probe:
            return None
        best = None
        for group in self.groups:
            for candidate in [group.signature] + group.variants:
                score, tokens = self.space.match(probe, candidate)
                if score <= 0.0:
                    continue
                if best is None or score > best[1]:
                    best = (group, score, tokens)
        return best

    def group_of(self, message_id: str) -> Optional[G.MessageGroup]:
        rec = self.record(message_id)
        if rec is None or rec.group_id is None:
            return None
        return self.group(rec.group_id)

    def item(self, item_id: str) -> Optional[ExtractedItem]:
        return self._item_by_id.get(item_id)

    def routes(self) -> Dict[str, R.RouteDecision]:
        return {r.message_id: r.route for r in self.records}

    # -- Part 1 output ------------------------------------------------------
    def priority_records(self) -> List[dict]:
        """One record per actionable item, in the brief's shape.

        Priority is scored on the subject group, so every item extracted from
        the same subject carries the same band -- they are the same work seen
        from different messages. The record says which group it came from so
        that is visible rather than a coincidence.
        """
        out: List[dict] = []
        for rec in self.records:
            if rec.item is None:
                continue
            group = self.group(rec.group_id) if rec.group_id else None
            decision = self.priorities.get(rec.group_id or "")
            if decision is None:
                decision = self._solo_priority(rec)
            payload = decision.to_dict()
            payload.update({
                "message_id": rec.message_id,
                "item_id": rec.item.item_id,
                "item_type": rec.item.type,
                "item_title": rec.item.title,
                "group_id": rec.group_id,
                "group_status": group.status if group else None,
                "group_latest_deadline": group.latest_deadline if group else None,
                "item_own_deadline": rec.item.date,
                "shared_with_group": group is not None,
                "priority_history": [c.to_dict() for c in
                                     self.priority_history.get(rec.group_id or "", [])],
            })
            out.append(payload)
        return out

    def _solo_priority(self, rec: LedgerRecord) -> P.PriorityDecision:
        """Score an item whose subject could not be grouped.

        This happens when a message names nothing identifiable -- L1's "Let us
        meet sometime next week" extracts an event called "Meeting", which
        shares no distinguishing word with anything. Rather than force it into
        a group it does not belong in, it is scored alone from a one-member
        group built on the spot, and the answer says so.
        """
        stub = G.MessageGroup(
            group_id=f"SOLO_{rec.message_id}",
            title=rec.item.title if rec.item else rec.message_id,
            kind=rec.item.type if rec.item else "subject",
            signature=G.signature(rec.item.title if rec.item else ""),
            status=G.PENDING,
            status_reason="this message was not grouped with any other",
            first_seen=rec.timestamp, last_seen=rec.timestamp,
            confidence=0.55,
        )
        if rec.item and rec.item.type == "task":
            stub.latest_deadline = rec.item.date
            stub.latest_deadline_source = rec.message_id
        elif rec.item:
            stub.latest_date = rec.item.date
            stub.latest_time = rec.item.time
            stub.schedule_source = rec.message_id
        decision = P.score_group(
            stub, self.as_of, category=rec.classification.category,
            sensitive_high_risk=(rec.scan_result.overall_risk == "high"),
            senders=[rec.sender],
        )
        decision.group_id = None
        decision.needs_review = True
        return decision

    # -- Part 2 output ------------------------------------------------------
    def group_records(self) -> List[dict]:
        return [g.to_dict() for g in self.groups]

    # -- privacy output -----------------------------------------------------
    def routing_records(self) -> List[dict]:
        out = []
        for rec in self.records:
            payload = rec.route.to_dict()
            payload.update({
                "timestamp": rec.timestamp,
                "sender": rec.sender,
                "category": rec.classification.category,
                # Masked text is included only where seeing the mask is the
                # point -- messages that actually tripped a detector, minus the
                # blocked ones, whose masked form the sensitive report already
                # carries. For the ~90% of messages with nothing sensitive in
                # them the sentence adds nothing to a record of privacy
                # decisions and would republish the corpus into a file that
                # exists to document routing.
                "masked_text": (
                    rec.masked_text
                    if (rec.scan_result.is_sensitive
                        and rec.route.route != R.BLOCKED)
                    else None),
                "indexed": rec.route.indexable,
            })
            out.append(payload)
        return out

    # -- summary ------------------------------------------------------------
    def summary(self) -> dict:
        counts = {band: 0 for band in P.BANDS}
        for decision in self.priorities.values():
            counts[decision.priority] += 1
        statuses = {s: 0 for s in G.STATUSES}
        for g in self.groups:
            statuses[g.status] = statuses.get(g.status, 0) + 1
        routes = {r: 0 for r in R.ROUTES}
        for rec in self.records:
            routes[rec.route.route] += 1
        grouped = sum(len(g.members) for g in self.groups)
        return {
            "as_of": self.as_of,
            "total_messages": len(self.records),
            "batches": [b.to_dict() for b in self.batches],
            "groups": len(self.groups),
            "grouped_messages": grouped,
            "ungrouped_messages": len(self.records) - grouped,
            "group_status_counts": statuses,
            "group_priority_counts": counts,
            "items": len(self.items),
            "route_counts": routes,
            "excluded_from_index": len(self.index.excluded) if self.index else 0,
            "index": self.index.profile() if self.index else None,
            "contested_groups": sum(1 for g in self.groups if g.contested),
            "groups_with_conflicts": sum(1 for g in self.groups if g.conflicts),
            "mean_group_confidence": (
                round(sum(g.confidence for g in self.groups) / len(self.groups), 4)
                if self.groups else 0.0
            ),
        }


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def read_batch(source: str | Path | Iterable[str]) -> List[dict]:
    """Read one CSV into rows sorted chronologically *within* the batch."""
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8-sig")
    elif isinstance(source, str):
        looks_like_path = "\n" not in source and len(source) < 255
        if looks_like_path and Path(source).exists():
            text = Path(source).read_text(encoding="utf-8-sig")
        else:
            text = source
    else:
        text = "".join(source)

    rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
    if not rows:
        raise ValueError("the CSV contains no data rows")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(
            f"missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(REQUIRED_COLUMNS)}")
    rows.sort(key=lambda r: ((r.get("timestamp") or ""), r.get("message_id") or ""))
    return rows


def _kind_for(name: str, position: int) -> str:
    lowered = name.lower()
    if "demo" in lowered:
        return "demo"
    if "l2" in lowered:
        return "l2"
    return "l1" if position == 0 else "l2"


# --------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------
def run(
    sources: Sequence[Tuple[str, object]],
    classifier: Optional[TfidfLogisticModel] = None,
    semantic: Optional[SemanticModel] = None,
    build_index: bool = True,
) -> Ledger:
    """Process one or more message batches into a ledger.

    `sources` is an ordered sequence of ``(name, csv_source)`` pairs. Order is
    authoritative: batch two is processed after batch one whatever the
    timestamps say.
    """
    classifier = classifier if classifier is not None else load_classifier()
    semantic = semantic if semantic is not None else load_semantic()

    ledger = Ledger()
    ledger.reindex()

    # ---- pass 0: read everything and run the L1 pipeline over it ----------
    staged: List[Tuple[Batch, List[dict]]] = []
    for position, (name, source) in enumerate(sources):
        rows = read_batch(source)
        batch = Batch(name=name, kind=_kind_for(name, position))
        batch.message_ids = [r["message_id"] for r in rows]
        batch.first_seen = rows[0].get("timestamp", "")
        batch.last_seen = rows[-1].get("timestamp", "")
        staged.append((batch, rows))
        ledger.batches.append(batch)

    task_n = event_n = 0
    prepared: List[Tuple[Batch, List[Tuple[LedgerRecord, G.SubjectInput]]]] = []
    all_inputs: List[G.SubjectInput] = []

    for batch, rows in staged:
        prepared_rows: List[Tuple[LedgerRecord, G.SubjectInput]] = []
        for row in rows:
            # --- mask first; the raw text goes no further than this line ----
            scan_result = scan(row["message"])
            masked = scan_result.masked_text

            cls = classify_message(
                message_id=row["message_id"], masked_text=masked,
                sender=row.get("sender", ""), scan_result=scan_result,
                model=classifier)

            route = R.route_message(row["message_id"], scan_result)

            item = None
            if R.EXTRACT in route.allowed_operations:
                if cls.category == T.MEETING_EVENT:
                    event_n += 1
                    item = extract_item(row["message_id"], masked,
                                        row.get("sender", ""),
                                        row.get("timestamp", ""), cls.category,
                                        event_n, ())
                elif cls.category == T.ACTION_REQUIRED:
                    task_n += 1
                    item = extract_item(row["message_id"], masked,
                                        row.get("sender", ""),
                                        row.get("timestamp", ""), cls.category,
                                        task_n, ())

            verdict = A.from_l1_extraction(A.detect(masked),
                                           item.type if item else None)

            rec = LedgerRecord(
                message_id=row["message_id"], timestamp=row.get("timestamp", ""),
                sender=row.get("sender", ""), batch=batch.name,
                batch_kind=batch.kind, masked_text=masked, classification=cls,
                scan_result=scan_result, verdict=verdict, route=route, item=item)
            ledger.records.append(rec)
            if item is not None:
                ledger.items.append(item)

            subject_input = G.SubjectInput(
                message_id=rec.message_id, timestamp=rec.timestamp,
                sender=rec.sender, masked_text=masked, category=cls.category,
                verdict=verdict,
                item_id=item.item_id if item else None,
                item_title=item.title if item else None,
                item_frame=item.extraction_frame if item else "",
                item_type=item.type if item else None,
                item_date=item.date if item else None,
                item_time=item.time if item else None)
            prepared_rows.append((rec, subject_input))
            all_inputs.append(subject_input)
        prepared.append((batch, prepared_rows))

    # ---- pass 1: subject vocabulary --------------------------------------
    # Only the primary signature of each message feeds the IDF statistics. A
    # variant is another way of naming the *same* message's subject -- L1's
    # extracted title alongside the L2 frame's capture -- and counting it
    # separately would make a word look like it names two subjects when it
    # names one. It is still used for matching; it just does not vote on how
    # informative its own tokens are.
    signatures = []
    for inp in all_inputs:
        sig, _variants, _ = G.subject_for(inp)
        if sig:
            signatures.append(sig)
    space = G.SubjectSpace(signatures)
    ledger.space = space
    builder = G.GroupBuilder(space)

    # ---- pass 2: chronological assignment, with a snapshot per batch ------
    previous: Dict[str, P.PriorityDecision] = {}
    for batch, prepared_rows in prepared:
        touched: Dict[str, List[str]] = {}
        for rec, subject_input in prepared_rows:
            group = builder.add(subject_input)
            if group is not None:
                rec.group_id = group.group_id
                touched.setdefault(group.group_id, []).append(rec.message_id)

        ledger.as_of = batch.last_seen
        snapshot = Snapshot(batch=batch.name, as_of=batch.last_seen)
        for group in builder.groups:
            group.confidence = G.score_group(group)
            senders = {m.sender for m in group.members}
            high_risk = any(
                r.scan_result.overall_risk == "high"
                for r in ledger.records
                if r.message_id in set(group.message_ids))
            decision = P.score_group(
                group, batch.last_seen, category=group.kind,
                sensitive_high_risk=high_risk, senders=sorted(senders))
            snapshot.decisions[group.group_id] = decision
            change = P.track(previous.get(group.group_id), decision,
                             touched.get(group.group_id, []))
            if change is not None:
                snapshot.changes.append((group.group_id, change))
                ledger.priority_history.setdefault(
                    group.group_id, []).append(change)
        previous = snapshot.decisions
        ledger.snapshots.append(snapshot)

    ledger.groups = builder.finalise()
    ledger.priorities = previous
    ledger.reindex()

    # ---- pass 3: the retrieval index -------------------------------------
    if build_index:
        docs = []
        for rec in ledger.records:
            group = ledger.group(rec.group_id) if rec.group_id else None
            text = rec.masked_text
            if group is not None:
                text = f"{text} {group.title}"
            docs.append(Document(
                doc_id=rec.message_id, text=text,
                indexable=rec.route.indexable,
                meta={"group_id": rec.group_id,
                      "category": rec.classification.category,
                      "act": rec.verdict.act}))
        ledger.index = HybridIndex(docs, semantic)

    return ledger


def write_outputs(ledger: Ledger, out_dir: str | Path,
                  answers: Optional[List[dict]] = None) -> List[Path]:
    """Write the L2 structured output files. All contain masked text only.

    Masking happened before any of this data existed, so there is no redaction
    step here that could be forgotten. `tests/test_no_leakage.py` asserts the
    property against every file this writes.
    """
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    payloads: Dict[str, object] = {
        "priority_decisions.json": ledger.priority_records(),
        "message_groups.json": ledger.group_records(),
        "privacy_routing.json": {
            "policy_table": R.policy_table(),
            "decisions": ledger.routing_records(),
        },
        "priority_snapshots.json": [s.to_dict() for s in ledger.snapshots],
        "l2_summary.json": ledger.summary(),
    }
    if answers is not None:
        payloads["assistant_answers.json"] = answers

    for name, payload in payloads.items():
        path = out / name
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        written.append(path)

    # Two flat CSVs as well: easier to eyeball, and easier to show on screen
    # in a recording than a nested JSON document.
    groups_csv = out / "message_groups.csv"
    with groups_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group_id", "title", "kind", "status", "priority",
                         "latest_deadline", "latest_schedule", "messages",
                         "chases", "conflicts", "contested", "confidence",
                         "related_message_ids", "summary"])
        for group in ledger.groups:
            decision = ledger.priorities.get(group.group_id)
            schedule = " ".join(x for x in (group.latest_date, group.latest_time)
                                if x)
            writer.writerow([
                group.group_id, group.title, group.kind, group.status,
                decision.priority if decision else "",
                group.latest_deadline or group.pending_relative_deadline or "",
                schedule, len(group.members), group.chase_count,
                len(group.conflicts), group.contested, group.confidence,
                " ".join(group.message_ids), group.summary,
            ])
    written.append(groups_csv)

    priority_csv = out / "priority_decisions.csv"
    rows = ledger.priority_records()
    if rows:
        fields = ["message_id", "item_id", "group_id", "item_title", "priority",
                  "confidence", "score", "group_status",
                  "group_latest_deadline", "as_of", "reason"]
        with priority_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields + ["signals"])
            writer.writeheader()
            for row in rows:
                flat = {k: row.get(k) for k in fields}
                flat["signals"] = " ".join(row.get("signals", []))
                writer.writerow(flat)
        written.append(priority_csv)

    return written
