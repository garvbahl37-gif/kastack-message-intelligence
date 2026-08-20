"""The question-answering assistant.

Part 3 of the brief: answer questions over the L1 classifications, the
extracted items, the sensitive-information results, the priorities and the
subject groups -- always with supporting message IDs, relevance scores and an
explanation of why that evidence was chosen, and never with an unsupported
answer.

The central design decision is that **retrieval is not the answer path.**

Most of the questions in the brief are not "find me a document". "Which
critical or high-priority tasks are still pending?" is a filter over the
priority table. "What meetings were rescheduled?" is a filter over group
status. "What deadlines have changed?" is a read of the deadline history.
Answering those by searching text and hoping the right sentences come back
would be a worse system that fails in a more interesting way: it would return
plausible messages and get the count wrong.

So a question is first routed to an **intent**. If it names something the
ledger tracks structurally, it is answered from the ledger, and the answer is
exact. Retrieval handles what is left -- open-ended "show me messages about X"
-- and also runs alongside every structured answer so a relevance score is
always available for the reader to judge.

Refusal is a first-class outcome
--------------------------------
"The assistant must not generate an unsupported answer" is a requirement, so
there is a gate rather than a hope. An answer is withheld when the retrieval
evidence is too weak, when the referent cannot be resolved, or -- the
interesting case -- when the best-matching messages are themselves *questions
nobody answered*. "Was the compliance form approved by the finance director?"
has a near-perfect lexical match in this corpus, because another message asks
exactly the same thing. Returning that message as the answer would be the most
confident possible way to be wrong. The assistant instead reports what it
found: someone asked, and nothing in the corpus replies.

Privacy
-------
Every answer is routed through `mint/routing.py` before it is returned.
Blocked messages were never indexed, so they cannot be retrieved; when a
question is *about* them the answer names them by ID and sensitivity type and
withholds the content. Messages carrying private contact details are gated
behind an explicit confirmation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from . import groups as G
from . import priority as P
from . import routing as R
from .ledger import Ledger, LedgerRecord

#: Below this retrieval score there is nothing worth calling evidence.
MIN_EVIDENCE = 0.22

#: Retrieval hits attached to every answer so a relevance score is always shown.
CONTEXT_HITS = 5

MESSAGE_ID = re.compile(r"\b((?:MSG|DEMO|TASK|EVENT|GROUP)_[A-Za-z0-9]+)\b",
                        re.IGNORECASE)


@dataclass
class Evidence:
    """One piece of supporting evidence, and why it was selected."""

    message_id: str
    score: float
    source: str                 # "structured" | "retrieval"
    why: str
    masked_text: Optional[str] = None
    withheld: bool = False
    group_id: Optional[str] = None
    item_id: Optional[str] = None
    timestamp: str = ""
    sender: str = ""

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "relevance_score": round(self.score, 4),
            "evidence_source": self.source,
            "why_selected": self.why,
            "masked_text": self.masked_text,
            "content_withheld": self.withheld,
            "group_id": self.group_id,
            "item_id": self.item_id,
            "timestamp": self.timestamp,
            "sender": self.sender,
        }


@dataclass
class Answer:
    query: str
    intent: str
    answer: str
    reason: str
    confidence: float
    sufficient: bool = True
    supporting_message_ids: List[str] = field(default_factory=list)
    group_ids: List[str] = field(default_factory=list)
    item_ids: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    retrieved: List[dict] = field(default_factory=list)
    route: Optional[R.AnswerRoute] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "intent": self.intent,
            "answer": self.answer,
            "sufficient_evidence": self.sufficient,
            "supporting_message_ids": self.supporting_message_ids,
            "group_ids": self.group_ids,
            "item_ids": self.item_ids,
            "reason": self.reason,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "retrieved": self.retrieved,
            "privacy_route": self.route.to_dict() if self.route else None,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Intent routing
# --------------------------------------------------------------------------
def _c(expr: str):
    return re.compile(expr, re.IGNORECASE)


#: Ordered. The first frame that matches wins, so the specific questions are
#: listed before the general ones -- "which tasks were completed or cancelled"
#: has to be tried before "which tasks ...".
INTENT_FRAMES: List[Tuple[str, object]] = [
    ("blocked_messages", _c(
        r"\b(?:blocked?|must\s+be\s+blocked|not\s+be\s+(?:sent|shared)|"
        r"external\s+processing|do\s+not\s+send)\b")),
    ("needs_confirmation", _c(
        r"\b(?:require|requires|requiring|need|needs)\s+(?:a\s+)?confirmation|"
        r"\bconfirmation\s+before\b|\bask\s+for\s+confirmation\b")),
    ("became_priority", _c(
        r"\bbecame\s+(critical|high|medium|low)\b|"
        r"\bnewly\s+(critical|high)\b|\bescalated\b|"
        r"\bwhich\s+.*\bbecame\b")),
    ("explain_priority", _c(
        r"\bwhy\s+(?:was|is|were)\b.*\b(?:critical|high|medium|low|priority|"
        r"marked|flagged)\b|\bexplain\s+the\s+priority\b")),
    ("deadline_changes", _c(
        r"\bdeadlines?\s+(?:have\s+)?(?:changed|moved|shifted|been\s+(?:changed|"
        r"moved|extended))\b|\bwhat\s+deadlines?\b|\bchanged\s+deadlines?\b")),
    ("conflicts", _c(
        r"\bconflict(?:s|ing)?\b|\bcontradict\w*\b|\bdisagree\w*\b|"
        r"\buncertain\s+deadlines?\b|\bambiguous\b")),
    ("rescheduled", _c(
        r"\breschedul\w+\b|\bmoved\s+to\s+a\s+new\b|\bnew\s+schedule\b|"
        r"\bchanged\s+(?:its\s+)?(?:time|slot|schedule)\b")),
    # Before by_status, and by_status no longer matches the bare verb
    # "complete". "What tasks should I complete today?" is a question about
    # what is outstanding, and reading its verb as a status made the assistant
    # answer it with a list of finished work.
    ("due_today", _c(
        r"\b(?:due|complete|do|finish|work\s+on)\b.*\b(?:today|now|right\s+now)\b|"
        r"\btoday'?s?\s+tasks?\b|\bwhat\s+should\s+i\s+do\b")),
    ("by_status", _c(
        r"\b(?:completed|finished|cancelled|canceled|done|"
        r"(?:are|is|been)\s+complete)\b")),
    ("pending_by_priority", _c(
        r"\b(?:critical|high|urgent|top)[\s-]*(?:or\s+high[\s-]*)?"
        r"(?:priority)?\b.*\b(?:pending|outstanding|open|still|left|remain)\b|"
        r"\b(?:pending|outstanding|open)\b.*\b(?:critical|high|priority)\b|"
        r"\bwhat\s+is\s+(?:still\s+)?(?:pending|outstanding)\b")),
    ("latest_status", _c(
        r"\blatest\s+status\b|\bcurrent\s+status\b|\bstatus\s+of\b|"
        r"\bwhere\s+(?:do|does)\s+.*\bstand\b|\bwhat\s+happened\s+to\b")),
    # Placed after every structured intent and before the retrieval fallback.
    # "Are there any conflicting messages?" is also a yes/no question, but it
    # is one the ledger can answer, so `conflicts` claims it first. What
    # reaches here is a yes/no question about something nothing tracks.
    ("yes_no_verification", _c(
        r"^\s*(?:was|were|is|are|did|does|do|has|have|had|will|would|"
        r"should)\b.*\?\s*$")),
    ("messages_about", _c(
        r"\bshow\s+(?:me\s+)?(?:all\s+)?messages?\b|\bmessages?\s+(?:related|"
        r"about|regarding|concerning)\b|\ball\s+messages?\s+for\b|"
        r"\bfind\s+(?:me\s+)?messages?\b")),
]

_ABOUT = _c(
    r"\b(?:about|regarding|concerning|related\s+to|for|of|on)\s+"
    r"(?:the\s+)?(?P<subject>.+?)\s*[?.]?$")

_BAND = _c(r"\b(critical|high|medium|low)\b")
_STATUS_WORD = _c(r"\b(completed|complete|finished|cancelled|canceled|"
                  r"pending|rescheduled|unclear)\b")
_DEMO_SCOPE = _c(r"\b(?:demo|test)\s+(?:data|messages?|batch|set)\b|"
                 r"\bin\s+the\s+demo\b")


@dataclass
class Intent:
    name: str
    subject: Optional[str] = None
    message_ids: List[str] = field(default_factory=list)
    bands: List[str] = field(default_factory=list)
    statuses: List[str] = field(default_factory=list)
    demo_scope: bool = False


def route_intent(query: str) -> Intent:
    """Decide what kind of question this is, and what it is about."""
    ids = [m.group(1).upper() for m in MESSAGE_ID.finditer(query)]
    demo = bool(_DEMO_SCOPE.search(query)) or any(i.startswith("DEMO_")
                                                  for i in ids)
    bands = [b.lower() for b in _BAND.findall(query)]
    statuses = []
    for word in _STATUS_WORD.findall(query):
        w = word.lower()
        w = {"complete": "completed", "finished": "completed",
             "canceled": "cancelled"}.get(w, w)
        if w not in statuses:
            statuses.append(w)

    name = "open_search"
    for candidate, pattern in INTENT_FRAMES:
        if pattern.search(query):
            name = candidate
            break

    subject = None
    m = _ABOUT.search(query)
    if m:
        subject = m.group("subject").strip()
        # Strip the tail of "...related to the project report" style phrasing
        # that the frame's own vocabulary contributed.
        subject = re.sub(r"^(?:the\s+)?(?:task|item|meeting|event)\s+", "",
                         subject, flags=re.IGNORECASE).strip()

    return Intent(name=name, subject=subject or None, message_ids=ids,
                  bands=bands, statuses=statuses, demo_scope=demo)


# --------------------------------------------------------------------------
# The assistant
# --------------------------------------------------------------------------
class Assistant:
    """Answers questions against a `Ledger`. Holds no state of its own."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.routes = ledger.routes()

    # -- helpers ------------------------------------------------------------
    def _record(self, message_id: str) -> Optional[LedgerRecord]:
        return self.ledger.record(message_id)

    def _text_for(self, message_id: str) -> Tuple[Optional[str], bool]:
        """Masked text if the routing layer permits quoting it."""
        rec = self._record(message_id)
        if rec is None:
            return None, True
        if not rec.route.quotable:
            return None, True
        return rec.masked_text, False

    def _evidence(self, message_id: str, score: float, source: str,
                  why: str) -> Evidence:
        rec = self._record(message_id)
        text, withheld = self._text_for(message_id)
        return Evidence(
            message_id=message_id, score=score, source=source, why=why,
            masked_text=text, withheld=withheld,
            group_id=rec.group_id if rec else None,
            item_id=(rec.item.item_id if rec and rec.item else None),
            timestamp=rec.timestamp if rec else "",
            sender=rec.sender if rec else "")

    def _context(self, query: str, k: int = CONTEXT_HITS) -> List[dict]:
        if self.ledger.index is None:
            return []
        return [h.to_dict() for h in self.ledger.index.search(query, k)]

    def _active(self) -> List[G.MessageGroup]:
        return [g for g in self.ledger.groups if g.status not in G.TERMINAL]

    def _in_scope(self, group: G.MessageGroup, intent: Intent) -> bool:
        if not intent.demo_scope:
            return True
        demo_ids = {r.message_id for r in self.ledger.records
                    if r.batch_kind == "demo"}
        return any(m.message_id in demo_ids for m in group.members)

    def _latest_batch(self) -> Optional[str]:
        return self.ledger.batches[-1].name if self.ledger.batches else None

    def _latest_ids(self) -> set:
        if not self.ledger.batches:
            return set()
        return set(self.ledger.batches[-1].message_ids)

    def _batch_note(self, message_ids: Sequence[str]) -> str:
        """Name the subset that arrived in the newest batch.

        Several of the brief's questions are naturally asked about "the latest
        messages" without saying so. Answering over the whole corpus is
        correct, but burying the two IDs the user just loaded inside a list of
        twenty-five is not useful. So the complete answer is given, and the
        newest batch is called out inside it.
        """
        latest = self._latest_ids()
        recent = [m for m in message_ids if m in latest]
        if not recent or len(recent) == len(message_ids):
            return ""
        return (f" In the most recent batch ({self._latest_batch()}) the "
                f"matching message(s) are {', '.join(recent)}.")

    # -- the entry point ----------------------------------------------------
    def answer(self, query: str, confirmed: bool = False) -> Answer:
        intent = route_intent(query)
        handler = getattr(self, f"_answer_{intent.name}", self._answer_open_search)
        result = handler(query, intent)
        result.retrieved = self._context(query)
        result.supporting_message_ids = [e.message_id for e in result.evidence]
        result.route = R.route_answer(result.supporting_message_ids,
                                      self.routes, confirmed=confirmed)
        if result.route.route == R.CONFIRM_REQUIRED:
            result.notes.append(result.route.rationale)
        return result

    # -- structured intents -------------------------------------------------
    def _answer_blocked_messages(self, query: str, intent: Intent) -> Answer:
        rows = [r for r in self.ledger.records if r.route.route == R.BLOCKED]
        if intent.demo_scope:
            rows = [r for r in rows if r.batch_kind == "demo"]
        evidence = [
            self._evidence(
                r.message_id, 1.0, "structured",
                f"routing policy {r.route.policy_id} classifies this as "
                f"{', '.join(r.route.sensitivity_types)} at "
                f"{r.scan_result.overall_risk} risk")
            for r in rows]
        if not rows:
            return Answer(query, "blocked_messages",
                          "No message in scope is blocked.",
                          "no message matched a do-not-store policy", 0.9,
                          evidence=[])
        listed = ", ".join(r.message_id for r in rows)
        types = sorted({t for r in rows for t in r.route.sensitivity_types})
        return Answer(
            query, "blocked_messages",
            f"{len(rows)} message(s) are blocked from any downstream or "
            f"external use: {listed}. They carry "
            f"{', '.join(t.replace('_', ' ') for t in types)}. Their content "
            f"is withheld here and they were never added to the search index, "
            f"so no query can retrieve them."
            + self._batch_note([r.message_id for r in rows]),
            reason=("selected from the privacy-routing table rather than by "
                    "search: every message is routed at ingest, so this is the "
                    "complete set rather than the top-scoring matches"),
            confidence=0.95, evidence=evidence,
            notes=["Content withheld by policy P1-CREDENTIAL."])

    def _answer_needs_confirmation(self, query: str, intent: Intent) -> Answer:
        rows = [r for r in self.ledger.records
                if r.route.route == R.CONFIRM_REQUIRED]
        if intent.demo_scope:
            rows = [r for r in rows if r.batch_kind == "demo"]
        if not rows:
            return Answer(query, "needs_confirmation",
                          "No message in scope requires confirmation.",
                          "no message matched an ask-for-confirmation policy",
                          0.9)
        evidence = [
            self._evidence(r.message_id, 1.0, "structured",
                           r.route.confirmation_prompt or r.route.rationale)
            for r in rows]
        return Answer(
            query, "needs_confirmation",
            f"{len(rows)} message(s) need explicit confirmation before their "
            f"content is quoted or exported: "
            f"{', '.join(r.message_id for r in rows)}. They contain private "
            f"contact details, which are searchable in masked form but gated "
            f"for anything further."
            + self._batch_note([r.message_id for r in rows]),
            reason=("selected from the privacy-routing table; policy P3-CONTACT "
                    "applies to every message whose L1 recommendation was "
                    "ask_for_confirmation"),
            confidence=0.94, evidence=evidence)

    def _answer_became_priority(self, query: str, intent: Intent) -> Answer:
        target = intent.bands[0] if intent.bands else P.CRITICAL
        batch = self._latest_batch()
        snapshot = self.ledger.snapshots[-1] if self.ledger.snapshots else None
        if snapshot is None:
            return self._insufficient(query, "became_priority",
                                      "no priority snapshots were recorded")

        crossed = [(gid, c) for gid, c in snapshot.changes
                   if c.new == target and c.previous != target
                   and c.trigger == "message"]
        escalated = [(gid, c) for gid, c in snapshot.changes
                     if c.new == target and c.trigger == "message"
                     and c.delta > 0]
        by_time = [(gid, c) for gid, c in snapshot.changes
                   if c.new == target and c.trigger == "elapsed_time"]

        picked = crossed or escalated
        if not picked:
            note = (f"{len(by_time)} subject(s) reached {target} in this batch "
                    f"because the reference time advanced past their deadline, "
                    f"not because of a message."
                    if by_time else "")
            return self._insufficient(
                query, "became_priority",
                f"no subject moved to {target} in {batch} as a result of a "
                f"message" + (f". {note}" if note else ""))

        evidence: List[Evidence] = []
        lines: List[str] = []
        for gid, change in picked:
            group = self.ledger.group(gid)
            decision = snapshot.decisions.get(gid)
            for mid in change.trigger_message_ids:
                evidence.append(self._evidence(
                    mid, 1.0, "structured",
                    f"this message arrived in {batch} and is what moved "
                    f"{gid} to {change.new}"))
            verb = ("crossed into" if change.previous != target
                    else "escalated further within")
            lines.append(
                f"{group.title} ({gid}, {', '.join(group.item_ids[:3]) or 'no item'}) "
                f"{verb} {target}: score {change.previous_score} -> "
                f"{change.score} on {change.trigger_message_ids}. "
                + (decision.reason if decision else ""))

        extra = ""
        if by_time and not crossed:
            extra = (f" No subject crossed into {target} from a lower band in "
                     f"this batch; {len(by_time)} reached it only because the "
                     f"reference time advanced past a deadline, which is "
                     f"reported separately because nobody escalated them.")
        return Answer(
            query, "became_priority", " ".join(lines) + extra,
            reason=("taken from the priority snapshot recorded at the end of "
                    f"{batch}, comparing it against the previous batch's "
                    "snapshot and keeping only changes a message caused"),
            confidence=0.9, evidence=evidence,
            group_ids=[gid for gid, _ in picked],
            item_ids=[i for gid, _ in picked
                      for i in (self.ledger.group(gid).item_ids or [])][:8])

    def _answer_explain_priority(self, query: str, intent: Intent) -> Answer:
        group = self._resolve(query, intent)
        if group is None:
            return self._insufficient(
                query, "explain_priority",
                "the question does not name a subject, message or group that "
                "the ledger tracks")
        decision = self.ledger.priorities.get(group.group_id)
        if decision is None:
            return self._insufficient(query, "explain_priority",
                                      f"no priority was recorded for "
                                      f"{group.group_id}")
        history = self.ledger.priority_history.get(group.group_id, [])
        detail = "; ".join(
            f"{s.name} ({s.weight:+.2f}) -- {s.detail}" for s in decision.signals)
        moves = "; ".join(
            f"{c.as_of[:10]}: {c.previous} -> {c.new} via {c.trigger}"
            for c in history[-4:])
        evidence = [
            self._evidence(m.message_id, round(m.match_score, 4), "structured",
                           f"member of {group.group_id} ({m.act_label})")
            for m in group.members[-6:]]
        return Answer(
            query, "explain_priority",
            f"{group.title} ({group.group_id}) is {decision.priority} with a "
            f"score of {decision.score}. {decision.reason} Signals: {detail}. "
            f"History: {moves or 'no band changes recorded'}.",
            reason=("the priority engine records every signal it fired with a "
                    "weight and an explanation, so this is the decision's own "
                    "arithmetic rather than a reconstruction of it"),
            confidence=decision.confidence, evidence=evidence,
            group_ids=[group.group_id], item_ids=group.item_ids[:6])

    def _answer_deadline_changes(self, query: str, intent: Intent) -> Answer:
        rows: List[Tuple[G.MessageGroup, G.Change]] = []
        for group in self.ledger.groups:
            if not self._in_scope(group, intent):
                continue
            for change in group.deadline_history:
                if change.origin != "l2_update":
                    continue
                if change.direction in ("earlier", "later", "unresolved"):
                    rows.append((group, change))
        if not rows:
            return self._insufficient(query, "deadline_changes",
                                      "no message changed a deadline in scope")
        rows.sort(key=lambda gc: gc[1].timestamp)
        evidence = [
            self._evidence(
                c.message_id, 1.0, "structured",
                f"this message changed the deadline for {g.group_id} "
                f"({c.previous} -> {c.new or c.raw}, {c.direction})")
            for g, c in rows]
        lines = [
            f"{g.title}: {c.previous or 'none'} -> {c.new or c.raw} "
            f"({c.direction}) in {c.message_id}"
            + (" [sender flagged the earlier message as listing another date]"
               if c.flagged_conflict else "")
            for g, c in rows]
        return Answer(
            query, "deadline_changes",
            f"{len(rows)} deadline change(s) are on record. "
            + "; ".join(lines) + ".",
            reason=("read from each group's deadline history, which records "
                    "every change with the message that made it. Original "
                    "requests that merely restated a subject with a different "
                    "date are excluded -- they are recorded separately, "
                    "because restating is not changing"),
            confidence=0.92, evidence=evidence,
            group_ids=sorted({g.group_id for g, _ in rows}))

    #: Conflict kinds that are specifically about *when* something is due.
    DEADLINE_CONFLICTS = frozenset({
        "deadline_changed", "deadline_changed_after_closure",
        "unresolved_alternatives",
    })

    def _answer_conflicts(self, query: str, intent: Intent) -> Answer:
        # "conflicting or uncertain deadlines" is a narrower question than
        # "any conflicts", and answering the narrow one with the wide answer
        # buries it. The query's own wording selects which.
        deadline_only = bool(re.search(r"\bdeadlines?\b|\bdue\s+dates?\b|"
                                       r"\bdates?\b", query, re.IGNORECASE))
        rows: List[Tuple[G.MessageGroup, G.Conflict]] = []
        for group in self.ledger.groups:
            if not self._in_scope(group, intent):
                continue
            for conflict in group.conflicts:
                if deadline_only and conflict.kind not in self.DEADLINE_CONFLICTS:
                    continue
                rows.append((group, conflict))

        # Messages that are uncertain on their own account, whether or not they
        # were ever attached to a subject. "The deadline may be Monday, or it
        # may be Wednesday" names no subject, so it belongs to no group -- and
        # a conflict report that only reads groups would never mention the one
        # message in the corpus that is purely a contradiction.
        loose: List[LedgerRecord] = []
        latest = self._latest_ids()
        for rec in self.ledger.records:
            if intent.demo_scope and rec.batch_kind != "demo":
                continue
            v = rec.verdict
            if v.alternatives or (v.conflict_marker and rec.group_id is None) \
                    or (deadline_only and v.hedged and v.new_date_raw):
                loose.append(rec)
        if not rows and not loose:
            return self._insufficient(query, "conflicts",
                                      "no contradiction was recorded in scope")
        evidence: List[Evidence] = []
        seen: set = set()
        for group, conflict in rows:
            for mid in conflict.message_ids:
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                evidence.append(self._evidence(
                    mid, 1.0, "structured",
                    f"recorded in {group.group_id} as a "
                    f"{conflict.kind.replace('_', ' ')}: {conflict.detail}"))
        for rec in loose:
            if rec.message_id in seen:
                continue
            seen.add(rec.message_id)
            detail = ("offers mutually exclusive options ("
                      + " or ".join(rec.verdict.alternatives) + ") without "
                      "choosing one") if rec.verdict.alternatives else (
                "states a deadline in relative terms that was never resolved"
                if rec.verdict.new_date_raw else
                "flags an earlier message as giving a different date")
            evidence.append(self._evidence(
                rec.message_id, 1.0, "structured",
                f"this message {detail}"
                + ("" if rec.group_id else "; it names no subject, so it "
                   "belongs to no group")))

        pending_rel = [g for g in self.ledger.groups
                       if g.pending_relative_deadline
                       and self._in_scope(g, intent)]

        kinds: Dict[str, int] = {}
        for _, conflict in rows:
            kinds[conflict.kind] = kinds.get(conflict.kind, 0) + 1
        breakdown = ", ".join(f"{n} x {k.replace('_', ' ')}"
                              for k, n in sorted(kinds.items(),
                                                 key=lambda kv: -kv[1]))
        contested = [g.title for g in self.ledger.groups
                     if g.contested and self._in_scope(g, intent)]
        unresolved = ""
        if pending_rel:
            unresolved = (" Unresolved: " + "; ".join(
                f'{g.title} -- newest deadline given only as '
                f'"{g.pending_relative_deadline}" in {g.pending_relative_source}'
                for g in pending_rel) + ".")
        loose_note = ""
        if loose:
            latest_first = ([r.message_id for r in loose
                             if r.message_id in latest]
                            + [r.message_id for r in loose
                               if r.message_id not in latest])
            shown = latest_first[:10]
            more = (f" and {len(latest_first) - len(shown)} more"
                    if len(latest_first) > len(shown) else "")
            loose_note = (f" {len(loose)} message(s) are uncertain in "
                          f"themselves -- they hedge a date or offer options "
                          f"without choosing: {', '.join(shown)}{more}.")
        return Answer(
            query, "conflicts",
            f"{len(rows)} contradiction(s) are recorded across "
            f"{len({g.group_id for g, _ in rows})} subject(s): {breakdown}."
            + loose_note + unresolved + " "
            + (f"Contested subjects (messages disagree about their state): "
               f"{', '.join(contested[:6])}." if contested else ""),
            reason=("conflicts are recorded by the grouper as it walks the "
                    "messages in time order -- a hedge that contradicts a firm "
                    "status, a chase after a closure, a deadline restated "
                    "differently, or options offered without a choice"),
            confidence=0.9, evidence=evidence[:24],
            group_ids=sorted({g.group_id for g, _ in rows}))

    def _answer_rescheduled(self, query: str, intent: Intent) -> Answer:
        moved = [g for g in self.ledger.groups
                 if g.schedule_history and self._in_scope(g, intent)
                 and any(c.message_id for c in g.schedule_history)]
        # Only subjects an L2 message actually moved, not events whose L1
        # duplicates each carried their own date.
        moved = [g for g in moved
                 if any(m.act == "reschedule" for m in g.members)]
        if not moved:
            return self._insufficient(query, "rescheduled",
                                      "no message rescheduled a meeting in scope")
        latest = self._latest_ids()
        recent = [g for g in moved
                  if any(m.message_id in latest and m.act == "reschedule"
                         for m in g.members)]
        evidence: List[Evidence] = []
        lines: List[str] = []
        for group in sorted(moved, key=lambda g: g.last_seen, reverse=True):
            when = " at ".join(x for x in (group.latest_date, group.latest_time)
                               if x) or "unspecified"
            status = "" if group.status != G.CANCELLED else " (later cancelled)"
            lines.append(f"{group.title}: now {when} "
                         f"(from {group.schedule_source}){status}")
            for member in group.members:
                if member.act == "reschedule":
                    evidence.append(self._evidence(
                        member.message_id, round(member.match_score, 4),
                        "structured",
                        f"a reschedule act on {group.group_id}"))
            hedged = [m for m in group.members
                      if m.act == "ambiguous_update" and m.hedged]
            if hedged:
                lines[-1] += (f"; {hedged[-1].message_id} later hedges about "
                              f"moving it again without confirming, so the "
                              f"schedule above is unchanged")
        lead = ""
        if recent and len(recent) < len(moved):
            names = "; ".join(
                f"{g.title}, now "
                + (" at ".join(x for x in (g.latest_date, g.latest_time) if x)
                   or "unspecified")
                + f" (from {g.schedule_source})"
                for g in recent)
            lead = (f"In the most recent batch ({self._latest_batch()}), "
                    f"{len(recent)} meeting was rescheduled: {names}. ")
        return Answer(
            query, "rescheduled",
            lead + f"Across the whole corpus {len(moved)} meeting(s) were "
            f"rescheduled. " + "; ".join(lines) + ".",
            reason=("taken from each group's schedule history. A message that "
                    "changes only the time keeps the date already on record, "
                    "so the latest schedule is the merge of every change in "
                    "order rather than the contents of the last message"),
            confidence=0.92, evidence=evidence[:20],
            group_ids=[g.group_id for g in moved],
            item_ids=[i for g in moved for i in g.item_ids][:10])

    def _answer_by_status(self, query: str, intent: Intent) -> Answer:
        wanted = intent.statuses or [G.COMPLETED, G.CANCELLED]
        wanted = [w for w in wanted if w in G.STATUSES]
        rows = [g for g in self.ledger.groups
                if g.status in wanted and self._in_scope(g, intent)]
        if not rows:
            return self._insufficient(
                query, "by_status",
                f"no subject in scope has status {', '.join(wanted)}")
        evidence: List[Evidence] = []
        lines: List[str] = []
        for group in sorted(rows, key=lambda g: g.status):
            lines.append(f"{group.title} -- {G.STATUS_LABEL[group.status]} "
                         f"({group.status_source})"
                         + (" [contested]" if group.contested else ""))
            if group.status_source:
                evidence.append(self._evidence(
                    group.status_source, round(group.confidence, 4),
                    "structured",
                    f"this message set {group.group_id} to {group.status}: "
                    f"{group.status_reason}"))
        contested = sum(1 for g in rows if g.contested)
        return Answer(
            query, "by_status",
            f"{len(rows)} subject(s) are {' or '.join(wanted)}: "
            + "; ".join(lines) + "."
            + (f" {contested} of them are contested -- later messages disagree."
               if contested else ""),
            reason=("read from the status each group's state machine settled "
                    "on, which only firm acts can set; the cited message is "
                    "the one that made the transition"),
            confidence=0.93, evidence=evidence,
            group_ids=[g.group_id for g in rows],
            item_ids=[i for g in rows for i in g.item_ids][:12])

    def _answer_due_today(self, query: str, intent: Intent) -> Answer:
        today = self.ledger.as_of[:10]
        rows: List[Tuple[G.MessageGroup, str]] = []
        for group in self._active():
            due = group.latest_deadline or group.latest_date
            if due and due <= today:
                rows.append((group, due))
            elif group.pending_relative_deadline and any(
                    w in group.pending_relative_deadline.lower()
                    for w in ("today", "tonight")):
                rows.append((group, group.pending_relative_deadline))
        if not rows:
            return self._insufficient(
                query, "due_today",
                f"nothing open has a deadline on or before {today}")
        rows.sort(key=lambda gd: (self.ledger.priorities.get(
            gd[0].group_id, P.PriorityDecision(P.LOW, 0, 0, "")).score * -1))
        evidence: List[Evidence] = []
        lines: List[str] = []
        for group, due in rows:
            decision = self.ledger.priorities.get(group.group_id)
            band = decision.priority if decision else "unscored"
            overdue = " (overdue)" if due < today else ""
            lines.append(f"[{band}] {group.title} -- due {due}{overdue}")
            src = group.latest_deadline_source or group.schedule_source
            if src:
                evidence.append(self._evidence(
                    src, decision.confidence if decision else 0.5,
                    "structured",
                    f"this message set the current deadline for "
                    f"{group.group_id}"))
        return Answer(
            query, "due_today",
            f"{len(rows)} open subject(s) are due on or before {today}, the "
            f"date of the most recent message. " + "; ".join(lines) + ".",
            reason=("filtered on each group's current deadline against the "
                    "ledger's reference date, which is the newest message's "
                    "timestamp rather than the wall clock, and restricted to "
                    "subjects that are not completed or cancelled"),
            confidence=0.9, evidence=evidence[:12],
            group_ids=[g.group_id for g, _ in rows],
            item_ids=[i for g, _ in rows for i in g.item_ids][:12])

    def _answer_pending_by_priority(self, query: str, intent: Intent) -> Answer:
        bands = intent.bands or [P.CRITICAL, P.HIGH]
        bands = [b for b in bands if b in P.BANDS]
        rows = []
        for group in self._active():
            decision = self.ledger.priorities.get(group.group_id)
            if decision and decision.priority in bands \
                    and self._in_scope(group, intent):
                rows.append((group, decision))
        if not rows:
            return self._insufficient(
                query, "pending_by_priority",
                f"nothing open is currently {' or '.join(bands)}")
        rows.sort(key=lambda gd: -gd[1].score)
        evidence: List[Evidence] = []
        lines: List[str] = []
        for group, decision in rows:
            due = group.latest_deadline or group.pending_relative_deadline \
                or group.latest_date or "no date on record"
            lines.append(f"[{decision.priority}] {group.title} -- "
                         f"{G.STATUS_LABEL[group.status]}, due {due}")
            src = (group.latest_deadline_source or group.status_source
                   or group.members[-1].message_id)
            evidence.append(self._evidence(
                src, decision.confidence, "structured",
                f"{group.group_id} is {decision.priority}: {decision.reason}"))
        return Answer(
            query, "pending_by_priority",
            f"{len(rows)} open subject(s) are {' or '.join(bands)}. "
            + "; ".join(lines) + ".",
            reason=("filtered on the current priority snapshot, excluding "
                    "anything a message has reported complete or cancelled; "
                    "ordered by the priority score so the reason for the "
                    "ordering is the same number that set the band"),
            confidence=0.91, evidence=evidence[:12],
            group_ids=[g.group_id for g, _ in rows],
            item_ids=[i for g, _ in rows for i in g.item_ids][:12])

    def _answer_latest_status(self, query: str, intent: Intent) -> Answer:
        group = self._resolve(query, intent)
        if group is None:
            return self._insufficient(
                query, "latest_status",
                "the question does not name a subject the ledger tracks, and "
                "no message in it resolves to one")
        decision = self.ledger.priorities.get(group.group_id)
        hedges = [m for m in group.members if m.hedged]
        evidence: List[Evidence] = []
        if group.status_source:
            evidence.append(self._evidence(
                group.status_source, round(group.confidence, 4), "structured",
                f"this is the message that set the status: "
                f"{group.status_reason}"))
        for member in group.members[-4:]:
            if member.message_id != group.status_source:
                evidence.append(self._evidence(
                    member.message_id, round(member.match_score, 4),
                    "structured",
                    f"the most recent traffic on {group.group_id} "
                    f"({member.act_label})"))

        parts = [f"{group.title} ({group.group_id}) is "
                 f"{G.STATUS_LABEL[group.status]}."]
        parts.append(group.summary)
        if hedges:
            last = hedges[-1]
            parts.append(
                f"{last.message_id} hedges about this subject, so it does not "
                f"change the recorded status -- an uncertain report is not a "
                f"status.")
        if decision:
            parts.append(f"Current priority: {decision.priority}.")
        return Answer(
            query, "latest_status", " ".join(parts),
            reason=(f"resolved the question to {group.group_id} and read the "
                    f"status its state machine settled on after processing all "
                    f"{len(group.members)} messages in time order"),
            confidence=round(min(group.confidence, 0.95), 4),
            evidence=evidence, group_ids=[group.group_id],
            item_ids=group.item_ids[:6])

    def _answer_messages_about(self, query: str, intent: Intent) -> Answer:
        group = self._resolve(query, intent)
        hits = self._context(intent.subject or query, k=12)
        if group is None and (not hits or hits[0]["score"] < MIN_EVIDENCE):
            return self._insufficient(
                query, "messages_about",
                "no subject group matched, and retrieval found nothing above "
                "the evidence threshold")
        evidence: List[Evidence] = []
        if group is not None:
            for member in group.members:
                evidence.append(self._evidence(
                    member.message_id, round(member.match_score, 4),
                    "structured",
                    f"grouped into {group.group_id} on the shared subject "
                    f"terms {member.match_tokens or 'the group seed'} "
                    f"({member.act_label})"))
            body = (f"{len(group.members)} message(s) belong to "
                    f"{group.group_id} ({group.title}). {group.summary} "
                    f"Current status: {G.STATUS_LABEL[group.status]}.")
            reason = ("answered from the subject group rather than from search: "
                      "the group is the complete set of messages about this "
                      "subject, whereas retrieval returns the top matches and "
                      "would silently miss a follow-up that shares no wording")
            confidence = round(min(group.confidence, 0.95), 4)
            weak = self._weak_resolution_note(query, intent, group)
            if weak:
                body += " " + weak
                confidence = round(confidence * 0.85, 4)
        else:
            for hit in hits:
                evidence.append(self._evidence(
                    hit["message_id"], hit["score"], "retrieval",
                    f"retrieved at {hit['score']:.3f} on the terms "
                    f"{hit['matched_terms']}"))
            body = (f"No tracked subject matched, so these are the "
                    f"{len(hits)} best-matching messages by relevance.")
            reason = ("no subject group matched the phrase, so this falls back "
                      "to retrieval and the answer is a ranked list rather "
                      "than a complete set")
            confidence = 0.55
        return Answer(query, "messages_about", body, reason, confidence,
                      evidence=evidence,
                      group_ids=[group.group_id] if group else [],
                      item_ids=group.item_ids[:8] if group else [])

    def _answer_yes_no_verification(self, query: str, intent: Intent) -> Answer:
        """Answer a yes/no question -- or say why it cannot be answered.

        A question of the form "was X approved?" is only answerable by a
        message that *asserts* something about X. A message that asks the same
        question is the highest-scoring possible match and the least useful
        possible answer, so the test here is not similarity but assertion:
        does any message about this subject perform a completion, a
        cancellation, a reschedule, a deadline change or a progress report?

        This is the case the brief has in mind when it says the assistant must
        say so if sufficient evidence is unavailable.
        """
        assertions = {"completion", "cancellation", "reschedule",
                      "deadline_change", "progress_update"}
        group = self._resolve(query, intent)
        hits = self._context(query, k=6)

        if group is None:
            best = hits[0] if hits else None
            if best is None or best["score"] < MIN_EVIDENCE:
                return self._insufficient(
                    query, "yes_no_verification",
                    "no subject in the processed messages matches this "
                    "question, and nothing retrieved above the evidence "
                    "threshold")
            return self._insufficient(
                query, "yes_no_verification",
                f"nothing the ledger tracks matches this question. The closest "
                f"message is {best['message_id']} at {best['score']:.3f}, which "
                f"is a similarity, not an answer")

        asserting = [m for m in group.members if m.act in assertions]
        if not asserting:
            # Anything that raises the subject without asserting an outcome.
            # That includes a question L1 read as actionable and turned into a
            # task -- being filed as a task is not the same as being answered.
            askers = [m.message_id for m in group.members
                      if m.act not in assertions]
            evidence = [
                self._evidence(mid, round(group.confidence, 4), "structured",
                               "this message raises the same subject without "
                               "asserting anything about it")
                for mid in (askers or group.message_ids)[:4]]
            detail = (f"{group.group_id} ({group.title}) collects "
                      f"{len(group.members)} message(s) on this subject and "
                      f"none of them asserts an outcome -- nothing reports it "
                      f"done, cancelled, rescheduled or given a new deadline")
            if askers:
                detail += (f". {', '.join(askers[:3])} "
                           f"{'raise' if len(askers) > 1 else 'raises'} the "
                           f"subject and nothing in the corpus replies")
            return Answer(
                query, "yes_no_verification",
                f"There is not enough evidence to answer this. {detail}.",
                reason=("a yes/no question can only be answered by a message "
                        "that asserts an outcome. The subject was located, but "
                        "every message about it is a question or a request, so "
                        "the honest answer is that the corpus does not say"),
                confidence=0.78, sufficient=False, evidence=evidence,
                group_ids=[group.group_id])

        last = asserting[-1]
        evidence = [self._evidence(
            last.message_id, round(group.confidence, 4), "structured",
            f"the most recent assertion about {group.group_id} "
            f"({last.act_label})")]
        return Answer(
            query, "yes_no_verification",
            f"{group.title} ({group.group_id}) is "
            f"{G.STATUS_LABEL[group.status]}. {group.summary}",
            reason=(f"the subject was located and {last.message_id} asserts an "
                    f"outcome for it, so the question is answerable from the "
                    f"group's state rather than from similarity"),
            confidence=round(min(group.confidence, 0.93), 4),
            evidence=evidence, group_ids=[group.group_id],
            item_ids=group.item_ids[:4])

    # -- retrieval fallback -------------------------------------------------
    def _answer_open_search(self, query: str, intent: Intent) -> Answer:
        if intent.message_ids:
            resolved = self._resolve(query, intent)
            if resolved is not None:
                return self._answer_latest_status(query, intent)

        hits = self._context(query, k=8)
        if not hits or hits[0]["score"] < MIN_EVIDENCE:
            return self._insufficient(
                query, "open_search",
                "nothing in the corpus scores above the evidence threshold "
                f"of {MIN_EVIDENCE}")

        # The interesting refusal: the best match is itself an unanswered
        # question. High lexical similarity to a question is not an answer to
        # it -- it is the most confident possible way to be wrong -- so the
        # test is whether anything ever *asserted* something about that
        # subject. A subject with no firm act and no extracted item was asked
        # about and never resolved.
        top = self._record(hits[0]["message_id"])
        asking: List[LedgerRecord] = []
        if top is not None and top.verdict.act == "open_question":
            group = (self.ledger.group(top.group_id) if top.group_id else None)
            resolved = (group is not None
                        and (group.status_history or group.item_ids))
            if not resolved:
                asking = [self._record(h["message_id"]) for h in hits[:3]]
                asking = [r for r in asking
                          if r is not None
                          and r.verdict.act == "open_question"
                          and not (r.group_id
                                   and (self.ledger.group(r.group_id).status_history
                                        or self.ledger.group(r.group_id).item_ids))]
        if asking:
            score_by_id = {h["message_id"]: h["score"] for h in hits}
            evidence = [
                self._evidence(r.message_id, score_by_id.get(r.message_id, 0.0),
                               "retrieval",
                               "this message asks the same question; nothing "
                               "in the corpus answers it")
                for r in asking]
            return Answer(
                query, "open_search",
                "There is not enough evidence to answer this. "
                f"{', '.join(r.message_id for r in asking)} "
                f"{'ask' if len(asking) > 1 else 'asks'} the same thing, but no "
                f"message in the corpus answers it, and no task, event or "
                f"subject group was created for it.",
                reason=("the best-matching messages are themselves unanswered "
                        "questions. Lexical similarity to a question is not an "
                        "answer to it, so the assistant reports what it found "
                        "instead of returning the question as the answer"),
                confidence=0.8, sufficient=False, evidence=evidence)

        evidence = [
            self._evidence(h["message_id"], h["score"], "retrieval",
                           f"retrieved at {h['score']:.3f}; lexical "
                           f"{h['lexical_score']:.3f}, semantic "
                           f"{h['semantic_score']:.3f} on "
                           f"{h['matched_terms']}")
            for h in hits if h["score"] >= MIN_EVIDENCE]
        groups = sorted({e.group_id for e in evidence if e.group_id})
        lead = self.ledger.group(groups[0]) if groups else None
        body = (f"The {len(evidence)} most relevant message(s) are "
                f"{', '.join(e.message_id for e in evidence[:6])}.")
        if lead is not None:
            body += (f" Most of them belong to {lead.group_id} ({lead.title}), "
                     f"currently {G.STATUS_LABEL[lead.status]}.")
        return Answer(
            query, "open_search", body,
            reason=("this question did not match a structured intent, so it "
                    "was answered by hybrid retrieval -- lexical scoring with "
                    "a semantic rerank -- and the scores are shown so the "
                    "strength of the match is visible"),
            confidence=round(min(0.4 + hits[0]["score"], 0.8), 4),
            evidence=evidence, group_ids=groups[:4])

    # -- shared machinery ---------------------------------------------------
    def _resolve_detail(self, query: str, intent: Intent):
        """Resolve and report *how* strong the resolution was."""
        for phrase in filter(None, (intent.subject, query)):
            match = self.ledger.resolve_subject(phrase)
            if match:
                return match
        return None

    def _weak_resolution_note(self, query: str, intent: Intent,
                              group: G.MessageGroup) -> Optional[str]:
        detail = self._resolve_detail(query, intent)
        if not detail or detail[0].group_id != group.group_id:
            return None
        _, _, tokens = detail
        if len(tokens) == 1:
            return (f'Resolved on the single shared term "{tokens[0]}". If you '
                    f'meant a different subject containing that word, name it '
                    f'more fully -- this is the weakest link the matcher will '
                    f'accept.')
        return None

    def _resolve(self, query: str, intent: Intent) -> Optional[G.MessageGroup]:
        """Find the group a question is about, by ID, by phrase, or by search."""
        for mid in intent.message_ids:
            if mid.startswith("GROUP_"):
                group = self.ledger.group(mid)
                if group is not None:
                    return group
            group = self.ledger.group_of(mid)
            if group is not None:
                return group
            rec = self._record(mid)
            if rec is not None and rec.verdict.subject_phrase:
                match = self.ledger.resolve_subject(rec.verdict.subject_phrase)
                if match:
                    return match[0]
        for phrase in filter(None, (intent.subject, query)):
            match = self.ledger.resolve_subject(phrase)
            if match:
                return match[0]
        # Last resort: let retrieval nominate a group, but only when its top
        # hits agree on one. A single loose match is not a resolution.
        hits = self._context(intent.subject or query, k=5)
        counts: Dict[str, float] = {}
        for hit in hits:
            if hit["score"] < MIN_EVIDENCE:
                continue
            rec = self._record(hit["message_id"])
            if rec and rec.group_id:
                counts[rec.group_id] = counts.get(rec.group_id, 0.0) + hit["score"]
        if len(counts) == 1 or (counts and max(counts.values()) >= 0.6):
            return self.ledger.group(max(counts, key=counts.get))
        return None

    def _insufficient(self, query: str, intent: str, why: str) -> Answer:
        return Answer(
            query, intent,
            f"There is not enough evidence in the processed messages to answer "
            f"this. {why[0].upper()}{why[1:]}.",
            reason=("the assistant refuses rather than composing an answer "
                    "from evidence that does not support one"),
            confidence=0.5, sufficient=False)
