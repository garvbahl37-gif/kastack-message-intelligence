"""Related-message grouping and subject lifecycle.

Part 2 of the L2 brief: connect the messages that talk about the same task,
meeting or request, and say what is true about it *now*.

    MSG_0002  "Can you review the privacy checklist before 2026-09-09?"
    MSG_0901  "Can you share an update on review the privacy checklist?"
    MSG_0980  "Please note that review the privacy checklist is due on 2026-09-30,
               although the earlier message listed another date."
    MSG_0921  "Follow-up: please confirm whether you started to review the
               privacy checklist."

One group. One current deadline. One status. One recorded conflict.

How grouping works
------------------
Two passes, and the split between them is deliberate:

* **Pass 1** collects every subject signature in the corpus and builds the IDF
  statistics in `mint/subject.py`. These are corpus-level statistics -- how
  many distinct subjects a word names -- and nothing about a message's position
  in time enters them.
* **Pass 2** walks the messages in **chronological order**, assigning each to a
  group and advancing that group's state machine. No message is ever influenced
  by the content of a later one.

Keeping those apart is what lets the matcher know that "review" is a cheap word
without letting the future leak into the past.

How status is decided
---------------------
A small state machine, driven by the speech acts in `mint/acts.py`, under four
rules that matter more than the transition table:

1. **Only firm acts move the state.** A hedged sentence never produces a firm
   status. If nothing firm has ever been said, the status is ``unclear`` -- not
   ``pending``, because those mean different things.
2. **Restatements do not re-assert.** "Follow-up: X has been completed" repeats
   an earlier claim; it must not out-rank something said in between. A repeated
   act is absorbed when the group already carries that act, and applied when it
   does not -- a follow-up can still be the first time a fact arrives.
3. **Completion and cancellation are terminal.** Later chases and deadline
   edits do not silently reopen the item; they are recorded as contradictions,
   which is what they are.
4. **Contradictions are kept, not resolved away.** When a subject is reported
   both complete and cancelled, the group says so and its confidence drops.
   Averaging that disagreement into a clean-looking answer would be a lie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import acts as A
from .subject import SubjectSignature, SubjectSpace, signature

# --------------------------------------------------------------------------
# Statuses
# --------------------------------------------------------------------------
PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
RESCHEDULED = "rescheduled"
CANCELLED = "cancelled"
UNCLEAR = "unclear"

STATUSES = [PENDING, IN_PROGRESS, COMPLETED, RESCHEDULED, CANCELLED, UNCLEAR]

#: Once a subject reaches one of these, later non-terminal traffic is treated
#: as contradiction rather than as a reopening.
TERMINAL = frozenset({COMPLETED, CANCELLED})

STATUS_LABEL = {
    PENDING: "Pending",
    IN_PROGRESS: "In progress",
    COMPLETED: "Completed",
    RESCHEDULED: "Rescheduled",
    CANCELLED: "Cancelled",
    UNCLEAR: "Unclear",
}


@dataclass
class Member:
    """One message inside a group, with the evidence that put it there."""

    message_id: str
    timestamp: str
    sender: str
    act: str
    act_label: str
    frame: str
    hedged: bool
    repetition: bool
    match_score: float
    match_tokens: List[str]
    #: True when the link rests on a single shared token. Kept visible because
    #: it is the weakest kind of evidence the grouper will accept.
    weak_link: bool
    seed: bool = False
    item_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "act": self.act,
            "act_label": self.act_label,
            "frame": self.frame,
            "hedged": self.hedged,
            "repetition": self.repetition,
            "match_score": self.match_score,
            "match_tokens": self.match_tokens,
            "weak_link": self.weak_link,
            "seed": self.seed,
            "item_id": self.item_id,
        }


@dataclass
class Change:
    """One recorded edit to a deadline or a schedule."""

    message_id: str
    timestamp: str
    field: str                 # "deadline" | "date" | "time"
    previous: Optional[str]
    new: Optional[str]
    raw: Optional[str] = None
    direction: str = "set"     # set | earlier | later | unchanged | unresolved
    flagged_conflict: bool = False
    #: "l1_request" -- one of several original requests, each carrying its own
    #: date; "l2_update" -- a message whose purpose is to change the date.
    #: Both are real, but only the second is a deliberate change, and only the
    #: second should raise a conflict on its own.
    origin: str = "l2_update"

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "field": self.field,
            "previous": self.previous,
            "new": self.new,
            "raw": self.raw,
            "direction": self.direction,
            "flagged_conflict": self.flagged_conflict,
            "origin": self.origin,
        }


@dataclass
class Transition:
    """One status change, and why it happened."""

    message_id: str
    timestamp: str
    previous: str
    new: str
    act: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "from": self.previous,
            "to": self.new,
            "act": self.act,
            "reason": self.reason,
        }


@dataclass
class Conflict:
    kind: str
    message_ids: List[str]
    detail: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "message_ids": self.message_ids,
                "detail": self.detail}


@dataclass
class MessageGroup:
    """One subject, everything said about it, and where it stands now."""

    group_id: str
    title: str
    kind: str                                  # task | event | subject
    signature: SubjectSignature
    variants: List[SubjectSignature] = field(default_factory=list)

    members: List[Member] = field(default_factory=list)
    item_ids: List[str] = field(default_factory=list)

    status: str = UNCLEAR
    status_reason: str = "no firm statement has been made about this subject"
    status_source: Optional[str] = None
    status_history: List[Transition] = field(default_factory=list)

    latest_deadline: Optional[str] = None
    latest_deadline_source: Optional[str] = None
    #: Set when the newest deadline was stated relatively ("tomorrow"). The
    #: phrase is preserved; it is never resolved into a stored date.
    pending_relative_deadline: Optional[str] = None
    pending_relative_source: Optional[str] = None

    latest_date: Optional[str] = None          # events
    latest_time: Optional[str] = None
    schedule_source: Optional[str] = None

    deadline_history: List[Change] = field(default_factory=list)
    schedule_history: List[Change] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)

    chase_count: int = 0
    restatement_count: int = 0
    #: Original requests that repeated the subject with a different date.
    deadline_restatements: int = 0
    response_required: bool = False
    urgency_flagged: bool = False
    de_escalated: bool = False
    unresolved_alternatives: List[str] = field(default_factory=list)
    contested: bool = False

    #: Firm acts already applied to this group. A restatement of one of these
    #: is absorbed rather than re-asserted. Kept separately from
    #: `status_history` because an act can fire without moving the status --
    #: a second deadline change on an already-pending subject, for instance.
    applied_acts: List[str] = field(default_factory=list)

    first_seen: str = ""
    last_seen: str = ""
    confidence: float = 0.0
    summary: str = ""

    # -- helpers ------------------------------------------------------------
    @property
    def message_ids(self) -> List[str]:
        return [m.message_id for m in self.members]

    @property
    def last_act_message(self) -> Optional[Member]:
        return self.members[-1] if self.members else None

    def has_firm_act(self, act: str) -> bool:
        return act in self.applied_acts

    def to_dict(self) -> dict:
        """The Part-2 record required by the brief, plus its provenance."""
        return {
            "group_id": self.group_id,
            "title": self.title,
            "kind": self.kind,
            "related_message_ids": self.message_ids,
            "related_item_ids": self.item_ids,
            "status": self.status,
            "status_label": STATUS_LABEL.get(self.status, self.status),
            "status_reason": self.status_reason,
            "status_source_message_id": self.status_source,
            "latest_deadline": self.latest_deadline,
            "latest_deadline_source": self.latest_deadline_source,
            "pending_relative_deadline": self.pending_relative_deadline,
            "pending_relative_source": self.pending_relative_source,
            "latest_schedule": (
                {"date": self.latest_date, "time": self.latest_time,
                 "source_message_id": self.schedule_source}
                if (self.latest_date or self.latest_time) else None
            ),
            "summary": self.summary,
            "confidence": self.confidence,
            "message_count": len(self.members),
            "chase_count": self.chase_count,
            "restatement_count": self.restatement_count,
            "deadline_restatements": self.deadline_restatements,
            "response_required": self.response_required,
            "contested": self.contested,
            "unresolved_alternatives": self.unresolved_alternatives,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "subject_tokens": sorted(self.signature.tokens),
            "members": [m.to_dict() for m in self.members],
            "status_history": [t.to_dict() for t in self.status_history],
            "deadline_history": [c.to_dict() for c in self.deadline_history],
            "schedule_history": [c.to_dict() for c in self.schedule_history],
            "conflicts": [c.to_dict() for c in self.conflicts],
        }


# --------------------------------------------------------------------------
# Building the groups
# --------------------------------------------------------------------------
@dataclass
class SubjectInput:
    """What the grouper needs about one message. Masked text only."""

    message_id: str
    timestamp: str
    sender: str
    masked_text: str
    category: str
    verdict: A.ActVerdict
    item_id: Optional[str] = None
    item_title: Optional[str] = None
    item_type: Optional[str] = None
    item_date: Optional[str] = None
    item_time: Optional[str] = None


def subject_for(inp: SubjectInput) -> Tuple[Optional[SubjectSignature], List[SubjectSignature], str]:
    """Pick the subject phrase for a message and say where it came from.

    An L2 frame that captured a subject is the best source -- it was written to
    find exactly that span. Failing that, the title L1 already extracted is
    better than re-deriving one from the sentence. A bare unmatched question is
    the weakest source and is used only as a last resort.
    """
    variants: List[SubjectSignature] = []
    item_sig = signature(inp.item_title) if inp.item_title else None
    if item_sig:
        variants.append(item_sig)

    v = inp.verdict
    if v.subject_phrase and v.act not in (A.INFORMATIONAL, A.OPEN_QUESTION):
        sig = signature(v.subject_phrase)
        if sig:
            return sig, [s for s in variants if s], "act_frame"

    if item_sig:
        return item_sig, [], "l1_item_title"

    if v.subject_phrase and v.act == A.OPEN_QUESTION:
        sig = signature(v.subject_phrase)
        if sig:
            return sig, [], "open_question"

    return None, [], "none"


def _direction(previous: Optional[str], new: Optional[str]) -> str:
    if new is None:
        return "unresolved"
    if previous is None:
        return "set"
    if new < previous:
        return "earlier"
    if new > previous:
        return "later"
    return "unchanged"


class GroupBuilder:
    """Assigns messages to groups and advances each group's state machine."""

    def __init__(self, space: SubjectSpace) -> None:
        self.space = space
        self.groups: List[MessageGroup] = []
        self._n = 0

    # -- assignment ---------------------------------------------------------
    def _find_group(
        self, sig: SubjectSignature
    ) -> Optional[Tuple[MessageGroup, float, List[str]]]:
        best: Optional[Tuple[MessageGroup, float, List[str]]] = None
        for g in self.groups:
            for candidate in [g.signature] + g.variants:
                score, tokens = self.space.match(sig, candidate)
                if score <= 0.0:
                    continue
                if best is None or score > best[1]:
                    best = (g, score, tokens)
        return best

    @staticmethod
    def _title(sig: SubjectSignature) -> str:
        raw = sig.display or " ".join(sorted(sig.tokens))
        return raw[:1].upper() + raw[1:] if raw else raw

    def _new_group(self, sig: SubjectSignature, kind: str) -> MessageGroup:
        self._n += 1
        return MessageGroup(
            group_id=f"GROUP_{self._n:03d}",
            title=self._title(sig),
            kind=kind,
            signature=sig,
        )

    # -- the state machine --------------------------------------------------
    def _apply(self, g: MessageGroup, inp: SubjectInput, member: Member) -> None:
        v = inp.verdict
        mid, ts = inp.message_id, inp.timestamp

        if v.urgent:
            g.urgency_flagged = True
        if v.de_escalated:
            g.de_escalated = True
        if v.alternatives:
            for a in v.alternatives:
                if a not in g.unresolved_alternatives:
                    g.unresolved_alternatives.append(a)
            g.conflicts.append(Conflict(
                "unresolved_alternatives", [mid],
                "the sender offered mutually exclusive options ("
                + " or ".join(v.alternatives) + ") without choosing one",
            ))

        # ---- status chases -------------------------------------------------
        if v.act in A.RESPONSE_REQUIRED_ACTS:
            g.chase_count += 1
            g.response_required = True
            if g.status in TERMINAL:
                g.contested = True
                g.conflicts.append(Conflict(
                    "chased_after_closure", [mid, g.status_source or ""],
                    f"the subject was chased after it was reported "
                    f"{g.status}, so the two sides disagree about whether "
                    f"any work is outstanding",
                ))
            return

        # ---- hedged updates ------------------------------------------------
        if v.act == A.AMBIGUOUS_UPDATE:
            if g.status in TERMINAL or g.status_history:
                # Something firm is already on record; a hedge cannot move it.
                g.conflicts.append(Conflict(
                    "hedged_contradiction" if g.status in TERMINAL else "hedged_note",
                    [mid],
                    "a later message hedges about this subject without "
                    "asserting anything firm, so the recorded status is "
                    "unchanged",
                ))
            elif g.status != UNCLEAR:
                self._transition(g, mid, ts, UNCLEAR, v.act,
                                 "the only statement made about this subject is hedged")
            else:
                g.status_reason = (
                    "the only statements made about this subject are hedged "
                    f'("{v.evidence}"), so no firm status can be asserted'
                )
                g.status_source = mid
            return

        # ---- restatements --------------------------------------------------
        if v.repetition and g.has_firm_act(v.act):
            g.restatement_count += 1
            return

        # ---- firm acts -----------------------------------------------------
        if v.act in A.FIRM_ACTS and v.act not in g.applied_acts:
            g.applied_acts.append(v.act)

        if v.act in (A.NEW_TASK, A.NEW_EVENT):
            if not g.status_history:
                self._transition(g, mid, ts, PENDING, v.act,
                                 "the subject was raised as new work")
            self._set_deadline(g, inp, member)
            return

        if v.act == A.PROGRESS_UPDATE:
            if g.status not in TERMINAL:
                self._transition(g, mid, ts, IN_PROGRESS, v.act,
                                 "a message asserts that work has started")
            return

        if v.act == A.COMPLETION:
            if g.status == CANCELLED:
                g.contested = True
                g.conflicts.append(Conflict(
                    "completed_after_cancelled", [g.status_source or "", mid],
                    "the subject was reported complete after being cancelled",
                ))
            self._transition(g, mid, ts, COMPLETED, v.act,
                             "a message reports the work finished")
            g.response_required = False
            return

        if v.act == A.CANCELLATION:
            if g.status == COMPLETED:
                g.contested = True
                g.conflicts.append(Conflict(
                    "cancelled_after_completed", [g.status_source or "", mid],
                    "the subject was cancelled after being reported complete",
                ))
            self._transition(g, mid, ts, CANCELLED, v.act,
                             "a message withdraws the request")
            g.response_required = False
            return

        if v.act == A.RESCHEDULE:
            if g.status in TERMINAL:
                g.contested = True
                g.conflicts.append(Conflict(
                    "rescheduled_after_closure", [mid, g.status_source or ""],
                    f"the schedule was changed after the subject was "
                    f"reported {g.status}",
                ))
            else:
                self._transition(g, mid, ts, RESCHEDULED, v.act,
                                 "a message moves the meeting to a new slot")
            self._set_schedule(g, inp)
            return

        if v.act == A.DEADLINE_CHANGE:
            if g.status in TERMINAL:
                g.contested = True
                g.conflicts.append(Conflict(
                    "deadline_changed_after_closure", [mid, g.status_source or ""],
                    f"the deadline was changed after the subject was "
                    f"reported {g.status}",
                ))
            elif not g.status_history:
                self._transition(g, mid, ts, PENDING, v.act,
                                 "the subject carries a deadline and nothing "
                                 "reports it finished")
            self._set_deadline(g, inp, member)
            return

    def _transition(self, g: MessageGroup, mid: str, ts: str, new: str,
                    act: str, reason: str) -> None:
        if g.status == new and g.status_history:
            return
        g.status_history.append(
            Transition(message_id=mid, timestamp=ts, previous=g.status,
                       new=new, act=act, reason=reason))
        g.status = new
        g.status_reason = reason
        g.status_source = mid

    def _set_deadline(self, g: MessageGroup, inp: SubjectInput,
                      member: Member) -> None:
        v = inp.verdict
        origin = "l1_request" if v.frame == "l1_extraction" else "l2_update"
        new = v.new_date or (inp.item_date if v.act == A.NEW_TASK else None)
        if new:
            direction = _direction(g.latest_deadline, new)
            g.deadline_history.append(Change(
                message_id=inp.message_id, timestamp=inp.timestamp,
                field="deadline", previous=g.latest_deadline, new=new,
                direction=direction, flagged_conflict=v.conflict_marker,
                origin=origin,
            ))
            if origin == "l1_request":
                if direction in ("earlier", "later"):
                    g.deadline_restatements += 1
            elif direction in ("earlier", "later") and g.latest_deadline:
                g.conflicts.append(Conflict(
                    "deadline_changed", [inp.message_id],
                    f"the deadline moved from {g.latest_deadline} to {new} "
                    f"({direction})"
                    + (" and the sender flagged the earlier message as "
                       "listing another date" if v.conflict_marker else ""),
                ))
            g.latest_deadline = new
            g.latest_deadline_source = inp.message_id
            # A newly stated absolute date supersedes any pending relative one.
            g.pending_relative_deadline = None
            g.pending_relative_source = None
        elif v.new_date_raw:
            g.pending_relative_deadline = v.new_date_raw + (
                f" at {v.new_time}" if v.new_time else "")
            g.pending_relative_source = inp.message_id
            g.deadline_history.append(Change(
                message_id=inp.message_id, timestamp=inp.timestamp,
                field="deadline", previous=g.latest_deadline, new=None,
                raw=g.pending_relative_deadline, direction="unresolved",
                flagged_conflict=v.conflict_marker,
            ))

    def _set_schedule(self, g: MessageGroup, inp: SubjectInput) -> None:
        v = inp.verdict
        if v.new_date:
            g.schedule_history.append(Change(
                message_id=inp.message_id, timestamp=inp.timestamp, field="date",
                previous=g.latest_date, new=v.new_date,
                direction=_direction(g.latest_date, v.new_date)))
            g.latest_date = v.new_date
            g.schedule_source = inp.message_id
        if v.new_time:
            g.schedule_history.append(Change(
                message_id=inp.message_id, timestamp=inp.timestamp, field="time",
                previous=g.latest_time, new=v.new_time,
                direction="unchanged" if g.latest_time == v.new_time else "set"))
            g.latest_time = v.new_time
            g.schedule_source = inp.message_id

    # -- driver -------------------------------------------------------------
    def add(self, inp: SubjectInput) -> Optional[MessageGroup]:
        sig, variants, source = subject_for(inp)
        if sig is None:
            return None

        found = self._find_group(sig)
        if found is None:
            kind = "event" if (inp.item_type == "event"
                               or inp.verdict.act in (A.NEW_EVENT, A.RESCHEDULE)) \
                else ("task" if inp.item_type == "task" else "subject")
            g = self._new_group(sig, kind)
            self.groups.append(g)
            score, tokens, seed = 1.0, sorted(sig.tokens), True
        else:
            g, score, tokens = found
            seed = False
            # Keep the richest phrasing as the group's title and signature: a
            # later message that names the subject more fully improves the
            # group's identity, and matching against every variant means a
            # sparse reference like "the assignment" still lands here.
            if len(sig.specific) > len(g.signature.specific):
                g.signature = sig
                g.title = self._title(sig) or g.title
            if all(sig.tokens != v.tokens for v in g.variants) \
                    and sig.tokens != g.signature.tokens:
                g.variants.append(sig)
            if g.kind == "subject" and inp.item_type:
                g.kind = inp.item_type

        for extra in variants:
            if all(extra.tokens != v.tokens for v in g.variants) \
                    and extra.tokens != g.signature.tokens:
                g.variants.append(extra)

        member = Member(
            message_id=inp.message_id, timestamp=inp.timestamp,
            sender=inp.sender, act=inp.verdict.act,
            act_label=A.HUMAN_ACT.get(inp.verdict.act, inp.verdict.act),
            frame=inp.verdict.frame, hedged=inp.verdict.hedged,
            repetition=inp.verdict.repetition, match_score=round(score, 4),
            match_tokens=tokens, weak_link=(not seed and len(tokens) == 1),
            seed=seed, item_id=inp.item_id,
        )
        g.members.append(member)
        if inp.item_id and inp.item_id not in g.item_ids:
            g.item_ids.append(inp.item_id)
        if not g.first_seen:
            g.first_seen = inp.timestamp
        g.last_seen = inp.timestamp

        # Seed schedule and deadline from what L1 already extracted, so an L2
        # follow-up has a baseline to change rather than starting from nothing.
        # Skipped when the message's own act is about to set the same field --
        # otherwise the first history entry is a no-op change from a value to
        # itself.
        sets_own = inp.verdict.act in (A.NEW_TASK, A.NEW_EVENT,
                                       A.DEADLINE_CHANGE, A.RESCHEDULE)
        if not sets_own and (seed or (g.latest_deadline is None
                                      and g.latest_date is None)):
            if inp.item_type == "task" and inp.item_date and not g.latest_deadline:
                g.latest_deadline = inp.item_date
                g.latest_deadline_source = inp.message_id
            if inp.item_type == "event":
                if inp.item_date and not g.latest_date:
                    g.latest_date = inp.item_date
                    g.schedule_source = inp.message_id
                if inp.item_time and not g.latest_time:
                    g.latest_time = inp.item_time

        if source == "open_question":
            member.weak_link = True

        self._apply(g, inp, member)
        return g

    # -- finishing ----------------------------------------------------------
    def finalise(self) -> List[MessageGroup]:
        for g in self.groups:
            if not g.status_history and g.status == UNCLEAR and g.members:
                # Messages exist, none of them said anything firm. For a subject
                # L1 already turned into a task or event, "pending" is the
                # honest reading; otherwise it stays unclear.
                if g.item_ids:
                    g.status = PENDING
                    g.status_reason = (
                        "L1 extracted a task or event for this subject and no "
                        "later message reports it finished, cancelled or moved"
                    )
                    g.status_source = g.members[0].message_id
            g.summary = summarise(g)
            g.confidence = score_group(g)
        return self.groups


def summarise(g: MessageGroup) -> str:
    """A summary built only from what was actually recorded.

    Every clause below is generated from the group's own history, so there is
    nothing in the sentence that some message did not say. No paraphrasing
    model is involved, and none could be: the brief forbids sending message
    text to an external service, and inventing narrative locally would be the
    same failure with a shorter network path.
    """
    parts: List[str] = []
    first = g.members[0] if g.members else None
    if first:
        day = (first.timestamp or "")[:10]
        parts.append(f"Raised on {day} by {first.sender}" if day
                     else f"Raised by {first.sender}")

    if g.latest_deadline:
        src = f" (from {g.latest_deadline_source})" if g.latest_deadline_source else ""
        moves = [c for c in g.deadline_history
                 if c.direction in ("earlier", "later") and c.origin == "l2_update"]
        if moves:
            parts.append(
                f"the deadline changed {len(moves)} time(s) and now stands at "
                f"{g.latest_deadline}{src}")
        else:
            parts.append(f"deadline {g.latest_deadline}{src}")
    if g.pending_relative_deadline:
        parts.append(
            f'the newest deadline was given only as "{g.pending_relative_deadline}" '
            f"in {g.pending_relative_source} and has not been resolved to a date")

    if g.latest_date or g.latest_time:
        when = " at ".join(x for x in (g.latest_date, g.latest_time) if x)
        parts.append(f"latest schedule {when}"
                     + (f" (from {g.schedule_source})" if g.schedule_source else ""))

    if g.deadline_restatements:
        parts.append(f"{g.deadline_restatements} original request(s) restated the "
                     f"subject with a different date")
    if g.chase_count:
        parts.append(f"chased {g.chase_count} time(s)")
    if g.restatement_count:
        parts.append(f"{g.restatement_count} restatement(s) of earlier messages")

    if g.status in TERMINAL and g.status_source:
        verb = "reported complete" if g.status == COMPLETED else "cancelled"
        parts.append(f"{verb} in {g.status_source}")
    elif g.status == RESCHEDULED:
        parts.append("currently rescheduled")
    elif g.status == UNCLEAR:
        parts.append("no firm status has been stated")

    if g.contested:
        parts.append("the messages contradict each other, so the status is contested")
    if g.unresolved_alternatives:
        parts.append("mutually exclusive options were offered without a choice: "
                     + " or ".join(g.unresolved_alternatives))

    return "; ".join(parts) + "."


def score_group(g: MessageGroup) -> float:
    """Confidence that this group is one coherent subject with this status.

    Two independent things are being asserted, so both must be paid for: that
    the messages belong together, and that the status is right.
    """
    links = [m.match_score for m in g.members if not m.seed]
    cohesion = sum(links) / len(links) if links else 0.86   # lone message

    weak = sum(1 for m in g.members if m.weak_link)
    if weak:
        cohesion -= 0.06 * min(weak, 3)

    if g.status_history:
        status_conf = 0.90
    elif g.item_ids:
        status_conf = 0.78          # inferred from L1, never asserted
    else:
        status_conf = 0.55          # nothing firm was ever said

    if g.contested:
        status_conf -= 0.22
    if g.unresolved_alternatives:
        status_conf -= 0.10
    if g.conflicts:
        status_conf -= 0.04 * min(len(g.conflicts), 3)

    return round(max(0.30, min(0.5 * cohesion + 0.5 * status_conf + 0.08, 0.97)), 4)


def build(inputs: Sequence[SubjectInput]) -> List[MessageGroup]:
    """Group a chronologically ordered sequence of messages.

    Pass 1 gathers the subject vocabulary; pass 2 walks time forwards. The
    caller is responsible for the ordering -- `mint/ledger.py` enforces it.
    """
    signatures: List[SubjectSignature] = []
    for inp in inputs:
        sig, variants, _ = subject_for(inp)
        if sig:
            signatures.append(sig)
        signatures.extend(variants)

    builder = GroupBuilder(SubjectSpace(signatures))
    for inp in inputs:
        builder.add(inp)
    return builder.finalise()
