"""The priority and action engine.

Part 1 of the L2 brief: give every actionable item a priority, explain it, and
update it when a later message changes the deadline, the urgency or the status.

The brief is explicit that priority must not be "random or only using one
keyword", so this is a **weighted signal model**, not a keyword lookup. Each
signal is a named, independently-detected fact with a fixed weight and a
sentence explaining what it means. The score is their sum; the band is a
threshold on the score; and the reason is assembled from the signals that
actually fired. Nothing about a decision is hidden -- reading the signal list
tells you exactly why the number came out where it did, and changing one weight
changes exactly one thing.

Two design decisions carry most of the weight.

**Priority is a property of the subject, not of a sentence.**
"Can you review the privacy checklist before 2026-09-09?" is not urgent because
of anything in the sentence; it is urgent because that deadline has passed and
thirteen later messages are still chasing it. So the engine scores the *group*
built in `mint/groups.py` -- its current status, its latest deadline, how often
it has been chased -- and every item extracted from that subject inherits the
result. A message read alone can only ever be scored on what it says alone.

**"Now" is the newest message, not the wall clock.**
Deadline proximity needs a reference point. Using the real current time would
make the same input produce different output on different days, which makes
results impossible to check and impossible to demonstrate. So the reference
time is the timestamp of the last message processed. Load one more batch and
every deadline moves closer -- which is exactly the behaviour the brief asks
for when it says priority must update, and it is reproducible.

A consequence worth stating plainly: priority can change with no new message at
all, simply because time advanced past a deadline. The engine records which of
the two happened, because "this became critical because someone escalated it"
and "this became critical because you ignored it" are different facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from . import groups as G

CRITICAL = "critical"
HIGH = "high"
MEDIUM = "medium"
LOW = "low"

BANDS = [CRITICAL, HIGH, MEDIUM, LOW]

#: Score at or above which each band applies, most severe first.
THRESHOLDS: List[Tuple[str, float]] = [
    (CRITICAL, 4.6),
    (HIGH, 3.0),
    (MEDIUM, 1.5),
]

#: A signal's family. The bands are a sum, but `critical` additionally requires
#: corroboration from two *different* families -- a single very loud signal is
#: not enough to demand that a human drop everything, and the brief explicitly
#: rules out deciding on one keyword.
DEADLINE = "deadline"
URGENCY = "urgency"
PRESSURE = "pressure"
SENDER = "sender"
SENSITIVITY = "sensitivity"
STATUS = "status"
CATEGORY = "category"


@dataclass(frozen=True)
class SignalSpec:
    name: str
    family: str
    weight: float
    explain: str


#: Every signal the engine can raise. Keeping them in one table means the
#: weights can be read, compared and argued with in one place.
SIGNALS: Dict[str, SignalSpec] = {s.name: s for s in [
    # -- deadline proximity ------------------------------------------------
    SignalSpec("deadline_overdue", DEADLINE, 3.2,
               "the deadline on record has already passed"),
    SignalSpec("deadline_today", DEADLINE, 2.4,
               "the deadline falls on the day of the most recent message"),
    SignalSpec("deadline_tomorrow", DEADLINE, 2.0,
               "the deadline is the day after the most recent message"),
    SignalSpec("deadline_within_3_days", DEADLINE, 1.4,
               "the deadline is within three days"),
    SignalSpec("deadline_within_7_days", DEADLINE, 0.7,
               "the deadline is within a week"),
    SignalSpec("deadline_beyond_7_days", DEADLINE, -0.5,
               "the deadline is more than a week away"),
    # Stated relatively ("due tomorrow"). Worth slightly less than an absolute
    # date because it was never resolved to one -- the proximity is asserted,
    # not verified.
    SignalSpec("deadline_relative_imminent", DEADLINE, 2.0,
               "the newest message says the deadline is today or tomorrow "
               "without giving a date"),
    SignalSpec("deadline_unresolved", DEADLINE, 0.0,
               "a deadline was mentioned but never resolved to a date"),
    SignalSpec("no_deadline", DEADLINE, 0.0,
               "no deadline has been stated for this subject"),
    SignalSpec("deadline_moved_earlier", DEADLINE, 0.8,
               "a later message brought the deadline forward"),
    SignalSpec("deadline_moved_later", DEADLINE, -0.4,
               "a later message pushed the deadline back"),
    SignalSpec("deadline_conflict", DEADLINE, 0.4,
               "two messages give different deadlines for this subject"),

    # -- urgency wording ----------------------------------------------------
    SignalSpec("explicit_urgency", URGENCY, 1.8,
               "a message about this subject uses explicit urgency wording"),
    SignalSpec("de_escalated", URGENCY, -1.6,
               "a message says this is no longer urgent"),

    # -- follow-up pressure -------------------------------------------------
    SignalSpec("chased_once", PRESSURE, 0.4,
               "someone has asked for a status update"),
    SignalSpec("chased_repeatedly", PRESSURE, 1.4,
               "the subject has been chased three or more times"),
    SignalSpec("response_required", PRESSURE, 0.5,
               "an unanswered question about this subject is outstanding"),
    SignalSpec("chased_after_closure", PRESSURE, 0.6,
               "the subject was chased after it was reported closed, so the "
               "two sides disagree about whether work is outstanding"),

    # -- who is asking ------------------------------------------------------
    SignalSpec("authority_sender", SENDER, 0.5,
               "the request comes from a role account rather than a peer"),

    # -- data handling ------------------------------------------------------
    SignalSpec("sensitive_high_risk", SENSITIVITY, 0.4,
               "the source message carries a high-risk value, so it needs "
               "handling before it is stored or forwarded"),

    # -- lifecycle ----------------------------------------------------------
    SignalSpec("status_completed", STATUS, -6.0,
               "the subject has been reported complete"),
    SignalSpec("status_cancelled", STATUS, -6.0,
               "the subject has been cancelled"),
    SignalSpec("status_unclear", STATUS, -0.2,
               "no firm statement has been made about this subject"),
    SignalSpec("status_contested", STATUS, 0.3,
               "the messages contradict each other about this subject"),

    # -- what kind of thing it is -------------------------------------------
    SignalSpec("category_action_required", CATEGORY, 0.4,
               "the subject is something the recipient must do"),
    SignalSpec("category_meeting_event", CATEGORY, 0.25,
               "the subject is something the recipient must attend"),
]}

#: Roles whose requests carry organisational weight. Deliberately a small,
#: explicit list rather than an inference from the name: guessing seniority
#: from a display name is exactly the kind of invention the brief rules out.
AUTHORITY_SENDERS = frozenset({
    "hr team", "project lead", "mentor", "operations", "management",
    "compliance", "finance",
})


@dataclass
class Signal:
    name: str
    family: str
    weight: float
    detail: str

    def to_dict(self) -> dict:
        return {"signal": self.name, "family": self.family,
                "weight": self.weight, "detail": self.detail}


@dataclass
class PriorityDecision:
    """One priority decision, with everything that produced it."""

    priority: str
    score: float
    confidence: float
    reason: str
    signals: List[Signal] = field(default_factory=list)

    group_id: Optional[str] = None
    item_id: Optional[str] = None
    message_id: Optional[str] = None
    as_of: str = ""
    needs_review: bool = False
    #: Set when the score cleared the critical threshold but only one family of
    #: evidence supported it, so the engine held it at high.
    gated_from_critical: bool = False

    @property
    def signal_names(self) -> List[str]:
        return [s.name for s in self.signals]

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "item_id": self.item_id,
            "group_id": self.group_id,
            "priority": self.priority,
            "reason": self.reason,
            "signals": self.signal_names,
            "confidence": self.confidence,
            "score": self.score,
            "as_of": self.as_of,
            "needs_review": self.needs_review,
            "gated_from_critical": self.gated_from_critical,
            "signal_detail": [s.to_dict() for s in self.signals],
        }


@dataclass
class PriorityChange:
    """A recorded move between bands, and what caused it."""

    as_of: str
    previous: str
    new: str
    trigger: str                       # "message" | "elapsed_time" | "initial"
    trigger_message_ids: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {"as_of": self.as_of, "from": self.previous, "to": self.new,
                "trigger": self.trigger,
                "trigger_message_ids": self.trigger_message_ids,
                "reason": self.reason}


def _parse_day(value: Optional[str]):
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M",
                "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def _band(score: float) -> str:
    for name, floor in THRESHOLDS:
        if score >= floor:
            return name
    return LOW


def _deadline_signals(group: G.MessageGroup, today) -> List[Signal]:
    out: List[Signal] = []
    spec = SIGNALS

    def add(name: str, detail: str) -> None:
        s = spec[name]
        out.append(Signal(s.name, s.family, s.weight, detail))

    if group.pending_relative_deadline:
        phrase = group.pending_relative_deadline.lower()
        if any(w in phrase for w in ("today", "tomorrow", "tonight")):
            add("deadline_relative_imminent",
                f'{group.pending_relative_source} states the deadline as '
                f'"{group.pending_relative_deadline}"; it was not resolved to a '
                f'date, so the proximity is taken from the wording')
        else:
            add("deadline_unresolved",
                f'the newest deadline is only "{group.pending_relative_deadline}" '
                f'({group.pending_relative_source}) and cannot be placed on the '
                f'calendar')

    due = _parse_day(group.latest_deadline) or _parse_day(group.latest_date)
    label = "deadline" if group.latest_deadline else "event"
    if due is None:
        if not group.pending_relative_deadline:
            add("no_deadline", "no message has given a date for this subject")
    elif today is not None:
        days = (due - today).days
        src = group.latest_deadline_source or group.schedule_source or "?"
        if days < 0:
            add("deadline_overdue",
                f"the {label} of {due.isoformat()} ({src}) passed "
                f"{abs(days)} day(s) before the most recent message")
        elif days == 0:
            add("deadline_today", f"the {label} of {due.isoformat()} ({src}) "
                                  f"is the day of the most recent message")
        elif days == 1:
            add("deadline_tomorrow", f"the {label} of {due.isoformat()} ({src}) "
                                     f"is one day after the most recent message")
        elif days <= 3:
            add("deadline_within_3_days",
                f"the {label} of {due.isoformat()} ({src}) is {days} days away")
        elif days <= 7:
            add("deadline_within_7_days",
                f"the {label} of {due.isoformat()} ({src}) is {days} days away")
        else:
            add("deadline_beyond_7_days",
                f"the {label} of {due.isoformat()} ({src}) is {days} days away")

    moves = [c for c in group.deadline_history
             if c.origin == "l2_update" and c.direction in ("earlier", "later")]
    if moves:
        last = moves[-1]
        if last.direction == "earlier":
            add("deadline_moved_earlier",
                f"{last.message_id} moved the deadline from {last.previous} "
                f"forward to {last.new}")
        else:
            add("deadline_moved_later",
                f"{last.message_id} pushed the deadline from {last.previous} "
                f"back to {last.new}")
    if any(c.kind == "deadline_changed" and c.detail for c in group.conflicts) \
            and len(moves) > 1:
        add("deadline_conflict",
            f"{len(moves)} separate messages give different deadlines for this "
            f"subject")
    return out


def score_group(
    group: G.MessageGroup,
    as_of: str,
    category: str = "",
    sensitive_high_risk: bool = False,
    senders: Sequence[str] = (),
) -> PriorityDecision:
    """Score one subject group as of `as_of` (a message timestamp)."""
    today = _parse_day(as_of)
    signals: List[Signal] = []

    def add(name: str, detail: str) -> None:
        s = SIGNALS[name]
        signals.append(Signal(s.name, s.family, s.weight, detail))

    # -- lifecycle first: a closed subject is not a priority question -------
    if group.status == G.COMPLETED:
        add("status_completed",
            f"{group.status_source} reports this complete")
    elif group.status == G.CANCELLED:
        add("status_cancelled", f"{group.status_source} cancels this")
    elif group.status == G.UNCLEAR:
        add("status_unclear", group.status_reason)
    if group.contested:
        add("status_contested",
            "the recorded messages disagree about this subject's state")

    if group.status not in G.TERMINAL:
        signals.extend(_deadline_signals(group, today))

        if group.urgency_flagged:
            add("explicit_urgency",
                "at least one message about this subject uses explicit "
                "urgency wording")
        if group.de_escalated:
            add("de_escalated",
                "a message states that this is no longer urgent")

        if group.chase_count >= 3:
            add("chased_repeatedly",
                f"the subject has been chased {group.chase_count} times")
        elif group.chase_count >= 1:
            add("chased_once",
                f"the subject has been chased {group.chase_count} time(s)")
        if group.response_required:
            add("response_required",
                "the most recent traffic about this subject is an unanswered "
                "question")

        roles = sorted({s for s in senders if s.strip().lower()
                        in AUTHORITY_SENDERS})
        if roles:
            add("authority_sender",
                f"raised or chased by {', '.join(roles)}")
        if sensitive_high_risk:
            add("sensitive_high_risk",
                "a message in this group carries a high-risk value")
        if category == "meeting_or_event" or group.kind == "event":
            add("category_meeting_event", "this subject is a meeting or event")
        else:
            add("category_action_required",
                "this subject is work the recipient must do")
    else:
        if any(c.kind == "chased_after_closure" for c in group.conflicts):
            add("chased_after_closure",
                "the subject was chased after it was reported closed")

    score = round(sum(s.weight for s in signals), 3)
    band = _band(score)

    # -- the corroboration gate --------------------------------------------
    gated = False
    if band == CRITICAL:
        families = {s.family for s in signals if s.weight > 0
                    and s.family in (DEADLINE, URGENCY, PRESSURE, SENDER,
                                     SENSITIVITY)}
        if len(families) < 2:
            band = HIGH
            gated = True

    confidence = _confidence(group, signals, gated)
    reason = _reason(group, band, signals, gated)

    return PriorityDecision(
        priority=band, score=score, confidence=confidence, reason=reason,
        signals=signals, group_id=group.group_id, as_of=as_of,
        needs_review=(group.contested or bool(group.unresolved_alternatives)
                      or confidence < 0.6),
        gated_from_critical=gated,
    )


def _confidence(group: G.MessageGroup, signals: Sequence[Signal],
                gated: bool) -> float:
    """How much to trust this decision.

    Confidence is about the *inputs*, not the score. A perfectly-computed
    priority built on a deadline nobody ever resolved is a confident number
    resting on an uncertain fact, and saying so is the point.
    """
    conf = 0.62
    names = {s.name for s in signals}

    # Corroboration across independent families is the main source of trust.
    families = {s.family for s in signals if s.weight != 0}
    conf += 0.07 * min(len(families), 4)

    if "deadline_unresolved" in names or "no_deadline" in names:
        conf -= 0.14
    if "deadline_relative_imminent" in names:
        conf -= 0.06          # asserted proximity, never resolved to a date
    if group.contested:
        conf -= 0.16
    if group.unresolved_alternatives:
        conf -= 0.10
    if gated:
        conf -= 0.04
    # The grouping itself has to be right for any of this to mean anything.
    conf = conf * (0.6 + 0.4 * group.confidence)
    return round(max(0.25, min(conf, 0.96)), 4)


def _reason(group: G.MessageGroup, band: str, signals: Sequence[Signal],
            gated: bool) -> str:
    """A sentence assembled from the signals that actually carried the decision."""
    positive = sorted([s for s in signals if s.weight > 0],
                      key=lambda s: -s.weight)[:3]
    negative = sorted([s for s in signals if s.weight < 0],
                      key=lambda s: s.weight)[:2]

    if not positive and not negative:
        return (f"No priority signal fired for this subject, so it defaults to "
                f"{band}.")

    lead = "; ".join(s.detail for s in positive) if positive else ""
    trail = "; ".join(s.detail for s in negative) if negative else ""
    parts = [p for p in (lead, trail) if p]
    body = ". ".join(parts)
    out = f"Assessed {band}: {body}."
    if gated:
        out += (" Held at high rather than critical because only one kind of "
                "evidence supported it, and critical requires corroboration "
                "from two.")
    return out


def track(
    previous: Optional[PriorityDecision],
    current: PriorityDecision,
    new_message_ids: Sequence[str],
) -> Optional[PriorityChange]:
    """Record a band change and attribute it to a message or to elapsed time."""
    if previous is None:
        return PriorityChange(
            as_of=current.as_of, previous="-", new=current.priority,
            trigger="initial", trigger_message_ids=list(new_message_ids),
            reason="first assessment of this subject",
        )
    if previous.priority == current.priority:
        return None
    if new_message_ids:
        return PriorityChange(
            as_of=current.as_of, previous=previous.priority,
            new=current.priority, trigger="message",
            trigger_message_ids=list(new_message_ids),
            reason=(f"new messages about this subject changed its assessment "
                    f"from {previous.priority} to {current.priority}"),
        )
    return PriorityChange(
        as_of=current.as_of, previous=previous.priority, new=current.priority,
        trigger="elapsed_time", trigger_message_ids=[],
        reason=(f"no new message mentioned this subject; the assessment moved "
                f"from {previous.priority} to {current.priority} because the "
                f"reference time advanced past its deadline"),
    )
