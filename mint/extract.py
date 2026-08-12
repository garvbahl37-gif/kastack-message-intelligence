"""Task and event extraction.

The governing constraint from the brief is "do not guess missing information".
That is taken literally here:

* A date is only filled in when an **absolute** date appears in the text. A
  message saying "tomorrow" leaves ``date`` as ``null`` -- the raw phrase is
  preserved in ``date_raw`` and ``date_status`` becomes ``unresolved_relative``,
  so nothing is lost and nothing is invented. Resolving "tomorrow" against the
  message timestamp would usually be right, but "usually right" is exactly the
  kind of silent inference this system is supposed to avoid; it is offered as a
  clearly-labelled suggestion instead of an assertion.
* A person is only filled in when a known participant is actually named. The
  sender is recorded separately and never promoted into the ``person`` field --
  "who sent this" and "who is involved in the task" are different questions.
* Priority always carries the reason it was assigned.

Extraction is frame-based for the same reason classification is: the frames say
where the title, date, time and location live inside a sentence, which is what
lets the title come out as "review the privacy checklist" rather than as the
whole message.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import List, Optional, Pattern, Sequence, Tuple

from . import taxonomy as T
from .normalize import body_of

ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\b")
AMPM = re.compile(r"\b(\d{1,2})\s*(am|pm)\b", re.IGNORECASE)

#: Time expressions that refer to a real moment but cannot be resolved to a
#: calendar date without guessing.
RELATIVE_TIME = re.compile(
    r"\b(today|tomorrow|tonight|yesterday|this\s+(?:week|month|morning|afternoon|"
    r"evening)|next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)|sometime\s+next\s+week|soon|shortly|later|"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:\s+(?:morning|afternoon|evening))?|this\s+weekend|end\s+of\s+(?:the\s+)?"
    r"(?:day|week|month)|asap|every\s+\w+)\b",
    re.IGNORECASE,
)

URGENCY = re.compile(
    r"\b(urgent|urgently|asap|immediately|critical|right\s+away|top\s+priority|"
    r"as\s+soon\s+as\s+possible|high\s+priority)\b",
    re.IGNORECASE,
)

#: Capitalised strings that look like names but are not people. Venue words
#: matter most here: "Meeting Room A" and "Google Meet" are full of tokens that
#: pass every shallow name test.
NOT_A_PERSON = {
    "zoom", "google", "meet", "google meet", "meeting", "conference", "room",
    "auditorium", "cafeteria", "clinic", "library", "office", "hall", "centre",
    "center", "portal", "python", "wi-fi", "wifi", "ai", "hr", "otp", "id",
    "rc", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "please",
    "remember", "reminder", "calendar", "important", "special", "flash",
    "limited", "free", "use", "code", "save", "the", "location", "team",
    "project", "promotions", "workshop", "seminar", "orientation", "briefing",
    "interview", "demo", "review", "session", "dinner", "training", "main",
}

#: Grammatical positions that actually indicate a person. Used instead of a
#: blanket "any capitalised word" fallback, which happily returned "Meeting"
#: out of "Meeting Room A".
PERSON_CONTEXT = [
    re.compile(r"\b(?:call|email|contact|ask|remind|meet|inform|notify|tell|"
               r"chase|ping|thank)\s+(?P<name>[A-Z][a-z]{2,})\b"),
    re.compile(r"\b(?P<name>[A-Z][a-z]{2,})\s+(?:asked|said|wants|needs|"
               r"requested|mentioned|reported|confirmed|will\s+send)\b"),
    re.compile(r"\b(?:with|from|for|to|cc)\s+(?P<name>[A-Z][a-z]{2,})\b"),
]

_DEF = re.compile(r"^(?:the|a|an|our|your|my)\s+", re.IGNORECASE)


@dataclass
class ExtractedItem:
    """One task or event. Field names follow the brief's example schema."""

    item_id: str
    type: str                      # "task" | "event"
    title: str
    description: str
    date: Optional[str]            # ISO date, or None when not stated absolutely
    time: Optional[str]            # HH:MM 24h, or None
    person: Optional[str]
    priority: str                  # "high" | "medium" | "low"
    source_message_id: str

    # -- provenance and unresolved-field bookkeeping ------------------------
    deadline: Optional[str] = None      # alias of `date` for tasks
    location: Optional[str] = None
    date_raw: Optional[str] = None
    date_status: str = "resolved"       # resolved | unresolved_relative | missing
    date_suggestion: Optional[str] = None
    time_status: str = "resolved"
    person_status: str = "resolved"
    priority_reason: str = ""
    source_sender: str = ""
    source_timestamp: str = ""
    extraction_frame: str = ""
    unresolved_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
D = r"(?P<date>\d{4}-\d{2}-\d{2})"
TM = r"(?P<time>\d{1,2}:\d{2})"

EVENT_FRAMES: List[Tuple[str, Pattern[str]]] = [
    ("calendar_entry", re.compile(
        r"^calendar\s+(?:update|invite|entry)\s*:\s*(?P<title>[^,]+?)\s*,\s*"
        + D + r"\s+at\s+" + TM + r"\s*,\s*(?P<loc>.+?)\.?$", re.IGNORECASE)),
    ("reminder_occurrence", re.compile(
        r"^reminder\s*:\s*(?P<title>.+?)\s+happens\s+on\s+" + D
        + r"\s+at\s+" + TM + r"\s+(?:in|at)\s+(?P<loc>.+?)\.?$", re.IGNORECASE)),
    ("invitation", re.compile(
        r"\bplease\s+join\s+(?:the\s+|our\s+)?(?P<title>.+?)\s+on\s+" + D
        + r",?\s*(?:at\s+)?" + TM + r"\s+(?:at|in)\s+(?P<loc>.+?)\.?$",
        re.IGNORECASE)),
    ("scheduled", re.compile(
        r"\b(?:the\s+)?(?P<title>[a-z][a-z\s-]+?)\s+is\s+scheduled\s+for\s+" + D
        + r"\s+at\s+" + TM + r"\s+(?:in|at)\s+(?P<loc>.+?)\.?$", re.IGNORECASE)),
    ("availability_probe", re.compile(
        r"\bare\s+you\s+(?:available|free)\s+for\s+(?:the\s+)?(?P<title>.+?)\s+at\s+"
        + TM + r"\s+on\s+" + D + r"\s*\?\s*location\s*:\s*(?P<loc>.+?)\.?$",
        re.IGNORECASE)),
    # Under-specified events: real, but with nothing concrete to resolve.
    ("meet_proposal", re.compile(
        r"\blet(?:'s|\s+us)\s+meet\b(?P<title>)(?P<loc>)", re.IGNORECASE)),
    ("hedged_event", re.compile(
        r"\bthe\s+(?P<title>meeting|review|discussion|session|call|sync|demo|"
        r"interview|catch[-\s]?up|stand[-\s]?up|appointment|briefing)\s+"
        r"(?:could|might|may|should)\s+be\b(?P<loc>)", re.IGNORECASE)),
]

TASK_FRAMES: List[Tuple[str, Pattern[str]]] = [
    ("request_with_deadline", re.compile(
        r"\b(?:can|could|would|will)\s+you\s+(?P<title>.+?)\s+"
        r"(?:by|before|no\s+later\s+than)\s+" + D, re.IGNORECASE)),
    ("explicit_request_with_deadline", re.compile(
        r"\bi\s+need\s+you\s+to\s+(?P<title>.+?)\s+"
        r"(?:by|before|no\s+later\s+than)\s+" + D, re.IGNORECASE)),
    ("directive_with_deadline", re.compile(
        r"\b(?:please|kindly)\s+(?P<title>.+?)\s+"
        r"(?:by|before|no\s+later\s+than)\s+" + D, re.IGNORECASE)),
    ("reminder_with_deadline", re.compile(
        r"\b(?:don'?t\s+forget\s+to|remember\s+to|make\s+sure\s+(?:you\s+|to\s+)?)"
        r"(?P<title>.+?)\s*[;,]?\s*deadline\s+is\s+" + D, re.IGNORECASE)),
    ("due_statement", re.compile(
        r"^(?P<title>.+?)\s+is\s+due\s+(?:on|by)\s+" + D, re.IGNORECASE)),
    # Deadline-free task frames -- still tasks, just without a resolvable date.
    ("conditional_request", re.compile(
        r"\bif\s+possible,?\s+(?P<title>.+?)\.?$", re.IGNORECASE)),
    ("explicit_request", re.compile(
        r"\bi\s+need\s+you\s+to\s+(?P<title>.+?)\.?$", re.IGNORECASE)),
    ("reminder_directive", re.compile(
        r"\b(?:don'?t\s+forget\s+to|remember\s+to)\s+(?P<title>.+?)\.?$",
        re.IGNORECASE)),
    ("directive", re.compile(
        r"\b(?:please|kindly)\s+(?P<title>.+?)\s*[.?]?$", re.IGNORECASE)),
    ("request_question", re.compile(
        r"\b(?:can|could|would)\s+you\s+(?P<title>.+?)\s*\??$", re.IGNORECASE)),
    ("relayed_request", re.compile(
        r"\b(?P<who>[A-Z][a-z]+)\s+asked\s+(?:whether|if)\s+(?P<title>.+?)\.?$")),
    ("hedged_need", re.compile(
        r"^(?P<title>.+?)\s+(?:may|might)\s+be\s+needed\b", re.IGNORECASE)),
]


def _clean_title(raw: str) -> str:
    t = re.sub(r"\s+", " ", (raw or "").strip(" \t.,;:!?"))
    t = _DEF.sub("", t)
    if not t:
        return ""
    return t[0].upper() + t[1:]


def _find_person(text: str, roster: Sequence[str]) -> Optional[str]:
    """Return a named participant, or None. Never falls back to the sender."""
    lowered = text.lower()
    for name in roster:
        n = name.strip()
        if not n or n.lower() in NOT_A_PERSON:
            continue
        if re.search(rf"\b{re.escape(n.lower())}\b", lowered):
            return n
    # Otherwise require the capitalised token to sit in a position that
    # actually implies a person -- being addressed, doing something, or being
    # named as a counterpart. A bare capitalised word is not enough.
    for pat in PERSON_CONTEXT:
        for m in pat.finditer(text):
            cand = m.group("name")
            if cand.lower() not in NOT_A_PERSON:
                return cand
    return None


def _parse_time(text: str) -> Optional[str]:
    m = CLOCK.search(text)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mi <= 59:
            return f"{h:02d}:{mi:02d}"
    m = AMPM.search(text)
    if m:
        h = int(m.group(1)) % 12
        if m.group(2).lower() == "pm":
            h += 12
        return f"{h:02d}:00"
    return None


def _suggest_date(phrase: str, sent_on: Optional[date]) -> Optional[str]:
    """What the relative phrase would resolve to -- offered, never asserted."""
    if sent_on is None:
        return None
    p = phrase.lower().strip()
    from datetime import timedelta

    if p in ("today", "tonight", "this morning", "this afternoon", "this evening"):
        return sent_on.isoformat()
    if p == "tomorrow":
        return (sent_on + timedelta(days=1)).isoformat()
    if p == "yesterday":
        return (sent_on - timedelta(days=1)).isoformat()
    return None


def _priority(
    resolved_date: Optional[str],
    sent_on: Optional[date],
    text: str,
    kind: str,
) -> Tuple[str, str]:
    """Priority plus the reason for it. Never silently defaults."""
    if URGENCY.search(text):
        m = URGENCY.search(text)
        return "high", f'the message uses explicit urgency wording ("{m.group(0)}")'

    if resolved_date and sent_on:
        try:
            days = (datetime.strptime(resolved_date, "%Y-%m-%d").date() - sent_on).days
        except ValueError:
            days = None
        noun = "event" if kind == "event" else "deadline"
        if days is not None:
            if days < 0:
                return "high", f"the stated {noun} is {abs(days)} day(s) in the past"
            if days <= 3:
                return "high", f"the {noun} is {days} day(s) after the message"
            if days <= 7:
                return "medium", f"the {noun} is {days} day(s) after the message"
            return "low", f"the {noun} is {days} day(s) away"

    if resolved_date:
        return "medium", "a firm date is given but the send date is unknown"

    if kind == "event":
        return "medium", "an event is stated but its date could not be resolved"
    return "low", "no deadline could be resolved from the message"


def _sent_date(timestamp: str) -> Optional[date]:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(timestamp.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def extract(
    message_id: str,
    masked_text: str,
    sender: str,
    timestamp: str,
    category: str,
    item_index: int,
    roster: Sequence[str] = (),
) -> Optional[ExtractedItem]:
    """Extract a task or event from one classified message, or None.

    Only ``action_required`` and ``meeting_or_event`` messages are considered;
    the other categories carry nothing schedulable by definition.
    """
    if category not in (T.ACTION_REQUIRED, T.MEETING_EVENT):
        return None

    body = body_of(masked_text)
    sent_on = _sent_date(timestamp)
    frames = EVENT_FRAMES if category == T.MEETING_EVENT else TASK_FRAMES
    kind = "event" if category == T.MEETING_EVENT else "task"

    frame_name, match = "", None
    for name, pat in frames:
        m = pat.search(body)
        if m:
            frame_name, match = name, m
            break

    if match is None:
        # The classifier says there is something here but no frame located it.
        # Emit a deliberately empty record rather than inventing structure.
        frame_name = "unmatched"

    groups = match.groupdict() if match else {}
    title = _clean_title(groups.get("title") or "")
    if not title:
        title = "Meeting" if kind == "event" else _clean_title(body)
    location = _clean_title(groups.get("loc") or "") or None

    # -- date ---------------------------------------------------------------
    resolved_date = groups.get("date")
    if not resolved_date:
        m = ISO_DATE.search(body)
        resolved_date = m.group(1) if m else None

    date_raw: Optional[str] = None
    date_status = "resolved"
    date_suggestion: Optional[str] = None
    if not resolved_date:
        rel = RELATIVE_TIME.search(body)
        if rel:
            date_raw = rel.group(0)
            date_status = "unresolved_relative"
            date_suggestion = _suggest_date(date_raw, sent_on)
        else:
            date_status = "missing"

    # -- time ---------------------------------------------------------------
    # Normalise through _parse_time so a captured "9:00" becomes "09:00" and
    # every downstream consumer sees one 24-hour format.
    raw_time = groups.get("time")
    resolved_time = _parse_time(raw_time) if raw_time else _parse_time(body)
    time_status = "resolved" if resolved_time else "missing"

    # -- person -------------------------------------------------------------
    person = groups.get("who") or _find_person(body, roster)
    person_status = "resolved" if person else "missing"

    priority, priority_reason = _priority(resolved_date, sent_on, body, kind)

    unresolved = []
    if date_status != "resolved":
        unresolved.append("date")
    if time_status != "resolved":
        unresolved.append("time")
    if person_status != "resolved":
        unresolved.append("person")
    if frame_name == "unmatched":
        unresolved.append("title")

    prefix = "EVENT" if kind == "event" else "TASK"
    return ExtractedItem(
        item_id=f"{prefix}_{item_index:04d}",
        type=kind,
        title=title,
        description=body,
        date=resolved_date,
        time=resolved_time,
        person=person,
        priority=priority,
        source_message_id=message_id,
        deadline=resolved_date if kind == "task" else None,
        location=location,
        date_raw=date_raw,
        date_status=date_status,
        date_suggestion=date_suggestion,
        time_status=time_status,
        person_status=person_status,
        priority_reason=priority_reason,
        source_sender=sender,
        source_timestamp=timestamp,
        extraction_frame=frame_name,
        unresolved_fields=unresolved,
    )
