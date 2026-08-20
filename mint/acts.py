"""Speech acts: what a message *does* to the subject it names.

L1 classified messages by what they are. That is not enough for L2, because
four messages can share one category and still mean four different things:

    "Can you share an update on review the privacy checklist?"   -> still open
    "Update: review the privacy checklist has been completed."    -> closed
    "You can cancel review the privacy checklist."                -> abandoned
    "The privacy checklist might already be done, I'm not sure."  -> unknown

All four are *about* the same task. Only the third and fourth are not requests.
Grouping them together is Part 2 of the brief; deciding which of them is true
*now* is what makes the group's status meaningful, and it is what the priority
engine needs in order to stop chasing a task that was finished yesterday.

So this module answers a narrow question with a frame grammar, in the same
style as `mint/rules.py`: **which act is this message performing, and on what
subject?** Every frame captures the subject span as a named group, so act
detection and subject extraction happen in one pass and can never disagree
about which words were the subject.

Two properties matter more than coverage:

* **Hedges are first-class.** "might already be done" is not a completion. The
  frame that matches it produces ``AMBIGUOUS_UPDATE``, and downstream nothing
  is allowed to promote it into a firm status. The brief asks for an "unclear"
  status and for uncertainty not to be invented away; this is where that is
  enforced.
* **Repetition is recorded, not collapsed.** "Follow-up:" and "Additional
  update:" prefixes mark a message as a re-send. The prefix is stripped before
  frames run (otherwise every frame would need a variant), but the fact is kept:
  a subject chased four times is a stronger priority signal than one chased once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Pattern, Tuple

from .normalize import body_of

# --------------------------------------------------------------------------
# The act vocabulary
# --------------------------------------------------------------------------
PROGRESS_UPDATE = "progress_update"
NEW_TASK = "new_task"
NEW_EVENT = "new_event"
STATUS_QUERY = "status_query"
COMPLETION = "completion"
CANCELLATION = "cancellation"
RESCHEDULE = "reschedule"
DEADLINE_CHANGE = "deadline_change"
AMBIGUOUS_UPDATE = "ambiguous_update"
OPEN_QUESTION = "open_question"
INFORMATIONAL = "informational"

#: Acts that assert a firm, durable fact about a subject's lifecycle. Only
#: these may overwrite a status; everything else annotates it.
FIRM_ACTS = frozenset({NEW_TASK, NEW_EVENT, COMPLETION, CANCELLATION,
                       RESCHEDULE, DEADLINE_CHANGE, PROGRESS_UPDATE})

#: Acts that mean "someone is still waiting" -- a response is required.
RESPONSE_REQUIRED_ACTS = frozenset({STATUS_QUERY, OPEN_QUESTION})

HUMAN_ACT = {
    PROGRESS_UPDATE: "work started / in progress",
    NEW_TASK: "new task raised",
    NEW_EVENT: "new meeting or event announced",
    STATUS_QUERY: "status chased",
    COMPLETION: "reported complete",
    CANCELLATION: "cancelled",
    RESCHEDULE: "rescheduled",
    DEADLINE_CHANGE: "deadline changed",
    AMBIGUOUS_UPDATE: "uncertain update",
    OPEN_QUESTION: "question asked",
    INFORMATIONAL: "informational",
}

# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
D = r"(?P<date>\d{4}-\d{2}-\d{2})"
TM = r"(?P<time>\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))"
SUBJ = r"(?P<subj>.+?)"

#: Prefixes that mark a message as a re-send of an earlier one. Stripped
#: (recursively -- the corpus stacks them) before frames run.
_REPEAT_PREFIX = re.compile(
    r"^\s*(?:follow[-\s]?up|additional\s+update|reminder|another\s+reminder)\s*:\s*",
    re.IGNORECASE,
)

#: Modality that turns an assertion into a guess. These are the difference
#: between "the report is done" and "the report might be done", and the brief
#: forbids collapsing that difference.
HEDGE = re.compile(
    r"\b(?:might|may|maybe|perhaps|probably|possibly|could\s+be|"
    r"not\s+(?:completely\s+|entirely\s+|fully\s+)?sure|"
    r"cannot\s+confirm|can'?t\s+confirm|unconfirmed|"
    r"i\s+(?:will|'ll)\s+confirm(?:\s+\w+)?\s+later|"
    r"wait\s+for\s+(?:the\s+)?(?:official\s+)?(?:confirmation|update)|"
    r"please\s+wait|to\s+be\s+confirmed|tbc|someone\s+probably)\b",
    re.IGNORECASE,
)

#: An explicit statement that two sources disagree. Distinct from a hedge: the
#: sender is certain that there is a contradiction.
CONFLICT_MARKER = re.compile(
    r"(?:although\s+the\s+earlier\s+message\s+listed\s+another\s+date|"
    r"one\s+message\s+says\s+.+?,?\s*but\s+the\s+latest|"
    r"earlier\s+message\s+listed|"
    r"but\s+the\s+latest\s+instruction|"
    r"conflicting|contradicts?)",
    re.IGNORECASE,
)

URGENCY = re.compile(
    r"\b(?:urgent(?:ly)?|asap|immediately|critical|right\s+away|top\s+priority|"
    r"as\s+soon\s+as\s+possible|high\s+priority|treat\s+this\s+as\s+urgent)\b",
    re.IGNORECASE,
)

DE_ESCALATION = re.compile(
    r"\b(?:no\s+longer\s+(?:be\s+)?urgent|not\s+urgent(?:\s+any\s*more)?|"
    r"lower\s+priority|deprioriti[sz]ed?|can\s+wait)\b",
    re.IGNORECASE,
)

#: Relative day references. Kept as raw text -- resolving them into a stored
#: date is the guessing the brief forbids -- but their *proximity* is a real,
#: stated signal that the priority engine is allowed to read.
RELATIVE_DAY = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|this\s+(?:week|weekend|morning|"
    r"afternoon|evening)|next\s+week|end\s+of\s+(?:the\s+)?(?:day|week)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

#: "Monday or Wednesday" -- two candidate dates and no way to choose. This is
#: unresolvable by construction, which is precisely why it must be reported
#: rather than silently narrowed to one of them.
ALTERNATIVES = re.compile(
    r"\b(?:could\s+be|may\s+be|might\s+be|is\s+either)\s+"
    r"(?P<a>[A-Za-z0-9-]+)\s*,?\s*or\s+(?:it\s+(?:may|might|could)\s+be\s+)?"
    r"(?P<b>[A-Za-z0-9-]+)\b",
    re.IGNORECASE,
)


@dataclass
class ActVerdict:
    """One act, the subject it acts on, and everything it asserts about it."""

    act: str
    confidence: float
    frame: str
    evidence: str
    subject_phrase: Optional[str] = None

    hedged: bool = False
    repetition: bool = False
    repeat_depth: int = 0
    conflict_marker: bool = False
    urgent: bool = False
    de_escalated: bool = False

    #: A newly asserted absolute date (ISO) -- only when literally written out.
    new_date: Optional[str] = None
    #: The raw relative phrase when the date was stated relatively.
    new_date_raw: Optional[str] = None
    new_time: Optional[str] = None
    #: Mutually exclusive candidates the sender offered without choosing.
    alternatives: List[str] = field(default_factory=list)

    @property
    def is_firm(self) -> bool:
        """A firm act may set status; a hedged one may only annotate it."""
        return self.act in FIRM_ACTS and not self.hedged

    def to_dict(self) -> dict:
        return {
            "act": self.act,
            "act_label": HUMAN_ACT.get(self.act, self.act),
            "confidence": self.confidence,
            "frame": self.frame,
            "evidence": self.evidence,
            "subject_phrase": self.subject_phrase,
            "hedged": self.hedged,
            "repetition": self.repetition,
            "conflict_marker": self.conflict_marker,
            "urgent": self.urgent,
            "new_date": self.new_date,
            "new_date_raw": self.new_date_raw,
            "new_time": self.new_time,
            "alternatives": self.alternatives,
        }


def _c(expr: str) -> Pattern[str]:
    return re.compile(expr, re.IGNORECASE)


Frame = Tuple[str, str, Pattern[str]]   # (act, frame name, pattern)

# --------------------------------------------------------------------------
# Frames, in precedence order.
#
# Order is load-bearing. A completion report and a status chase both contain
# the subject and both mention the work; what separates them is the frame, and
# the more specific frame has to be tried first. Terminal acts (completed,
# cancelled) come before schedule changes, which come before status chases,
# which come before the hedged catch-alls.
# --------------------------------------------------------------------------
FRAMES: List[Frame] = [
    # ---- completion ------------------------------------------------------
    (COMPLETION, "completion_report", _c(
        r"^(?:update|confirmed|status)\s*:\s*" + SUBJ +
        r"\s+(?:has\s+been|is|was)\s+(?:completed|finished|done|submitted|"
        r"delivered|signed\s+off|wrapped\s+up|sorted)\b")),
    (COMPLETION, "completion_statement", _c(
        r"\b" + SUBJ + r"\s+(?:has\s+been|have\s+been|is|was)\s+"
        r"(?:successfully\s+)?(?:completed|finished|submitted|delivered|"
        r"closed\s+out|signed\s+off|wrapped\s+up)\b(?!\s*\?)")),

    # ---- work in progress -------------------------------------------------
    # No message in the supplied corpus asserts progress -- the corpus only
    # ever *asks* whether something is in progress, which is a status chase,
    # not a status. These frames exist so that "in progress" is a status the
    # system can actually reach when an assertion does arrive, rather than a
    # value in the schema that nothing can ever produce.
    (PROGRESS_UPDATE, "progress_assertion", _c(
        r"\b" + SUBJ + r"\s+is\s+(?:currently\s+)?(?:in\s+progress|underway|"
        r"being\s+worked\s+on)\b(?!\s*\?)")),
    (PROGRESS_UPDATE, "started_assertion", _c(
        r"\b(?:i|we)\s+(?:have\s+|'ve\s+)?(?:started|begun)\s+"
        r"(?:work\s+on\s+|working\s+on\s+)?" + SUBJ + r"\s*\.?$")),

    # ---- cancellation ----------------------------------------------------
    (CANCELLATION, "cancel_permission", _c(
        r"\byou\s+can\s+cancel\s+" + SUBJ +
        r"\s*[;,.]?\s*it\s+is\s+no\s+longer\s+(?:required|needed)")),
    (CANCELLATION, "cancel_directive", _c(
        r"^(?:please\s+)?cancel\s+" + SUBJ +
        r"\s*[;,.]?\s*it\s+is\s+no\s+longer\s+(?:required|needed)")),
    (CANCELLATION, "cancel_statement", _c(
        r"\b(?:the\s+)?" + SUBJ + r"\s+(?:has\s+been|was|is)\s+cancelled\b")),
    (CANCELLATION, "cancel_bare", _c(
        r"^(?:please\s+)?cancel\s+" + SUBJ + r"\s*[.;]?$")),

    # ---- rescheduling ----------------------------------------------------
    (RESCHEDULE, "moved_to_datetime", _c(
        r"\b(?:the\s+)?" + SUBJ + r"\s+(?:has\s+been|has|was|is)\s+"
        r"(?:been\s+)?(?:moved|rescheduled|shifted|pushed)\s+to\s+"
        + D + r"(?:\s+at\s+" + TM + r")?")),
    (RESCHEDULE, "time_only_change", _c(
        r"\bthe\s+date\s+for\s+" + SUBJ + r"\s+stays\s+the\s+same"
        r".*?\btime\s+is\s+now\s+" + TM)),
    (RESCHEDULE, "moved_relative", _c(
        r"\b(?:the\s+)?" + SUBJ + r"\s+(?:has\s+been|has|was|is)\s+"
        r"(?:been\s+)?(?:moved|rescheduled|shifted)\s+to\s+"
        r"(?P<rel>today|tomorrow|next\s+week|\w+day)\b")),
    (AMBIGUOUS_UPDATE, "may_move_event", _c(
        r"\bwe\s+(?:may|might)\s+move\s+" + SUBJ +
        r"\s*[;,.]?\s*i\s+(?:will|'ll)\s+confirm")),

    # ---- deadline changes -------------------------------------------------
    (DEADLINE_CHANGE, "deadline_now_absolute", _c(
        r"\bthe\s+deadline\s+(?:to|for)\s+" + SUBJ +
        r"\s+(?:is\s+now|has\s+been\s+(?:moved|changed|extended)\s+to|"
        r"is\s+moved\s+to)\s+" + D)),
    (DEADLINE_CHANGE, "deadline_now_relative", _c(
        r"\bthe\s+deadline\s+(?:to|for)\s+" + SUBJ +
        r"\s+(?:is\s+now|has\s+been\s+(?:moved|changed|extended)\s+to)\s+"
        r"(?P<rel>today|tomorrow|next\s+week|this\s+week|\w+day)"
        r"(?:\s+at\s+" + TM + r")?")),
    (DEADLINE_CHANGE, "due_restatement", _c(
        r"\b(?:please\s+note\s+that\s+|the\s+latest\s+instruction\s+says\s+)"
        + SUBJ + r"\s+is\s+due\s+(?:on|by)\s+" + D)),
    (DEADLINE_CHANGE, "due_conflict", _c(
        r"\bone\s+message\s+says\s+\w+.*?\blatest\s+instruction\s+says\s+"
        + SUBJ + r"\s+is\s+due\s+(?:on|by)\s+" + D)),

    # ---- creation ---------------------------------------------------------
    (NEW_TASK, "new_task_marker", _c(
        r"^new\s+task\s*:\s*" + SUBJ + r"\s+by\s+" + D)),
    (NEW_TASK, "new_task_bare", _c(r"^new\s+task\s*:\s*" + SUBJ + r"\s*\.?$")),
    (NEW_EVENT, "new_session_scheduled", _c(
        r"\ba\s+new\s+" + SUBJ + r"\s+(?:session|meeting|review|call|slot)?\s*"
        r"is\s+scheduled\s+for\s+" + D + r"(?:\s+at\s+" + TM + r")?")),

    # ---- status chases ----------------------------------------------------
    (STATUS_QUERY, "share_update", _c(
        r"\bcan\s+you\s+share\s+an\s+update\s+on\s+" + SUBJ + r"\s*\??$")),
    (STATUS_QUERY, "following_up", _c(
        r"\bfollowing\s+up\s+on\s+" + SUBJ + r"\s*[;,]\s*is\s+it\s+in\s+progress")),
    (STATUS_QUERY, "confirm_started", _c(
        r"\bplease\s+confirm\s+whether\s+you\s+(?:have\s+)?started\s+(?:to\s+)?"
        + SUBJ + r"\s*\.?$")),
    (STATUS_QUERY, "any_progress", _c(
        r"\bany\s+progress\s+on\s+(?:the\s+item\s+concerning\s+)?" + SUBJ + r"\s*\??$")),
    (STATUS_QUERY, "latest_status", _c(
        r"\b(?:please\s+)?check\s+the\s+latest\s+status\s+of\s+" + SUBJ + r"\s*\.?$")),
    (STATUS_QUERY, "still_needs_attention", _c(
        r"\bthe\s+work\s+we\s+discussed\s+about\s+" + SUBJ +
        r"\s+still\s+needs\s+attention")),
    (STATUS_QUERY, "handled_yet", _c(
        r"\bhas\s+(?:the\s+)?(?:material\s+for\s+(?:our\s+)?(?:earlier\s+)?)?"
        + SUBJ + r"\s+(?:item\s+)?been\s+handled(?:\s+yet)?\s*\??")),
    (STATUS_QUERY, "referring_to_request", _c(
        r"\bi\s+am\s+referring\s+to\s+our\s+earlier\s+request\s+about\s+"
        + SUBJ + r"\s*\.?$")),
    (STATUS_QUERY, "any_update", _c(
        r"\bany\s+update\s+on\s+" + SUBJ + r"\s*\??$")),
    (STATUS_QUERY, "explicit_status_request", _c(
        r"\bthis\s+is\s+(?:another|a)\s+status\s+request\s+about\s+" + SUBJ +
        r"\s*(?:,\s*not\s+a\s+new\s+(?:task|request))?\s*\.?$")),

    # ---- hedged updates ---------------------------------------------------
    # Reached only when no firm frame matched, so these never shadow a real
    # completion. Each one asserts something *about* a subject without
    # asserting it firmly.
    (AMBIGUOUS_UPDATE, "hedged_completion", _c(
        r"\b" + SUBJ + r"\s+(?:might|may|could)\s+(?:already\s+)?be\s+"
        r"(?:done|finished|completed|handled|signed\s+off|wrapped\s+up|"
        r"sorted)\b")),
    (AMBIGUOUS_UPDATE, "hedged_third_party", _c(
        r"\b(?P<who>[A-Z][a-z]+)\s+said\s+someone\s+probably\s+handled\s+"
        + SUBJ + r"\s*\.?$")),
    # Names no subject -- "the deadline" is the only noun phrase present, and
    # inventing one from the surrounding words is exactly the guess the brief
    # forbids. The candidate dates are captured as `alternatives` instead.
    (AMBIGUOUS_UPDATE, "hedged_deadline", _c(
        r"\bthe\s+deadline\s+(?:could|may|might)\s+be\b(?P<subj>)")),
    (AMBIGUOUS_UPDATE, "hedged_deprioritisation", _c(
        r"\bthis\s+(?:may|might)\s+no\s+longer\s+be\s+urgent\b(?P<subj>)")),
]

#: A bare question that names no tracked subject and that nothing in the corpus
#: answers. Detected last, and deliberately: it is the shape that must trigger
#: the assistant's "insufficient evidence" response rather than a guess.
_QUESTION = re.compile(r"\?\s*$")

_TITLE_JUNK = re.compile(
    r"^(?:the|a|an|our|your|my|to|that|this)\s+", re.IGNORECASE)


def strip_repeats(text: str) -> Tuple[str, int]:
    """Remove stacked re-send prefixes; return the core and how many were found."""
    depth = 0
    cur = text.strip()
    while True:
        m = _REPEAT_PREFIX.match(cur)
        if not m:
            break
        cur = cur[m.end():].strip()
        depth += 1
        if depth > 4:                      # pathological input guard
            break
    return cur, depth


def _clean_subject(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = re.sub(r"\s+", " ", raw).strip(" \t.,;:!?\"'")
    s = _TITLE_JUNK.sub("", s)
    # Trailing scaffolding that frames can drag along with a greedy subject.
    s = re.sub(
        r"\s+(?:item|work|material|request)$", "", s,
        flags=re.IGNORECASE,
    )
    return s.strip(" .,;:!?") or None


def _norm_time(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    t = raw.strip().lower()
    m = re.match(r"^(\d{1,2}):(\d{2})\s*(am|pm)?$", t)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if m.group(3) == "pm" and h < 12:
            h += 12
        if m.group(3) == "am" and h == 12:
            h = 0
        return f"{h:02d}:{mi:02d}" if 0 <= h <= 23 and 0 <= mi <= 59 else None
    m = re.match(r"^(\d{1,2})\s*(am|pm)$", t)
    if m:
        h = int(m.group(1)) % 12
        if m.group(2) == "pm":
            h += 12
        return f"{h:02d}:00"
    return None


def from_l1_extraction(verdict: ActVerdict, item_type: Optional[str]) -> ActVerdict:
    """Promote an L1 request into the act that raised its subject.

    This is the hinge between the two systems. An L1 message such as

        "Can you review the privacy checklist before 2026-09-09?"

    performs no *L2* act -- none of the frames above describe it, because L2's
    frames are about following up on work that already exists. But L1 already
    read that message, decided it was action_required, and extracted a task
    from it. That extraction *is* the assertion "this task now exists", and
    treating it as one is what lets an L2 follow-up attach to the moment the
    task was raised rather than to the first follow-up about it.

    Without this the timeline starts in the middle: a subject would appear to
    begin with somebody chasing it, and its original deadline would have no
    source message. It also stops L1 requests from being miscounted as chases
    -- "Can you review X before Friday?" is a request, not a reminder, and only
    the second one is evidence of pressure.

    Applied only when L1 actually produced an item and no L2 frame claimed the
    message, so a genuine L2 act is never overwritten.

    The subject phrase is dropped rather than carried over. The only phrase
    available on this path is whatever the unmatched-question fallback
    captured, which for "Can you review the privacy checklist before
    2026-09-09?" is the entire sentence. Clearing it makes `subject_for` fall
    through to the title L1 already extracted -- "Review the privacy
    checklist" -- which is both the better identity and the better label.
    """
    if item_type not in ("task", "event"):
        return verdict
    if verdict.act not in (INFORMATIONAL, OPEN_QUESTION):
        return verdict
    return ActVerdict(
        act=NEW_TASK if item_type == "task" else NEW_EVENT,
        confidence=0.85,
        frame="l1_extraction",
        evidence="L1 classified this message as actionable and extracted an item",
        subject_phrase=None,
        hedged=verdict.hedged,
        repetition=verdict.repetition,
        repeat_depth=verdict.repeat_depth,
        conflict_marker=verdict.conflict_marker,
        urgent=verdict.urgent,
        de_escalated=verdict.de_escalated,
        new_date=verdict.new_date,
        new_date_raw=verdict.new_date_raw,
        new_time=verdict.new_time,
        alternatives=verdict.alternatives,
    )


def detect(message: str) -> ActVerdict:
    """Classify the act performed by one already-masked message.

    Returns ``INFORMATIONAL`` when no frame fires -- which is the honest answer
    for a message that does not act on any subject at all ("The office
    cafeteria menu has been updated"), and is what keeps the general-update
    traffic out of the task ledger.
    """
    body = body_of(message)
    core, repeat_depth = strip_repeats(body)

    hedged = bool(HEDGE.search(core))
    conflict = bool(CONFLICT_MARKER.search(core))
    de_esc = bool(DE_ESCALATION.search(core))
    # "no longer urgent" contains "urgent". A de-escalation is the opposite of
    # an urgency signal, so it wins outright rather than both firing.
    urgent = bool(URGENCY.search(core)) and not de_esc

    for act, name, pattern in FRAMES:
        m = pattern.search(core)
        if not m:
            continue
        groups = m.groupdict()
        subject = _clean_subject(groups.get("subj"))

        # A firm act whose sentence is hedged is downgraded rather than
        # trusted. "The X may have been moved" is not a reschedule.
        effective_act = act
        if hedged and act in FIRM_ACTS:
            effective_act = AMBIGUOUS_UPDATE

        rel = groups.get("rel")
        new_date = groups.get("date")
        new_date_raw = rel
        if not new_date and not rel:
            found = RELATIVE_DAY.search(core)
            if found and act in (DEADLINE_CHANGE, RESCHEDULE):
                new_date_raw = found.group(0)

        alts: List[str] = []
        am = ALTERNATIVES.search(core)
        if am:
            alts = [am.group("a"), am.group("b")]

        base = 0.90 if act in FIRM_ACTS else 0.86
        if hedged:
            base -= 0.20
        if repeat_depth:
            base -= 0.02 * repeat_depth
        confidence = round(max(0.35, min(base, 0.95)), 4)

        return ActVerdict(
            act=effective_act,
            confidence=confidence,
            frame=name,
            evidence=m.group(0).strip()[:160],
            subject_phrase=subject,
            hedged=hedged,
            repetition=repeat_depth > 0,
            repeat_depth=repeat_depth,
            conflict_marker=conflict,
            urgent=urgent,
            de_escalated=de_esc,
            new_date=new_date,
            new_date_raw=new_date_raw,
            new_time=_norm_time(groups.get("time")),
            alternatives=alts,
        )

    # No frame fired. Two residual shapes are still worth naming.
    if hedged:
        alts = []
        am = ALTERNATIVES.search(core)
        if am:
            alts = [am.group("a"), am.group("b")]
        rel = RELATIVE_DAY.search(core)
        return ActVerdict(
            act=AMBIGUOUS_UPDATE,
            confidence=0.55,
            frame="hedged_residual",
            evidence=(HEDGE.search(core).group(0) if HEDGE.search(core) else ""),
            subject_phrase=None,
            hedged=True,
            repetition=repeat_depth > 0,
            repeat_depth=repeat_depth,
            conflict_marker=conflict,
            urgent=urgent,
            de_escalated=de_esc,
            new_date_raw=rel.group(0) if rel else None,
            alternatives=alts,
        )

    if _QUESTION.search(core):
        return ActVerdict(
            act=OPEN_QUESTION,
            confidence=0.70,
            frame="unmatched_question",
            evidence=core[:160],
            subject_phrase=_clean_subject(core.rstrip("? ")),
            repetition=repeat_depth > 0,
            repeat_depth=repeat_depth,
            urgent=urgent,
        )

    return ActVerdict(
        act=INFORMATIONAL,
        confidence=0.60,
        frame="no_frame",
        evidence="",
        repetition=repeat_depth > 0,
        repeat_depth=repeat_depth,
        urgent=urgent,
        de_escalated=de_esc,
    )
