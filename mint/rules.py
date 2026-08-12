"""The interpretable rule layer.

Every rule is a named linguistic *frame* -- a grammatical shape that carries
intent -- rather than a bag of keywords. "meeting" appearing anywhere does not
make a message a meeting; "the X is scheduled for Y" does. Frames are what let
this layer generalise past the exact sentences in the corpus, and they are what
make each decision explainable: the rule reports which frame fired and what
text triggered it.

The rule layer has two jobs:
  1. At inference time it is one of two voices in the hybrid classifier.
  2. At training time it is the *weak supervision* source -- the corpus ships
     with no labels, so high-precision frames generate the training labels that
     the statistical model then learns to generalise from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Pattern, Tuple

from . import taxonomy as T
from .normalize import canonical, has_absolute_date, has_clock_time


@dataclass
class Signal:
    """One frame that fired, and the span of text that fired it."""

    name: str
    evidence: str


@dataclass
class RuleVerdict:
    category: Optional[str]
    confidence: float
    signals: List[Signal] = field(default_factory=list)
    #: True when the frame that fired is hedged or incomplete, e.g. an event
    #: with no date. Used downstream to damp confidence and flag for review.
    hedged: bool = False

    @property
    def evidence_phrase(self) -> str:
        if not self.signals:
            return ""
        return "; ".join(f'"{s.evidence}"' for s in self.signals[:2])


def _c(expr: str) -> Pattern[str]:
    return re.compile(expr, re.IGNORECASE)


Frame = Tuple[str, Pattern[str]]

# --------------------------------------------------------------------------
# Promotional
# --------------------------------------------------------------------------
PROMO_FRAMES: List[Frame] = [
    ("coupon_code", _c(r"\buse\s+code\s+[A-Z0-9]{3,}\b")),
    ("discount_offer", _c(r"\b(\d+%\s*off|flat\s+\d+%|discount|flash\s+sale|"
                          r"limited[-\s]time\s+offer|special\s+(?:festival\s+)?offer)\b")),
    ("commercial_incentive", _c(r"\b(cashback|reward\s+points|free\s+delivery|"
                                r"buy\s+one\s+and\s+get\s+one|coupon|voucher)\b")),
    ("upsell", _c(r"\b(upgrade\s+your\s+\w+|join\s+our\s+\w+\s+plan|"
                  r"exclusive\s+benefits|you\s+may\s+like\s+our|"
                  r"our\s+new\s+\w+\s+plan|subscribe\s+now)\b")),
    ("sale_event", _c(r"\b(sale\s+on\b|offer\s+on\b|deal\s+on\b|"
                      r"\bsale\s+starts\b|\bsale\s+ends\b)")),
]

# --------------------------------------------------------------------------
# Meeting / event
# --------------------------------------------------------------------------
MEETING_FRAMES: List[Frame] = [
    ("calendar_marker", _c(r"^(calendar\s+(?:update|invite|entry)|invite|reminder)\s*:")),
    ("scheduled_copula", _c(r"\b(?:is|are|was|will\s+be)\s+scheduled\s+(?:for|on|at)\b")),
    ("occurrence_verb", _c(r"\b(?:happens|takes\s+place|is\s+happening|"
                           r"will\s+be\s+held|is\s+being\s+held)\s+(?:on|at|in)\b")),
    ("invitation", _c(r"\b(?:please\s+)?(?:join|attend|come\s+to|be\s+present\s+at)\s+"
                      r"(?:the|our|us\s+for)\b")),
    ("availability_probe", _c(r"\bare\s+you\s+(?:available|free)\s+for\b")),
    ("meet_proposal", _c(r"\b(?:let(?:'s|\s+us)\s+meet|shall\s+we\s+meet|"
                         r"can\s+we\s+meet|meeting\s+at)\b")),
]

#: A hedged event: an event noun with a modal and no firm date.
HEDGED_EVENT = _c(
    r"\bthe\s+(?:meeting|review|discussion|session|call|sync|demo|interview|"
    r"catch[-\s]?up|stand[-\s]?up|appointment|briefing)\s+"
    r"(?:could|might|may|should)\s+be\b"
)

EVENT_NOUNS = _c(
    r"\b(meeting|meet|appointment|session|workshop|seminar|demo|review|"
    r"stand[-\s]?up|orientation|briefing|interview|dinner|catch[-\s]?up|sync|"
    r"planning|discussion|ceremony|event|call|class|lecture|conference)\b"
)

# --------------------------------------------------------------------------
# Action required
# --------------------------------------------------------------------------
ACTION_FRAMES: List[Frame] = [
    ("polite_directive", _c(r"\b(?:please|kindly)\s+(?!join\b|note\b)[a-z]+\b")),
    ("request_question", _c(r"\b(?:can|could|would|will)\s+you\s+[a-z]+\b")),
    ("explicit_request", _c(r"\bi\s+need\s+you\s+to\s+[a-z]+\b")),
    ("reminder_directive", _c(r"\b(?:don'?t\s+forget\s+to|remember\s+to|"
                              r"make\s+sure\s+(?:you\s+|to\s+)?|be\s+sure\s+to|"
                              r"ensure\s+(?:you\s+)?)[a-z]+\b")),
    ("conditional_request", _c(r"\bif\s+possible,?\s+[a-z]+\b")),
    ("deadline_marker", _c(r"\b(?:deadline\s+is|is\s+due\s+(?:on|by)|due\s+date|"
                           r"due\s+(?:on|by)\b)")),
    ("date_bounded", _c(r"\b(?:by|before|no\s+later\s+than)\s+\d{4}-\d{2}-\d{2}\b")),
    ("bare_imperative", _c(r"^(?:submit|send|complete|prepare|review|upload|call|"
                           r"email|book|renew|verify|share|update|finish|reply|"
                           r"confirm|pay|check|fix|draft|arrange|collect|"
                           r"back\s+up)\s+(?:the|your|my|a|an|it)\b")),
]

#: A hedged obligation: something is expected of the recipient, but softly.
HEDGED_ACTION = _c(
    r"\b(?:may\s+be\s+needed|might\s+be\s+needed|could\s+you\s+send|"
    r"send\s+it\s+soon|asked\s+whether|wants\s+to\s+know\s+(?:if|whether)|"
    r"is\s+waiting\s+for)\b"
)

# --------------------------------------------------------------------------
# Personal information
# --------------------------------------------------------------------------
PERSONAL_FRAMES: List[Frame] = [
    ("profile_marker", _c(r"\b(?:for\s+my\s+profile|personal\s+note|about\s+me)\b\s*[:,]?")),
    ("remember_preference", _c(r"\b(?:remember|note)\s+that\s+i\b")),
    ("disclosure_marker", _c(r"\bjust\s+so\s+you\s+know,?\s+(?:i|my)\b")),
    ("first_person_trait", _c(r"\bi\s+(?:am|'m)\s+(?:a\s+)?(?!going|working\s+on)[a-z]+\b")),
    ("first_person_preference", _c(r"\bi\s+(?:might\s+)?(?:prefer|like|usually|always|"
                                   r"never|tend\s+to|use|drink|eat|study|work\s+best)\b")),
    ("first_person_situation", _c(r"\bi\s+live\s+(?:near|in|at|around)\b")),
    ("my_attribute", _c(r"\bmy\s+(?:favourite|favorite|t[-\s]?shirt\s+size|"
                        r"emergency\s+contact|dietary|preference|nickname|"
                        r"working\s+hours)\b")),
]


# --------------------------------------------------------------------------
# General information
# --------------------------------------------------------------------------
# General information is the residual category, but giving it only a residual
# definition would leave the statistical model with no positive examples to
# learn from. So it gets a real frame of its own: a third-person declarative
# statement of fact -- no question, no first-person self-disclosure, no
# directive. Everything the earlier frames already claimed has been removed by
# the time this is reached, so this is a genuine test, not a catch-all.
FIRST_PERSON = _c(r"\b(?:i|i'm|i'll|my|me|mine|myself)\b")

DECLARATIVE_SUBJECT = _c(
    r"^(?:the|a|an|our|this|that|there|tomorrow|today|tonight|yesterday|"
    r"all|new|most|some|every|[a-z]+ing)\s+\S+"
)

STATE_PREDICATE = _c(
    r"\b(?:is|are|was|were|has|have|had|will|can|closes|opens|starts|ends|"
    r"leaves|arrives|changed|moved|updated|reorganized|extended|says|"
    r"remains|continues|becomes)\b"
)


def _first_match(frames: List[Frame], text: str) -> List[Signal]:
    out: List[Signal] = []
    for name, pat in frames:
        m = pat.search(text)
        if m:
            out.append(Signal(name=name, evidence=m.group(0).strip()))
    return out


def _confidence(base: float, n_signals: int, hedged: bool) -> float:
    """Base confidence, raised a little per corroborating frame, damped if hedged.

    Deliberately capped below 1.0 -- a rule layer that claims certainty is
    lying, and the calibration step downstream needs headroom to work with.
    """
    conf = base + 0.05 * max(0, n_signals - 1)
    if hedged:
        conf -= 0.22
    return round(min(conf, 0.96), 4)


def classify(message: str, sender: str = "") -> RuleVerdict:
    """Apply the frames in precedence order and return the first firm verdict.

    `message` must already be masked -- the rule layer never needs to see a raw
    secret, and the sensitive category is decided by `sensitive.scan`, not here.
    """
    text = canonical(message)
    sender_l = (sender or "").strip().lower()

    # 1. Promotional. Checked first because marketing copy borrows the grammar
    #    of every other category ("Join our plan", "Offer ends 2026-09-10").
    promo = _first_match(PROMO_FRAMES, text)
    if promo:
        base = 0.90 if len(promo) > 1 else 0.86
        if sender_l in {"promotions", "marketing", "offers", "no-reply"}:
            promo.append(Signal("promotional_sender", f"sender={sender}"))
            base += 0.04
        return RuleVerdict(T.PROMOTIONAL, _confidence(base, len(promo), False), promo)

    # 2. Meeting or event. Requires a scheduling frame, not merely an event noun,
    #    so "The webinar recording is now available" stays general information.
    meet = _first_match(MEETING_FRAMES, text)
    if meet:
        firm = has_absolute_date(text) or has_clock_time(text)
        hedged = not firm
        base = 0.90 if firm else 0.80
        return RuleVerdict(T.MEETING_EVENT, _confidence(base, len(meet), hedged),
                           meet, hedged=hedged)

    m = HEDGED_EVENT.search(text)
    if m:
        return RuleVerdict(
            T.MEETING_EVENT,
            _confidence(0.78, 1, True),
            [Signal("hedged_event", m.group(0).strip())],
            hedged=True,
        )

    # 3. Action required.
    act = _first_match(ACTION_FRAMES, text)
    if act:
        firm = any(s.name in {"deadline_marker", "date_bounded"} for s in act)
        base = 0.91 if firm else 0.84
        return RuleVerdict(T.ACTION_REQUIRED, _confidence(base, len(act), False), act)

    m = HEDGED_ACTION.search(text)
    if m:
        return RuleVerdict(
            T.ACTION_REQUIRED,
            _confidence(0.72, 1, True),
            [Signal("hedged_action", m.group(0).strip())],
            hedged=True,
        )

    # 4. Personal information.
    pers = _first_match(PERSONAL_FRAMES, text)
    if pers:
        base = 0.90 if len(pers) > 1 else 0.84
        return RuleVerdict(T.PERSONAL_INFORMATION, _confidence(base, len(pers), False), pers)

    # 5. General information: a third-person declarative statement of fact.
    if not text.endswith("?") and not FIRST_PERSON.search(text):
        subj = DECLARATIVE_SUBJECT.search(text)
        pred = STATE_PREDICATE.search(text)
        if subj and pred:
            return RuleVerdict(
                T.GENERAL_INFORMATION,
                _confidence(0.84, 2, False),
                [
                    Signal("third_person_declarative", subj.group(0).strip()),
                    Signal("state_predicate", pred.group(0).strip()),
                ],
            )

    # 6. Nothing fired. `None` means "the rules abstain" -- distinct from
    #    "the rules are confident this is general information". The hybrid
    #    classifier treats abstention as a cue to trust the model instead.
    return RuleVerdict(None, 0.0, [])


def weak_label(message: str, sender: str = "") -> Optional[str]:
    """Label used for weak supervision. Only unhedged, confident frames count."""
    v = classify(message, sender)
    if v.category is None or v.hedged or v.confidence < 0.84:
        return None
    return v.category
