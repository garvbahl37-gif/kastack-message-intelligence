"""Text normalisation shared by the rule layer, the model and the extractor.

Messages in this corpus are wrapped in conversational filler ("FYI:", "Just
checking-", "Can you help?"). That filler is noise for classification -- and
worse, "Can you help?" looks exactly like a request even when the sentence it
introduces is a plain statement of fact. Stripping it first removes a whole
class of false positives.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Openers that carry no topical meaning. Matched only at the very start of the
# message so that a genuine "Can you help me move the server?" is untouched --
# that form has a verb phrase after it and never matches the bare variants here.
_OPENERS = [
    r"For today:",
    r"FYI:",
    r"Important:",
    r"One more thing:",
    r"Just checking[—–-]",
    r"Quick update:",
    r"Please note:",
    r"Can you help\?",
    r"Hi,",
    r"Hello,",
    r"Hey,",
    r"Heads up:",
    r"Note:",
    r"Reminder to self:",
]

_OPENER_RE = re.compile(r"^\s*(?:" + "|".join(_OPENERS) + r")\s*", re.IGNORECASE)

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9']+")

DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
TIME_RE = re.compile(
    r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b|\b(\d{1,2})\s*(am|pm)\b", re.IGNORECASE
)


def strip_opener(text: str) -> Tuple[str, str]:
    """Split a message into (opener, body). Opener is "" when absent."""
    m = _OPENER_RE.match(text)
    if not m:
        return "", text.strip()
    return m.group(0).strip(), text[m.end():].strip()


def body_of(text: str) -> str:
    """The message with its conversational opener removed."""
    return strip_opener(text)[1]


def canonical(text: str) -> str:
    """Lowercased, whitespace-collapsed body -- the surface the rules match on."""
    return _WS_RE.sub(" ", body_of(text).lower()).strip()


def tokenize(text: str) -> List[str]:
    """Word tokens for the TF-IDF vectoriser.

    Dates, times and bare numbers are folded into placeholder tokens. Without
    this the model would learn that "2026-09-09" means "deadline", which is
    memorisation of this corpus rather than a transferable signal; with it, the
    model learns that *the presence of a date* is what matters.
    """
    t = text.lower()
    t = DATE_RE.sub(" datetoken ", t)
    t = TIME_RE.sub(" timetoken ", t)
    t = re.sub(r"\*{3,}", " maskedtoken ", t)
    t = re.sub(r"\b\d+\b", " numtoken ", t)
    return _TOKEN_RE.findall(t)


def has_absolute_date(text: str) -> bool:
    return bool(DATE_RE.search(text))


def has_clock_time(text: str) -> bool:
    return bool(TIME_RE.search(text))
