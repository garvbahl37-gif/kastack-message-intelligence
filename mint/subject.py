"""Subject identity: what a message is *about*.

L1 asked "what kind of message is this?". L2 asks a harder question: "is this
message about the same thing as that one?". Everything in L2 -- grouping,
status tracking, deadline updates, priority re-evaluation, semantic search --
depends on being able to answer it.

The unit of identity here is a **subject signature**: the content words that
survive after the conversational scaffolding is stripped away.

    "Can you share an update on review the privacy checklist?"
    "Please confirm whether you started to review the privacy checklist."
    "The deadline to review the privacy checklist is now 2026-09-30."

All three reduce to ``{review, privacy, checklist}``. So does the L1 task whose
extracted title was "Review the privacy checklist" -- which is how an L2
follow-up finds the L1 task it follows up on.

Why not string equality on the title?
-------------------------------------
Because real follow-ups are lossy. "Any progress on the item concerning the
assignment?" never repeats the verb; "Has the material for our earlier
model-results review been handled?" reorders the words and hyphenates them.
Both must still land on the right subject, and a title comparison cannot do it.

Why not plain word overlap?
---------------------------
Because the brief is explicit that messages must not be grouped merely for
sharing one common word -- and this corpus is full of subjects that share
exactly one. "Review the privacy checklist" and "Review the model results"
overlap on *review*, and they are unrelated. So matching is governed by two
conditions that have to hold together:

1. **Cosine** over IDF-weighted tokens -- the usual "are these similar?" test.
2. **Containment of informative mass** -- of the weight carried by the *shorter*
   signature, most of it must be matched by the longer one.

Condition 2 is what makes "the assignment" match "upload the assignment"
(the short signature is entirely covered) while stopping "review the privacy
checklist" from matching "review the model results" (only the cheapest token is
shared, so the covered mass is small). A shared word only counts for as much as
it is worth, and *review* is not worth much in a corpus where a dozen subjects
contain it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

# Function words plus the conversational furniture this corpus wraps subjects
# in. "the item concerning X", "the work we discussed about X", "the material
# for our earlier X" all decorate a subject without changing it, so the
# decoration must not become part of its identity.
STOPWORDS = frozenset("""
a an the this that these those it its is are was were be been being am
and or but if then than so as of to for from in on at by with about into
before after until during while sometime someone somebody something
over under again further once here there when where why how all any both
each few more most other some such no nor not only own same too very can
will just should now i me my we our you your he she they them their who whom
please kindly do does did done doing have has had having would could shall may
might must let us also still yet ever never
item items work material thing things stuff matter subject topic regarding
concerning earlier previous latest new one another request required needed
progress update updates status
""".split())

# Generic heads that describe the *shape* of an activity rather than its
# identity. "session", "meeting" and "task" are how the corpus refers to a
# subject; they are never what distinguishes one subject from another, and
# leaving them in makes every meeting look like every other meeting.
GENERIC_HEADS = frozenset("""
session meet task deadline schedule instruction message reminder
""".split())

_WORD = re.compile(r"[a-z][a-z0-9'-]*")

#: Irregulars first, then suffix stripping. This is deliberately a stemmer of
#: about twelve rules rather than a real lemmatiser: the vocabulary here is a
#: few hundred office words, and a dependency-free approximation that handles
#: plurals and gerunds covers essentially all of it. `results`/`result` and
#: `scheduling`/`schedule` are the cases that actually occur.
_IRREGULAR = {
    "meetings": "meeting", "notes": "note", "minutes": "minute",
    "files": "file", "details": "detail", "results": "result",
    "labels": "label", "cases": "case", "receipts": "receipt",
    "slots": "slot", "checklists": "checklist", "presentations": "presentation",
    "documents": "document", "exercises": "exercise", "reports": "report",
    "videos": "video", "forms": "form", "books": "book", "bills": "bill",
    "emails": "email", "models": "model", "trackers": "tracker",
    "assignments": "assignment", "queries": "query", "decisions": "decision",
    "diagrams": "diagram", "charts": "chart", "embeddings": "embedding",
    # The -ing rule below would fold this to "meete", which then looks like a
    # content word rather than the generic head it is.
    "meeting": "meet", "meetings": "meet",
}


def _stem(word: str) -> str:
    """Crude but stable morphological folding."""
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) > 4:
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith("sses") or word.endswith("shes") or word.endswith("ches"):
            return word[:-2]
        if word.endswith("s") and not word.endswith("ss") and not word.endswith("us"):
            return word[:-1]
        if word.endswith("ing") and len(word) > 6:
            base = word[:-3]
            return base + "e" if base.endswith(("l", "t", "s", "v", "z")) else base
    return word


def content_tokens(text: str) -> List[str]:
    """Stemmed content words, in order, with duplicates removed.

    Hyphenated compounds are split as well as kept whole, so "model-results"
    contributes `model` and `result` and therefore matches "the model results".
    """
    out: List[str] = []
    seen = set()
    lowered = (text or "").lower().replace("—", " ").replace("–", " ")
    for raw in _WORD.findall(lowered.replace("-", " - ")):
        if raw == "-":
            continue
        w = _stem(raw)
        if w in STOPWORDS or len(w) < 2:
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


@dataclass(frozen=True)
class SubjectSignature:
    """The identity of a subject: its tokens, plus a phrase to display."""

    tokens: FrozenSet[str]
    display: str
    #: Tokens that are not generic activity heads. Two subjects that share only
    #: generic heads ("session", "meeting") are not the same subject.
    specific: FrozenSet[str] = field(default_factory=frozenset)

    def __bool__(self) -> bool:
        return bool(self.specific)


def signature(phrase: str) -> SubjectSignature:
    """Build a signature from a subject phrase or an extracted item title."""
    toks = content_tokens(phrase)
    tokset = frozenset(toks)
    specific = frozenset(t for t in tokset if t not in GENERIC_HEADS)
    display = " ".join(w for w in re.split(r"\s+", (phrase or "").strip()) if w)
    display = re.sub(r"^(?:the|a|an|our|your|my)\s+", "", display, flags=re.IGNORECASE)
    display = display.strip(" .,;:!?-")
    return SubjectSignature(tokens=tokset, display=display, specific=specific)


class SubjectSpace:
    """IDF statistics over the subjects seen in a corpus.

    A token's weight is how *rare* it is across distinct subjects, not across
    messages. That distinction matters: `review` appears in a handful of
    subjects but in hundreds of messages, and it is the subject count that says
    how much identifying power it carries.

    The statistics are a property of the corpus, not of any single message's
    position in it, so computing them in a first pass does not let later
    messages influence earlier decisions. Status and priority resolution stay
    strictly chronological -- see `mint/groups.py`.
    """

    def __init__(self, subjects: Iterable[SubjectSignature]) -> None:
        subs = [s for s in subjects if s]
        # Deduplicate identical signatures so a subject repeated 40 times does
        # not make its own tokens look common.
        distinct: Dict[FrozenSet[str], SubjectSignature] = {}
        for s in subs:
            distinct.setdefault(s.tokens, s)
        self.n_subjects = max(len(distinct), 1)
        df: Dict[str, int] = {}
        for s in distinct.values():
            for t in s.tokens:
                df[t] = df.get(t, 0) + 1
        self.df = df
        self._idf: Dict[str, float] = {
            t: math.log((self.n_subjects + 1) / (c + 1)) + 1.0 for t, c in df.items()
        }
        #: A token never seen before is maximally informative.
        self._default_idf = math.log(self.n_subjects + 1) + 1.0
        #: How many distinct subjects a token may name and still be allowed to
        #: carry a one-word match on its own. Expressed as a share of the
        #: subject vocabulary so it scales with the corpus instead of being a
        #: constant tuned to this one.
        self.distinctive_df = max(1, round(DISTINCTIVE_DF_SHARE * self.n_subjects))

    def idf(self, token: str) -> float:
        return self._idf.get(token, self._default_idf)

    def mass(self, sig: SubjectSignature) -> float:
        return sum(self.idf(t) for t in sig.tokens)

    def match(
        self, a: SubjectSignature, b: SubjectSignature
    ) -> Tuple[float, List[str]]:
        """Similarity in [0, 1] plus the tokens that carried it.

        Zero is returned -- not a small number -- when the two signatures fail
        the containment test, because "these are different subjects" is a
        decision, not a weak similarity.
        """
        if not a or not b:
            return 0.0, []
        shared = a.tokens & b.tokens
        if not shared:
            return 0.0, []
        # A match resting entirely on generic activity heads is not a match.
        if not (shared & (a.specific | b.specific)):
            return 0.0, []

        shared_mass = sum(self.idf(t) for t in shared)
        ma, mb = self.mass(a), self.mass(b)
        cosine = shared_mass / math.sqrt(ma * mb) if ma and mb else 0.0
        containment = shared_mass / min(ma, mb) if min(ma, mb) else 0.0

        # A reference qualifies in one of exactly two ways:
        #
        #   (a) it is a *sub-phrase* of the subject -- "the assignment" for
        #       "upload the assignment" -- in which case nearly all of the
        #       shorter signature's informative mass is covered; or
        #   (b) it is a *restatement* of the subject in near-identical
        #       vocabulary -- "model-results review" for "review the model
        #       results" -- in which case the cosine is high.
        #
        # Anything in between is two different subjects that happen to share a
        # word or two: "prepare the demo video" and "prepare the offline
        # inference demo" cover two thirds of each other and are unrelated.
        # Admitting that middle band is what merges them, so it is excluded.
        sub_phrase = containment >= CONTAINMENT_ACCEPT and cosine >= COSINE_FLOOR
        restatement = cosine >= COSINE_ACCEPT
        if not (sub_phrase or restatement):
            return 0.0, []

        # A match resting on a *single* shared token is only allowed when that
        # token is rare enough to identify a subject on its own. "assignment"
        # names one subject and may carry a match by itself; "review" names a
        # dozen and may not. Containment alone does not catch this, because a
        # one-word probe is trivially contained in anything that shares its
        # word -- which is precisely the "one common word" grouping the brief
        # rules out.
        if len(shared) == 1:
            token = next(iter(shared))
            if self.df.get(token, 0) > self.distinctive_df:
                return 0.0, []

        # Report the containment-weighted score: it is the one that decides,
        # and it is the one a reader can interpret ("83% of the shorter
        # subject's informative content is shared").
        score = round(0.5 * cosine + 0.5 * containment, 4)
        evidence = sorted(shared, key=lambda t: -self.idf(t))
        return score, evidence


#: Floor for the sub-phrase route: below this the two signatures are simply
#: dissimilar, however well one is contained in the other.
COSINE_FLOOR = 0.34
#: How much of the shorter signature's informative mass the longer one must
#: cover for the sub-phrase route to apply.
CONTAINMENT_ACCEPT = 0.92
#: How similar two signatures must be to match without containment.
COSINE_ACCEPT = 0.62

#: Share of the subject vocabulary a token may appear in and still identify a
#: subject on its own. At ~45 distinct subjects this admits tokens naming up to
#: four of them, which covers "assignment" and "report" while excluding
#: "review", "send" and "call".
DISTINCTIVE_DF_SHARE = 0.08


def best_match(
    space: SubjectSpace,
    probe: SubjectSignature,
    candidates: Sequence[Tuple[str, SubjectSignature]],
) -> Optional[Tuple[str, float, List[str]]]:
    """The best-scoring candidate for `probe`, or None if nothing qualifies."""
    best: Optional[Tuple[str, float, List[str]]] = None
    for key, sig in candidates:
        score, evidence = space.match(probe, sig)
        if score <= 0.0:
            continue
        if best is None or score > best[1]:
            best = (key, score, evidence)
    return best
