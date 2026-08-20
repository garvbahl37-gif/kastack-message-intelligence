"""Privacy-aware routing: what each message is allowed to be used for.

L1 answered "is there something sensitive here, and what should be done about
it?" and produced a recommended action per message. L2 has to *act* on that
answer, because L2 does things L1 never did: it builds a search index, it
retrieves evidence, and it composes answers out of messages the user did not
name. Each of those is a new way for a value to travel, so each needs a gate.

This module turns L1's recommendation into an enforceable decision. Every
message gets a **route**:

``local_only``
    Process it normally. Everything happens on this machine, which is where
    everything already happened.

``confirm_required``
    Usable, but not without a human saying so. Private addresses and phone
    numbers are real personal data and often exactly what the user meant to
    search for -- refusing outright would be theatre. Asking first is the
    honest middle.

``blocked``
    Credentials and financial identifiers. The masked stub is kept so the
    message can be counted, audited and reported; the content is never
    indexed, never retrieved, and never quoted in an answer. A one-time
    password has no analytical value and unlimited downside.

Two properties are worth stating plainly, because they are what make this more
than a label:

* **The index is filtered, not the results.** Blocked messages are never added
  to the retrieval index in the first place, so no query can rank one, and no
  ranking bug can leak one. Filtering after retrieval would mean the value had
  already been loaded, compared and scored before something remembered to hide
  it.
* **The external route does not exist.** ``send_external`` is denied for every
  message including entirely innocuous ones -- not because the policy says so,
  but because this package contains no network client and no outbound call.
  The policy records the fact; it is not what enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .sensitive import RECOMMENDED_ACTION, RISK_LEVEL, HUMAN_LABEL, ScanResult

LOCAL_ONLY = "local_only"
CONFIRM_REQUIRED = "confirm_required"
BLOCKED = "blocked"

ROUTES = [LOCAL_ONLY, CONFIRM_REQUIRED, BLOCKED]

ROUTE_LABEL = {
    LOCAL_ONLY: "Processed locally",
    CONFIRM_REQUIRED: "Confirmation required",
    BLOCKED: "Blocked from downstream use",
}

#: Order of strictness, so a message with several findings takes the strictest.
STRICTNESS = {LOCAL_ONLY: 0, CONFIRM_REQUIRED: 1, BLOCKED: 2}

# Operations a message can be subjected to, in the order they happen.
CLASSIFY = "classify"
EXTRACT = "extract"
INDEX = "index_for_search"
STORE_MASKED = "store_masked_text"
STORE_RAW = "store_raw_text"
QUOTE = "quote_in_answer"
EXPORT = "export_to_file"
SEND_EXTERNAL = "send_to_external_service"

OPERATIONS = [CLASSIFY, EXTRACT, INDEX, STORE_MASKED, STORE_RAW, QUOTE,
              EXPORT, SEND_EXTERNAL]


@dataclass(frozen=True)
class Policy:
    """One rule: which L1 recommendation it answers, and what it permits."""

    policy_id: str
    route: str
    applies_to_action: str
    allowed: frozenset
    rationale: str


#: The policy table. Keyed by the recommended action L1 already produced, so
#: the two layers cannot drift apart: adding a sensitivity type in
#: `mint/sensitive.py` automatically inherits the route of its action.
POLICIES: Dict[str, Policy] = {
    "do_not_store": Policy(
        policy_id="P1-CREDENTIAL",
        route=BLOCKED,
        applies_to_action="do_not_store",
        allowed=frozenset({CLASSIFY, STORE_MASKED}),
        rationale=(
            "the message contains a live credential or financial identifier. "
            "It is classified so it can be counted and audited, and its masked "
            "form is kept as a record, but it is never indexed, never quoted "
            "and never written to an output file"
        ),
    ),
    "do_not_send_to_external_service": Policy(
        policy_id="P2-IDENTITY",
        route=LOCAL_ONLY,
        applies_to_action="do_not_send_to_external_service",
        allowed=frozenset({CLASSIFY, EXTRACT, INDEX, STORE_MASKED, QUOTE}),
        rationale=(
            "the message contains identity or health data. Local analysis is "
            "permitted on the masked text, but it is excluded from exported "
            "files and could never be sent to a third party"
        ),
    ),
    "ask_for_confirmation": Policy(
        policy_id="P3-CONTACT",
        route=CONFIRM_REQUIRED,
        applies_to_action="ask_for_confirmation",
        allowed=frozenset({CLASSIFY, EXTRACT, INDEX, STORE_MASKED}),
        rationale=(
            "the message contains private contact details. These are often "
            "exactly what a user is searching for, so the message stays "
            "searchable in masked form -- but quoting or exporting it needs an "
            "explicit confirmation first"
        ),
    ),
    "safe_to_process_locally": Policy(
        policy_id="P4-MENTION",
        route=LOCAL_ONLY,
        applies_to_action="safe_to_process_locally",
        allowed=frozenset({CLASSIFY, EXTRACT, INDEX, STORE_MASKED, QUOTE,
                           EXPORT}),
        rationale=(
            "the message refers to credentials but contains no credential "
            "value, so there is nothing to protect beyond ordinary care"
        ),
    ),
}

DEFAULT_POLICY = Policy(
    policy_id="P0-DEFAULT",
    route=LOCAL_ONLY,
    applies_to_action="none",
    allowed=frozenset({CLASSIFY, EXTRACT, INDEX, STORE_MASKED, QUOTE, EXPORT}),
    rationale="no sensitive value was detected in this message",
)


@dataclass
class RouteDecision:
    """What one message may be used for, and why."""

    message_id: str
    route: str
    policy_id: str
    risk: Optional[str]
    sensitivity_types: List[str] = field(default_factory=list)
    allowed_operations: List[str] = field(default_factory=list)
    denied_operations: List[str] = field(default_factory=list)
    rationale: str = ""
    confirmation_prompt: Optional[str] = None

    @property
    def indexable(self) -> bool:
        return INDEX in self.allowed_operations

    @property
    def quotable(self) -> bool:
        return QUOTE in self.allowed_operations

    @property
    def exportable(self) -> bool:
        return EXPORT in self.allowed_operations

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "route": self.route,
            "route_label": ROUTE_LABEL[self.route],
            "policy_id": self.policy_id,
            "risk": self.risk,
            "sensitivity_types": self.sensitivity_types,
            "sensitivity_labels": [HUMAN_LABEL.get(t, t)
                                   for t in self.sensitivity_types],
            "allowed_operations": self.allowed_operations,
            "denied_operations": self.denied_operations,
            "rationale": self.rationale,
            "confirmation_prompt": self.confirmation_prompt,
        }


def route_message(message_id: str, scan_result: ScanResult) -> RouteDecision:
    """Decide what one message may be used for, from its L1 scan result."""
    if not scan_result.is_sensitive:
        policy = DEFAULT_POLICY
        types: List[str] = []
    else:
        # A message can trip several detectors. Take the strictest policy any
        # of them implies -- the weakest finding must not set the ceiling.
        candidates = [POLICIES.get(RECOMMENDED_ACTION.get(t, ""), DEFAULT_POLICY)
                      for t in scan_result.types]
        policy = max(candidates, key=lambda p: STRICTNESS[p.route])
        types = list(scan_result.types)

    allowed = sorted(policy.allowed, key=OPERATIONS.index)
    denied = [op for op in OPERATIONS if op not in policy.allowed]

    prompt = None
    if policy.route == CONFIRM_REQUIRED:
        label = ", ".join(HUMAN_LABEL.get(t, t) for t in types) or "personal data"
        prompt = (f"{message_id} contains {label}. Confirm before its masked "
                  f"content is quoted in an answer or written to a file.")

    return RouteDecision(
        message_id=message_id,
        route=policy.route,
        policy_id=policy.policy_id,
        risk=scan_result.overall_risk,
        sensitivity_types=types,
        allowed_operations=allowed,
        denied_operations=denied,
        rationale=policy.rationale,
        confirmation_prompt=prompt,
    )


@dataclass
class AnswerRoute:
    """The routing verdict for a whole answer, across all of its evidence."""

    route: str
    withheld_message_ids: List[str] = field(default_factory=list)
    confirm_message_ids: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "route_label": ROUTE_LABEL[self.route],
            "withheld_message_ids": self.withheld_message_ids,
            "confirm_message_ids": self.confirm_message_ids,
            "rationale": self.rationale,
        }


def route_answer(
    evidence_ids: Sequence[str],
    decisions: Dict[str, RouteDecision],
    confirmed: bool = False,
) -> AnswerRoute:
    """Decide how an assembled answer may be presented.

    An answer inherits the strictest route of anything it rests on. Blocked
    evidence is reported as *existing* -- its ID, type and risk -- with its
    content withheld, because "there are three messages here I will not show
    you" is a useful answer and pretending they do not exist is not.
    """
    withheld = [mid for mid in evidence_ids
                if mid in decisions and not decisions[mid].quotable
                and decisions[mid].route == BLOCKED]
    needs_confirm = [mid for mid in evidence_ids
                     if mid in decisions
                     and decisions[mid].route == CONFIRM_REQUIRED]

    if withheld:
        return AnswerRoute(
            route=BLOCKED, withheld_message_ids=withheld,
            confirm_message_ids=needs_confirm,
            rationale=(
                f"{len(withheld)} message(s) in scope carry credentials or "
                f"financial identifiers. They are listed by ID and type so the "
                f"answer is complete, but their content is withheld and they "
                f"were never added to the search index"
            ),
        )
    if needs_confirm and not confirmed:
        return AnswerRoute(
            route=CONFIRM_REQUIRED, confirm_message_ids=needs_confirm,
            rationale=(
                f"{len(needs_confirm)} message(s) in scope contain private "
                f"contact details. Confirm to include their masked content in "
                f"the answer"
            ),
        )
    if needs_confirm and confirmed:
        return AnswerRoute(
            route=LOCAL_ONLY, confirm_message_ids=needs_confirm,
            rationale=("the user confirmed inclusion of the private contact "
                       "details in scope; content shown is masked"),
        )
    return AnswerRoute(
        route=LOCAL_ONLY,
        rationale="no sensitive value is involved in this answer",
    )


def policy_table() -> List[dict]:
    """The policy table, for documentation and for the UI."""
    rows = []
    for policy in [DEFAULT_POLICY] + list(POLICIES.values()):
        types = [t for t, a in RECOMMENDED_ACTION.items()
                 if a == policy.applies_to_action]
        rows.append({
            "policy_id": policy.policy_id,
            "route": policy.route,
            "route_label": ROUTE_LABEL[policy.route],
            "applies_to_action": policy.applies_to_action,
            "sensitivity_types": types,
            "risk_levels": sorted({RISK_LEVEL[t] for t in types}) if types else [],
            "allowed_operations": sorted(policy.allowed, key=OPERATIONS.index),
            "denied_operations": [op for op in OPERATIONS
                                  if op not in policy.allowed],
            "rationale": policy.rationale,
        })
    return rows
