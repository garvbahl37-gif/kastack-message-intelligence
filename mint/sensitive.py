"""Sensitive-information detection and masking.

This module is the first stage of the pipeline and the only place that ever
touches raw message text. Everything downstream (classification, extraction,
serialisation, logging, the web UI) consumes the *masked* text produced here.

Design rules
------------
1. Detection is keyword-anchored regex, not free-floating pattern matching.
   A bare number is not a secret; a number introduced by "OTP is" is. Anchoring
   on the surrounding language is what keeps false positives low on a corpus
   where ordinary messages are full of dates, times and prices.
2. Masking is *surgical*: only the matched value span is replaced, so the
   sentence stays readable and downstream classification still works.
3. The mask token has a **fixed width** (`MASK_TOKEN`) regardless of how long
   the real value was. A mask that mirrored the original length would leak the
   secret's length, which is a genuine (if small) side channel.
4. Nothing here returns the raw value. `SensitiveFinding` deliberately has no
   field for it -- it cannot leak a secret it never stores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Pattern, Tuple

MASK_TOKEN = "******"

# Risk levels, ordered so that `max()` picks the most severe.
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}

# What the system is allowed to do with a message once a given type is found.
RECOMMENDED_ACTION = {
    # Live credentials: useless to retain, catastrophic to leak.
    "one_time_password": "do_not_store",
    "password": "do_not_store",
    "auth_token": "do_not_store",
    "account_recovery_code": "do_not_store",
    "payment_card": "do_not_store",
    "bank_account": "do_not_store",
    # Identity / special-category data: may legitimately need local retention,
    # but must never cross a network boundary to a third party.
    "government_id": "do_not_send_to_external_service",
    "health_information": "do_not_send_to_external_service",
    # Contactability data: useful, moderately sensitive -- let a human decide.
    "postal_address": "ask_for_confirmation",
    "phone_number": "ask_for_confirmation",
    # A reference to credentials that carries no credential.
    "credential_mention": "safe_to_process_locally",
}

RISK_LEVEL = {
    "one_time_password": "high",
    "password": "high",
    "auth_token": "high",
    "account_recovery_code": "high",
    "payment_card": "high",
    "bank_account": "high",
    "government_id": "high",
    # Special-category personal data under GDPR Art. 9 / India's DPDP Act.
    "health_information": "high",
    "postal_address": "medium",
    "phone_number": "medium",
    "credential_mention": "low",
}

HUMAN_LABEL = {
    "one_time_password": "one-time password (OTP)",
    "password": "account password",
    "auth_token": "authentication token",
    "account_recovery_code": "account recovery code",
    "payment_card": "payment card number",
    "bank_account": "bank account number",
    "government_id": "government / personal identification number",
    "health_information": "health information",
    "postal_address": "private postal address",
    "phone_number": "private phone number",
    "credential_mention": "reference to credentials (no value present)",
}


@dataclass(frozen=True)
class Detector:
    """One keyword-anchored rule.

    `pattern` must expose exactly one capturing group: the span to redact. The
    anchoring keywords live *outside* that group so they survive masking and
    remain available to the classifier.
    """

    sensitivity_type: str
    pattern: Pattern[str]
    why: str
    # Detectors with a higher precedence win when two spans overlap.
    precedence: int = 0


def _c(expr: str) -> Pattern[str]:
    return re.compile(expr, re.IGNORECASE)


# A "value-ish" token: letters/digits/symbols with no whitespace, at least 3
# characters, not ending on sentence punctuation.
_VALUE = r"([A-Za-z0-9][A-Za-z0-9@#$%^&*_\-+/.]{2,}[A-Za-z0-9#])"

DETECTORS: List[Detector] = [
    # --- one-time passwords -------------------------------------------------
    Detector(
        "one_time_password",
        _c(
            r"\b(?:OTP|one[\s-]?time\s+(?:password|passcode|code|pin)|"
            r"verification\s+code|security\s+code|auth(?:entication)?\s+code)\b"
            r"(?:\s+(?:is|:|=))?\s*"
            r"(\d[\d\s\-]{2,}\d)"
        ),
        "a one-time authentication code is present in the message body",
        precedence=90,
    ),
    # --- passwords / PINs ---------------------------------------------------
    Detector(
        "password",
        _c(r"\b(?:password|passwd|pwd|passphrase)\b(?:\s+(?:is|:|=))?\s+" + _VALUE),
        "a reusable account password is written out in the message",
        precedence=85,
    ),
    Detector(
        "password",
        _c(r"\b(?:PIN|pin\s+code)\b(?:\s+(?:is|:|=))?\s*(\d{4,8})\b"),
        "a numeric PIN is written out in the message",
        precedence=85,
    ),
    # --- payment cards ------------------------------------------------------
    Detector(
        "payment_card",
        _c(
            r"\b(?:card(?:\s+number)?|credit\s+card|debit\s+card|CVV)\b"
            r"(?:\s+(?:is|:|=))?\s*"
            r"(\d{4}(?:[\s-]?\d{4}){2,3}(?:-\d+)?)"
        ),
        "a payment card number follows an explicit card keyword",
        precedence=80,
    ),
    # --- bank accounts ------------------------------------------------------
    Detector(
        "bank_account",
        _c(
            r"\b(?:bank\s+account|account\s+number|A/?C\s+no\.?|IBAN|IFSC|UPI\s+id)\b"
            r"(?:\s+number)?(?:\s+(?:is|:|=))?\s*"
            r"([A-Z0-9][A-Z0-9\-]{5,})"
        ),
        "a bank account identifier follows an explicit banking keyword",
        precedence=75,
    ),
    # --- auth tokens --------------------------------------------------------
    Detector(
        "auth_token",
        _c(
            r"\b(?:access\s+token|auth(?:entication)?\s+token|bearer\s+token|"
            r"api[\s_-]?key|secret\s+key|session\s+token|token)\b"
            r"(?:\s+(?:is|:|=))?\s+" + _VALUE
        ),
        "an API/session token value follows an explicit token keyword",
        precedence=70,
    ),
    Detector(
        "auth_token",
        # Well-known secret prefixes stand on their own without a keyword.
        _c(r"\b((?:sk|pk|tok|ghp|gho|xox[baprs])[_-][A-Za-z0-9_\-]{6,})\b"),
        "the value carries a recognised secret-token prefix",
        precedence=71,
    ),
    # --- recovery codes -----------------------------------------------------
    Detector(
        "account_recovery_code",
        _c(
            r"\b(?:recovery\s+code|backup\s+code|reset\s+code|"
            r"account\s+recovery(?:\s+code)?)\b(?:\s+(?:is|:|=))?\s+" + _VALUE
        ),
        "an account-recovery secret is written out in the message",
        precedence=72,
    ),
    # --- government / personal IDs -----------------------------------------
    Detector(
        "government_id",
        _c(
            r"\b(?:identification\s+number|identity\s+number|ID\s+number|"
            r"aadhaar(?:\s+number)?|PAN(?:\s+number)?|passport(?:\s+number)?|"
            r"SSN|social\s+security\s+number|employee\s+id|student\s+id)\b"
            r"(?:\s+(?:is|:|=))?\s+" + _VALUE
        ),
        "a personal identification number is present in the message",
        precedence=65,
    ),
    # --- health information -------------------------------------------------
    Detector(
        "health_information",
        _c(
            r"\b(?:test\s+result|lab\s+report|medical\s+report|diagnosis|"
            r"prescription|blood\s+group|blood\s+pressure|medical\s+history)\b"
            r"\s*(?:says|shows|is|:|=)?\s*(.+?)(?=[.!?]|$)"
        ),
        "a personal medical detail is disclosed (special-category personal data)",
        precedence=60,
    ),
    # --- postal address -----------------------------------------------------
    Detector(
        "postal_address",
        _c(
            r"\b(?:home\s+address|residential\s+address|postal\s+address|"
            r"my\s+address|i\s+live\s+at|address\s+is)\b"
            r"\s*(?:is|:|=)?\s*(.+?)(?=[.!?]|$)"
        ),
        "a private residential address is disclosed",
        precedence=55,
    ),
    # --- phone numbers ------------------------------------------------------
    Detector(
        "phone_number",
        _c(
            r"\b(?:contact\s+me\s+(?:on|at)|phone\s+number|mobile\s+number|"
            r"call\s+me\s+(?:on|at)|reach\s+me\s+(?:on|at)|whatsapp)\b"
            r"\s*(?:is|:|=)?\s*"
            # Deliberately liberal about grouping: national conventions split
            # digits differently (5+5, 6+6, 3+3+4, +CC prefixed). Because the
            # keyword anchor already established this is a phone number, the
            # detector should not also insist on a particular local format.
            r"((?:\+\d{1,3}[\s-]?)?\d[\d\s-]{7,16}\d)"
        ),
        "a private phone number is disclosed",
        precedence=50,
    ),
]

# A credential *mention* with no value attached. Checked only after every
# value-bearing detector has failed, so "password is X" never lands here.
CREDENTIAL_MENTION = _c(
    r"\b(login\s+details|login\s+credentials|credentials|account\s+details|"
    r"password|otp|access\s+token)\b"
)


@dataclass
class SensitiveFinding:
    """One detected item. Deliberately holds no raw value."""

    sensitivity_type: str
    risk: str
    recommended_action: str
    reason: str
    # Character span in the *original* message, kept only so masking can be
    # applied and audited. The characters themselves are never retained.
    span: Tuple[int, int] = (0, 0)


@dataclass
class ScanResult:
    masked_text: str
    findings: List[SensitiveFinding] = field(default_factory=list)

    @property
    def is_sensitive(self) -> bool:
        return bool(self.findings)

    @property
    def primary_type(self) -> Optional[str]:
        """The highest-risk finding; ties broken by detector precedence order."""
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: RISK_ORDER[f.risk]).sensitivity_type

    @property
    def overall_risk(self) -> Optional[str]:
        if not self.findings:
            return None
        return max((f.risk for f in self.findings), key=lambda r: RISK_ORDER[r])

    @property
    def overall_action(self) -> Optional[str]:
        """The strictest recommended action across all findings."""
        if not self.findings:
            return None
        strictness = [
            "safe_to_process_locally",
            "ask_for_confirmation",
            "do_not_send_to_external_service",
            "do_not_store",
        ]
        actions = [f.recommended_action for f in self.findings]
        return max(actions, key=strictness.index)

    @property
    def types(self) -> List[str]:
        seen: List[str] = []
        for f in self.findings:
            if f.sensitivity_type not in seen:
                seen.append(f.sensitivity_type)
        return seen


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def scan(text: str) -> ScanResult:
    """Detect sensitive values in `text` and return it with those values masked.

    The returned `masked_text` is safe to log, serialise, display and feed to
    the classifier. The raw `text` should be dropped by the caller immediately.
    """
    candidates: List[Tuple[Detector, Tuple[int, int]]] = []

    for det in DETECTORS:
        for m in det.pattern.finditer(text):
            if m.group(1) is None:
                continue
            value = m.group(1).strip()
            if not value:
                continue
            start, end = m.span(1)
            # Trim trailing punctuation/whitespace out of the redacted span so
            # the masked sentence keeps its full stop.
            while end > start and text[end - 1] in " \t.,;:!?":
                end -= 1
            if end <= start:
                continue
            candidates.append((det, (start, end)))

    # Resolve overlaps: highest precedence wins, then the longer span.
    candidates.sort(key=lambda c: (-c[0].precedence, -(c[1][1] - c[1][0])))
    kept: List[Tuple[Detector, Tuple[int, int]]] = []
    for det, span in candidates:
        if any(_overlaps(span, k[1]) for k in kept):
            continue
        kept.append((det, span))

    findings = [
        SensitiveFinding(
            sensitivity_type=det.sensitivity_type,
            risk=RISK_LEVEL[det.sensitivity_type],
            recommended_action=RECOMMENDED_ACTION[det.sensitivity_type],
            reason=det.why,
            span=span,
        )
        for det, span in kept
    ]

    if not findings:
        # No value found -- is this at least a reference to credentials?
        m = CREDENTIAL_MENTION.search(text)
        if m:
            findings.append(
                SensitiveFinding(
                    sensitivity_type="credential_mention",
                    risk=RISK_LEVEL["credential_mention"],
                    recommended_action=RECOMMENDED_ACTION["credential_mention"],
                    reason=(
                        "the message refers to credentials but contains no "
                        "credential value, so there is nothing to redact"
                    ),
                )
            )
        return ScanResult(masked_text=text, findings=findings)

    # Apply masks right-to-left so earlier spans keep their offsets.
    masked = text
    for _, (start, end) in sorted(kept, key=lambda c: -c[1][0]):
        masked = masked[:start] + MASK_TOKEN + masked[end:]

    # Order findings by position for stable, readable output.
    findings.sort(key=lambda f: f.span)
    return ScanResult(masked_text=masked, findings=findings)


def mask(text: str) -> str:
    """Convenience wrapper returning only the masked text."""
    return scan(text).masked_text
