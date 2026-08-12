"""The hybrid classifier: deterministic detector + rule frames + linear model.

Three voices, deliberately given different authority:

* **The sensitive detector** is not a voice at all -- it is a veto. If a
  credential or identifier is in the body, the message is
  ``sensitive_information``, full stop. Whether to redact something is a safety
  decision, and safety decisions should not be made by a softmax.
* **The rule frames** are high precision, moderate recall. When a frame fires
  it is almost always right, but it stays silent on anything unusual.
* **The linear model** is high recall. It always has an opinion, including on
  wording the rules have never seen.

Combining them is what produces a *calibrated* confidence rather than a
decorative one. Agreement between two independent methods is real evidence and
earns a boost; disagreement is real evidence too, and earns a penalty plus a
``needs_review`` flag. A single number that only ever went up would tell the
user nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import rules, taxonomy as T
from .model import Prediction, TfidfLogisticModel, load_default
from .sensitive import ScanResult

#: Confidence assigned when the deterministic detector vetoes. Not 1.0: the
#: detector can in principle miss a novel format, and a classifier that claims
#: certainty is not being honest about that.
SENSITIVE_CONFIDENCE = 0.98

#: Below this, the result is surfaced for human review rather than trusted.
REVIEW_THRESHOLD = 0.60


class _Unset:
    """Sentinel distinguishing "load the default model" from "run without one".

    Passing ``model=None`` has to mean *no model*, otherwise there is no way to
    exercise the rules-only degraded path -- and that path is a real supported
    mode, not just a test fixture.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


UNSET = _Unset()


@dataclass
class Classification:
    message_id: str
    category: str
    confidence: float
    reason: str

    #: Topical label when `category` is sensitive_information, so the handling
    #: decision does not destroy the topical one.
    secondary_category: Optional[str] = None

    #: How the two independent voices related to each other.
    agreement: str = "unknown"
    needs_review: bool = False

    rule_category: Optional[str] = None
    rule_confidence: float = 0.0
    rule_signals: List[str] = field(default_factory=list)

    model_category: Optional[str] = None
    model_confidence: float = 0.0
    model_margin: float = 0.0
    top_features: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """The Part-1 record required by the brief, plus provenance."""
        return {
            "message_id": self.message_id,
            "category": self.category,
            "confidence": self.confidence,
            "reason": self.reason,
            "secondary_category": self.secondary_category,
            "needs_review": self.needs_review,
            "evidence": {
                "agreement": self.agreement,
                "rule_category": self.rule_category,
                "rule_confidence": self.rule_confidence,
                "rule_signals": self.rule_signals,
                "model_category": self.model_category,
                "model_confidence": self.model_confidence,
                "model_margin": round(self.model_margin, 4),
                "model_top_features": [
                    {"feature": f, "contribution": c} for f, c in self.top_features
                ],
            },
        }


def _rule_phrase(verdict: rules.RuleVerdict) -> str:
    if not verdict.signals:
        return "no rule frame matched"
    names = ", ".join(s.name.replace("_", " ") for s in verdict.signals[:2])
    return f"matched the {names} frame ({verdict.evidence_phrase})"


def classify(
    message_id: str,
    masked_text: str,
    sender: str,
    scan_result: ScanResult,
    model: Optional[TfidfLogisticModel] | _Unset = UNSET,
) -> Classification:
    """Classify one already-masked message.

    `masked_text` must come from `sensitive.scan`. Nothing in this module is
    designed to be safe on raw text, and it never needs to be.

    Pass ``model=None`` to force the rules-only path; omit `model` to use the
    shipped artifact.
    """
    if isinstance(model, _Unset):
        model = load_default()

    rule_verdict = rules.classify(masked_text, sender)
    rule_signal_names = [f"{s.name}: {s.evidence}" for s in rule_verdict.signals]

    prediction: Optional[Prediction] = model.predict(masked_text) if model else None
    features = model.explain(masked_text) if model else []

    base = Classification(
        message_id=message_id,
        category=T.GENERAL_INFORMATION,
        confidence=0.0,
        reason="",
        rule_category=rule_verdict.category,
        rule_confidence=rule_verdict.confidence,
        rule_signals=rule_signal_names,
        model_category=prediction.label if prediction else None,
        model_confidence=prediction.confidence if prediction else 0.0,
        model_margin=prediction.margin if prediction else 0.0,
        top_features=features,
    )

    # ---- 1. The deterministic veto ---------------------------------------
    if scan_result.is_sensitive and scan_result.overall_risk in ("high", "medium"):
        types = ", ".join(t.replace("_", " ") for t in scan_result.types)
        # Keep the topical reading as secondary information.
        topical = rule_verdict.category
        if topical in (None, T.GENERAL_INFORMATION) and prediction:
            topical = prediction.label
        base.category = T.SENSITIVE_INFORMATION
        base.confidence = SENSITIVE_CONFIDENCE
        base.secondary_category = topical
        base.agreement = "deterministic_detector"
        base.reason = (
            f"The message body contains {types}, detected by a deterministic "
            f"pattern rule rather than a statistical guess; the value has been "
            f"masked and the message is routed by risk, not topic."
        )
        return base

    # ---- 2. Rules abstained: the model decides alone ----------------------
    if rule_verdict.category is None:
        if prediction is None:
            base.category = T.GENERAL_INFORMATION
            base.confidence = 0.35
            base.agreement = "no_signal"
            base.needs_review = True
            base.reason = (
                "No rule frame matched and no trained model is available, so "
                "this falls back to general information with low confidence."
            )
            return base
        # A model-only decision is real but unsupported, so it is discounted.
        conf = round(min(prediction.confidence * 0.88, 0.88), 4)
        base.category = prediction.label
        base.confidence = conf
        base.agreement = "model_only"
        base.needs_review = conf < REVIEW_THRESHOLD
        base.reason = (
            f"No rule frame matched, so the decision rests on the trained "
            f"classifier, which assigned {prediction.label} at p="
            f"{prediction.confidence:.2f} (margin {prediction.margin:.2f} over "
            f"the next category). Confidence is discounted because no "
            f"independent rule corroborated it."
        )
        return base

    # ---- 3. Model unavailable: rules decide alone -------------------------
    if prediction is None:
        base.category = rule_verdict.category
        base.confidence = round(rule_verdict.confidence * 0.92, 4)
        base.agreement = "rule_only"
        base.needs_review = base.confidence < REVIEW_THRESHOLD
        base.reason = (
            f"The rule layer {_rule_phrase(rule_verdict)}; no trained model was "
            f"available to corroborate it."
        )
        return base

    # ---- 4. Both voices spoke --------------------------------------------
    if prediction.label == rule_verdict.category:
        # Independent agreement: take the stronger signal and add a small,
        # bounded bonus. Capped at 0.99 -- never claim certainty.
        conf = round(min(max(rule_verdict.confidence, prediction.confidence) + 0.04,
                         0.99), 4)
        base.category = rule_verdict.category
        base.confidence = conf
        base.agreement = "rule_and_model_agree"
        base.needs_review = rule_verdict.hedged and conf < REVIEW_THRESHOLD + 0.15
        hedge = (
            " The wording is hedged, so the underlying detail may be incomplete."
            if rule_verdict.hedged else ""
        )
        base.reason = (
            f"The rule layer {_rule_phrase(rule_verdict)}, and the trained "
            f"classifier independently agreed at p={prediction.confidence:.2f}."
            f"{hedge}"
        )
        return base

    # Disagreement: trust the more confident voice, but say so and damp hard.
    rule_wins = rule_verdict.confidence >= prediction.confidence
    winner = rule_verdict.category if rule_wins else prediction.label
    loser = prediction.label if rule_wins else rule_verdict.category
    spread = abs(rule_verdict.confidence - prediction.confidence)
    conf = round(max(0.30, min(0.72, 0.45 + spread * 0.5)), 4)

    base.category = winner
    base.confidence = conf
    base.agreement = "disagreement"
    base.needs_review = True
    base.reason = (
        f"The two methods disagreed: the rule layer "
        f"{_rule_phrase(rule_verdict)} suggesting {rule_verdict.category} "
        f"({rule_verdict.confidence:.2f}), while the trained classifier "
        f"preferred {prediction.label} (p={prediction.confidence:.2f}). "
        f"Resolved to {winner} as the more confident reading; flagged for "
        f"review because {loser} is a defensible alternative."
    )
    return base
