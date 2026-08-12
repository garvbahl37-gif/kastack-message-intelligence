"""Pure-Python inference for the TF-IDF + logistic-regression classifier.

The model is *trained* with scikit-learn (`scripts/train.py`) and then exported
to a plain JSON artifact: vocabulary, IDF weights, coefficient matrix and
intercepts. This module re-implements the forward pass in ~80 lines of standard
library Python.

Why bother, instead of just pickling the sklearn estimator?

* **Deployability.** The serving environment needs no numpy, scipy or
  scikit-learn. The Vercel function installs one dependency (FastAPI) and cold
  starts fast, instead of unpacking a ~100 MB scientific stack to do 900 dot
  products.
* **Auditability.** A pickle is opaque and executes code on load. A JSON file
  of numbers can be read, diffed and checked into version control, and it is
  easy to prove it contains no message text.
* **Explainability.** Because the forward pass is right here, the per-class
  score can be decomposed into the individual token contributions that produced
  it -- which is what `explain()` returns and what the UI shows.

The arithmetic mirrors scikit-learn exactly: sublinear TF, smoothed IDF, L2
normalisation, then a softmax over the linear scores. `tests/test_model.py`
asserts agreement with sklearn's own `predict_proba` to 1e-9.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .normalize import tokenize


@dataclass
class Prediction:
    label: str
    confidence: float
    #: Full probability distribution, highest first.
    distribution: List[Tuple[str, float]]

    @property
    def margin(self) -> float:
        """Gap between the top two classes. A small margin means 'unsure'."""
        if len(self.distribution) < 2:
            return self.distribution[0][1] if self.distribution else 0.0
        return self.distribution[0][1] - self.distribution[1][1]


class TfidfLogisticModel:
    """Feed-forward inference over an exported linear text classifier."""

    def __init__(self, artifact: dict) -> None:
        self.classes: List[str] = artifact["classes"]
        self.vocabulary: Dict[str, int] = artifact["vocabulary"]
        self.idf: List[float] = artifact["idf"]
        self.coef: List[List[float]] = artifact["coef"]
        self.intercept: List[float] = artifact["intercept"]
        self.ngram_max: int = artifact.get("ngram_max", 2)
        self.sublinear_tf: bool = artifact.get("sublinear_tf", True)
        self.metadata: dict = artifact.get("metadata", {})

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "TfidfLogisticModel":
        with Path(path).open(encoding="utf-8") as fh:
            return cls(json.load(fh))

    # -- feature extraction -------------------------------------------------
    def _ngrams(self, tokens: Sequence[str]) -> List[str]:
        """Unigrams through `ngram_max`-grams, joined by a space as sklearn does."""
        grams: List[str] = list(tokens)
        for n in range(2, self.ngram_max + 1):
            grams.extend(
                " ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
            )
        return grams

    def vectorize(self, text: str) -> Dict[int, float]:
        """Sparse L2-normalised TF-IDF vector as {feature_index: weight}."""
        counts: Dict[int, int] = {}
        for gram in self._ngrams(tokenize(text)):
            idx = self.vocabulary.get(gram)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1

        vec: Dict[int, float] = {}
        for idx, count in counts.items():
            tf = 1.0 + math.log(count) if self.sublinear_tf else float(count)
            vec[idx] = tf * self.idf[idx]

        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for idx in vec:
                vec[idx] /= norm
        return vec

    # -- prediction ---------------------------------------------------------
    def _scores(self, vec: Dict[int, float]) -> List[float]:
        out = []
        for c, row in enumerate(self.coef):
            s = self.intercept[c]
            for idx, val in vec.items():
                s += row[idx] * val
            out.append(s)
        return out

    @staticmethod
    def _softmax(scores: Sequence[float]) -> List[float]:
        # Binary logistic regression stores a single decision row; expand it.
        if len(scores) == 1:
            p = 1.0 / (1.0 + math.exp(-scores[0]))
            return [1.0 - p, p]
        top = max(scores)
        exps = [math.exp(s - top) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]

    def predict(self, text: str) -> Prediction:
        probs = self._softmax(self._scores(self.vectorize(text)))
        pairs = sorted(zip(self.classes, probs), key=lambda p: -p[1])
        return Prediction(
            label=pairs[0][0],
            confidence=round(pairs[0][1], 4),
            distribution=[(c, round(p, 4)) for c, p in pairs],
        )

    # -- interpretability ---------------------------------------------------
    def explain(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """The tokens that pushed hardest toward the predicted class.

        Because the model is linear, each feature's contribution to the winning
        class is exactly `weight * coefficient` -- no approximation needed.
        """
        vec = self.vectorize(text)
        if not vec:
            return []
        scores = self._scores(vec)
        winner = max(range(len(scores)), key=scores.__getitem__)
        inverse = {i: g for g, i in self.vocabulary.items()}
        contributions = [
            (inverse[idx], self.coef[winner][idx] * val) for idx, val in vec.items()
        ]
        contributions.sort(key=lambda c: -c[1])
        return [(g, round(c, 4)) for g, c in contributions[:top_k]]


_CACHE: Dict[str, TfidfLogisticModel] = {}


def default_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "classifier.json"


def load_default() -> Optional[TfidfLogisticModel]:
    """Load the shipped model, cached. Returns None if it has not been trained.

    The system degrades gracefully without it: the rule layer alone still
    classifies every message, just with less confidence separation.
    """
    path = default_model_path()
    key = str(path)
    if key not in _CACHE:
        if not path.exists():
            return None
        _CACHE[key] = TfidfLogisticModel.load(path)
    return _CACHE[key]
