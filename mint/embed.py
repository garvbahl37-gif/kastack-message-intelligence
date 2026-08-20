"""Local semantic embeddings, and the arithmetic that keeps them small.

Retrieval needs to find messages that mean the same thing in different words.
Lexical overlap gets a long way -- "model-results review" and "review the model
results" share three tokens -- but it fails exactly where it matters most, on
the paraphrase a real user types: "what is still outstanding for the client?"
shares one word with "reply to the client email" and nothing at all with
"following up; is it in progress?".

So each message also gets a dense vector in a space where words that occur in
the same contexts end up pointing the same way. That space is built by
**latent semantic analysis**: TF-IDF over the corpus, then a truncated SVD down
to `k` dimensions. Words never bridge by dictionary lookup; they bridge because
the corpus used them the same way.

Why LSA and not a sentence transformer
--------------------------------------
Honesty first: a MiniLM-class encoder would embed better. It is not used here,
and the reasons are constraints of this system rather than a claim that LSA is
superior.

* The brief forbids sending message text to an external service, so the encoder
  would have to run locally, inside a serverless function, alongside the rest
  of the pipeline. That means shipping PyTorch or ONNX Runtime plus 90 MB of
  weights to answer a query over 1,100 short messages.
* This corpus is ~1,100 messages over ~40 subjects with a 1,348-term
  vocabulary. LSA over a corpus that small and that domain-specific captures
  most of the available structure -- 89.6% of the variance in 128 components --
  and the remaining headroom is not where the system's errors come from.
* The exported model is 187 KB of integers with no message text in it, and the
  forward pass is forty lines of standard-library Python. Both properties are
  checkable by reading the file, which matters for something whose main claim
  is that nothing leaves the machine.

The limitation this buys is real and is stated in the README: LSA cannot relate
two words the corpus never used together, so a genuinely novel synonym will not
bridge.

Two size decisions
------------------
**A sparse projection.** The full projection is 1,348 x 128 floats, and
applying it to a message means touching every component for every term the
message contains. Keeping only each term's `top_components` largest-magnitude
components cuts the work per message by ~4x, because a term's contribution to
the components where its loading is near zero is near zero. What that costs in
retrieval quality is measured, not assumed -- see `scripts/benchmark.py`.

**int8 weights.** Loadings are stored as signed bytes with one scale per
component. That is a 4x reduction in artifact size against float32. It is *not*
a speed optimisation: Python integer arithmetic is no faster than float, and
claiming otherwise would be the kind of unmeasured assertion this project is
supposed to avoid.
"""

from __future__ import annotations

import array
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .normalize import tokenize

#: Sentinel returned when no artifact has been built.
NO_MODEL = None


def ngrams(tokens: Sequence[str], ngram_max: int = 2) -> List[str]:
    """Unigrams through `ngram_max`-grams, joined by a space as sklearn does."""
    grams: List[str] = list(tokens)
    for n in range(2, ngram_max + 1):
        grams.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return grams


@dataclass
class Quantised:
    """A vector stored as signed bytes plus the scale that restores it.

    The codes live in an ``array('b')`` rather than a list. A 128-element list
    of Python ints costs about 4 KB once the objects are counted, which across
    1,100 documents is 4 MB -- more than the float vectors the quantisation was
    supposed to shrink. The array holds the same 128 values in 128 bytes, which
    is what makes int8 an actual size reduction rather than a nominal one.
    """

    codes: array.array
    scale: float

    def dot(self, other: "Quantised") -> float:
        """Cosine similarity between two L2-normalised quantised vectors."""
        if len(self.codes) != len(other.codes):
            return 0.0
        acc = 0
        for a, b in zip(self.codes, other.codes):
            acc += a * b
        return acc * self.scale * other.scale

    def nbytes(self) -> int:
        return self.codes.buffer_info()[1] * self.codes.itemsize


def quantise(vec: Sequence[float]) -> Quantised:
    """Quantise one L2-normalised vector to int8 with a single shared scale."""
    peak = max((abs(v) for v in vec), default=0.0)
    if peak == 0.0:
        return Quantised(array.array("b", bytes(len(vec))), 0.0)
    scale = peak / 127.0
    return Quantised(
        array.array("b", [max(-127, min(127, int(round(v / scale))))
                          for v in vec]),
        scale,
    )


class SemanticModel:
    """Pure-Python forward pass for the exported LSA projection."""

    def __init__(self, artifact: dict) -> None:
        self.vocabulary: Dict[str, int] = artifact["vocabulary"]
        self.idf: List[float] = artifact["idf"]
        self.k: int = artifact["k"]
        self.ngram_max: int = artifact.get("ngram_max", 2)
        self.sublinear_tf: bool = artifact.get("sublinear_tf", True)
        self.component_scale: List[float] = artifact["component_scale"]
        #: term index -> [(component, int8 loading), ...]
        self.rows: Dict[int, List[Tuple[int, int]]] = {
            int(i): [(int(c), int(v)) for c, v in pairs]
            for i, pairs in artifact["rows"].items()
        }
        self.metadata: dict = artifact.get("metadata", {})

    @classmethod
    def load(cls, path: str | Path) -> "SemanticModel":
        with Path(path).open(encoding="utf-8") as fh:
            return cls(json.load(fh))

    # -- forward pass -------------------------------------------------------
    def tfidf(self, text: str) -> Dict[int, float]:
        """Sparse L2-normalised TF-IDF vector over the *model's* vocabulary."""
        counts: Dict[int, int] = {}
        for gram in ngrams(tokenize(text), self.ngram_max):
            idx = self.vocabulary.get(gram)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
        vec: Dict[int, float] = {}
        for idx, n in counts.items():
            tf = 1.0 + math.log(n) if self.sublinear_tf else float(n)
            vec[idx] = tf * self.idf[idx]
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for idx in vec:
                vec[idx] /= norm
        return vec

    def embed(self, text: str) -> List[float]:
        """Project a message into the latent space and L2-normalise it."""
        out = [0.0] * self.k
        for idx, weight in self.tfidf(text).items():
            for comp, code in self.rows.get(idx, ()):
                out[comp] += weight * code * self.component_scale[comp]
        norm = math.sqrt(sum(v * v for v in out))
        if norm > 0:
            out = [v / norm for v in out]
        return out

    def embed_quantised(self, text: str) -> Quantised:
        return quantise(self.embed(text))

    # -- introspection ------------------------------------------------------
    @property
    def density(self) -> float:
        """Mean components retained per term -- how sparse the projection is."""
        if not self.rows:
            return 0.0
        return sum(len(r) for r in self.rows.values()) / len(self.rows)

    def artifact_bytes(self) -> int:
        return int(self.metadata.get("artifact_bytes", 0))


_CACHE: Dict[str, SemanticModel] = {}


def default_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "semantic.json"


def load_default() -> Optional[SemanticModel]:
    """Load the shipped semantic model, cached. None if it has not been built.

    Retrieval degrades to lexical-only without it, which is exactly the
    baseline the benchmark compares against -- so the fallback path is a
    supported mode rather than a broken one.
    """
    path = default_model_path()
    key = str(path)
    if key not in _CACHE:
        if not path.exists():
            return None
        model = SemanticModel.load(path)
        model.metadata.setdefault("artifact_bytes", path.stat().st_size)
        _CACHE[key] = model
    return _CACHE[key]
