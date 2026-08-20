"""Retrieval: two indexes over the same corpus, one simple and one optimised.

The brief asks which component was optimised and how the two versions compare.
This module is that component, and it contains both versions rather than a
description of the old one, so the benchmark in `scripts/benchmark.py` measures
running code on both sides.

**`ExactLexicalIndex` -- the baseline.**
TF-IDF cosine, scored document-at-a-time over every document in the corpus.
Roughly forty lines, no approximation anywhere. Its ranking is also the
reference the optimised index is scored against: "how much ranking did the
speed cost?" only has an answer if something exact is there to ask.

**`HybridIndex` -- the optimised version.** Three changes:

1. *An inverted index instead of a full scan.* The baseline visits all 1,100
   documents and their ~19 terms each. This visits only the posting lists of
   the query's terms. Terms whose IDF is below a floor are not given posting
   lists at all -- they appear in most documents, so they cost the most to
   traverse and separate the least.
2. *A semantic signal.* Each document also carries an int8 LSA embedding
   (`mint/embed.py`). Lexical retrieval cannot match a paraphrase that shares
   no words; this can.
3. *Two stages.* Cheap lexical scoring proposes candidates; the exact hybrid
   score is computed only for the top `rerank_depth` of them. Reranking 60
   documents instead of 1,100 is where most of the saving comes from.

The dense semantic scan is **adaptive**: it runs only when the lexical stage
returns too few candidates, which is precisely the paraphrase case. Running it
every time would cost more than the inverted index saves, and the benchmark
reports how often each path was taken so the claim is checkable rather than
asserted.

One property is not a performance decision: a document whose routing decision
denies indexing is never added to either index. Both classes count what they
refused so the exclusion is visible in the output.
"""

from __future__ import annotations

import array
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .embed import Quantised, SemanticModel, ngrams, quantise
from .normalize import tokenize

#: Terms appearing in more than this share of the corpus get no posting list.
#: Expressed as a share rather than as a raw IDF so the floor moves with the
#: corpus: an absolute IDF cut that prunes usefully at 1,000 documents prunes
#: nothing at 10,000, which is the mistake the first version of this made.
MAX_DF_SHARE = 0.22

#: Query terms used for candidate generation, highest IDF first.
MAX_QUERY_TERMS = 12

#: Documents rescored exactly in stage two. Chosen by sweep: at 60 the exact
#: index's top 10 is reproduced 97.1% of the time, at 240 it is 97.9%, and the
#: median query cost is the same to within noise on this corpus. 200 takes the
#: fidelity without leaving the depth free to grow unboundedly on a larger one.
RERANK_DEPTH = 200

#: Components used for the coarse dense scan. The SVD orders its components by
#: explained variance, so the first slice is not an arbitrary truncation -- it
#: is the most informative part of the vector. Scanning on 32 of 128 costs a
#: quarter as much and only has to be good enough to nominate candidates,
#: because the survivors are rescored on all 128.
COARSE_DIMS = 32

#: How much the semantic score may move a result once lexical scoring has
#: spoken. Small on purpose: measurement showed that blending the two as
#: equals made retrieval slower *and* slightly worse on this corpus, because
#: every subject here is written with a consistent vocabulary, so lexical
#: overlap is already close to exact and the latent space can only blur it.
#: See `scripts/benchmark.py` and the README for the numbers that set this.
SEMANTIC_TIEBREAK = 0.15


@dataclass
class Document:
    """One indexable unit. Masked text only, by the time it gets here."""

    doc_id: str
    text: str
    #: False when the routing layer denies indexing. Such documents are
    #: counted and skipped, never stored.
    indexable: bool = True
    meta: dict = field(default_factory=dict)


@dataclass
class Hit:
    doc_id: str
    score: float
    lexical: float = 0.0
    semantic: float = 0.0
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "message_id": self.doc_id,
            "score": round(self.score, 4),
            "lexical_score": round(self.lexical, 4),
            "semantic_score": round(self.semantic, 4),
            "matched_terms": self.matched_terms[:6],
        }


@dataclass
class SearchStats:
    """What the last search actually did. Used by the benchmark and the UI."""

    candidates: int = 0
    postings_visited: int = 0
    reranked: int = 0
    dense_scan: bool = False
    documents_scanned: int = 0


def _deep_size(obj, seen=None) -> int:
    """Approximate in-memory footprint. Approximate is enough for a ratio."""
    seen = seen if seen is not None else set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_size(k, seen) + _deep_size(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += _deep_size(item, seen)
    elif hasattr(obj, "__dict__"):
        size += _deep_size(vars(obj), seen)
    return size


class _LexicalBase:
    """Shared TF-IDF construction. Both indexes learn IDF from the corpus."""

    def __init__(self, ngram_max: int = 2) -> None:
        self.ngram_max = ngram_max
        self.doc_ids: List[str] = []
        self.meta: List[dict] = []
        self.vectors: List[Dict[str, float]] = []
        self.idf: Dict[str, float] = {}
        self.excluded: List[str] = []

    def _terms(self, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for gram in ngrams(tokenize(text), self.ngram_max):
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    def _fit(self, docs: Sequence[Document]) -> List[Dict[str, int]]:
        raw: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for doc in docs:
            if not doc.indexable:
                self.excluded.append(doc.doc_id)
                continue
            counts = self._terms(doc.text)
            raw.append(counts)
            self.doc_ids.append(doc.doc_id)
            self.meta.append(doc.meta)
            for term in counts:
                df[term] = df.get(term, 0) + 1
        n = max(len(raw), 1)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        return raw

    def _vector(self, counts: Dict[str, int]) -> Dict[str, float]:
        vec: Dict[str, float] = {}
        for term, c in counts.items():
            weight = self.idf.get(term)
            if weight is None:
                continue
            vec[term] = (1.0 + math.log(c)) * weight
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for term in vec:
                vec[term] /= norm
        return vec

    def query_vector(self, query: str) -> Dict[str, float]:
        """A query is vectorised exactly as a document is, with corpus IDF.

        Terms the corpus has never seen are dropped rather than given a
        maximal IDF: an unknown term carries no evidence about *these*
        documents, and treating it as maximally informative would let a typo
        dominate the ranking.
        """
        return self._vector(self._terms(query))

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def memory_bytes(self) -> int:
        return _deep_size(self.vectors) + _deep_size(self.idf)


class ExactLexicalIndex(_LexicalBase):
    """Baseline: exact TF-IDF cosine, scored against every document."""

    name = "v1-exact-lexical"

    def __init__(self, docs: Sequence[Document], ngram_max: int = 2) -> None:
        super().__init__(ngram_max)
        raw = self._fit(docs)
        self.vectors = [self._vector(counts) for counts in raw]
        self.stats = SearchStats()

    def search(self, query: str, k: int = 10) -> List[Hit]:
        qvec = self.query_vector(query)
        stats = SearchStats(documents_scanned=len(self.vectors))
        hits: List[Hit] = []
        if not qvec:
            self.stats = stats
            return hits
        for i, vec in enumerate(self.vectors):
            score = 0.0
            matched: List[str] = []
            # Iterate the smaller side. A query has ~8 terms and a document
            # ~19, so this is the cheaper direction and is what makes the
            # baseline a fair opponent rather than a straw man.
            for term, qw in qvec.items():
                dw = vec.get(term)
                if dw is not None:
                    score += qw * dw
                    matched.append(term)
            if score > 0:
                hits.append(Hit(self.doc_ids[i], score, lexical=score,
                                matched_terms=matched))
        stats.candidates = len(hits)
        self.stats = stats
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return hits[:k]


class HybridIndex(_LexicalBase):
    """Optimised: pruned inverted index, LSA rerank, adaptive dense fallback."""

    name = "v2-hybrid-pruned"

    def __init__(
        self,
        docs: Sequence[Document],
        model: Optional[SemanticModel] = None,
        ngram_max: int = 2,
        max_df_share: float = MAX_DF_SHARE,
        rerank_depth: int = RERANK_DEPTH,
        semantic_tiebreak: float = SEMANTIC_TIEBREAK,
        use_semantic: bool = True,
        always_semantic: bool = False,
    ) -> None:
        super().__init__(ngram_max)
        self.max_df_share = max_df_share
        self.rerank_depth = rerank_depth
        self.semantic_tiebreak = semantic_tiebreak
        #: Restores the design this one replaced: blend the semantic score into
        #: every query rather than using it as a fallback. Kept so the
        #: benchmark can measure the rejected alternative instead of asserting
        #: that it was worse.
        self.always_semantic = always_semantic
        self.model = model if use_semantic else None

        raw = self._fit(docs)
        self.vectors = [self._vector(counts) for counts in raw]

        # The share cut is converted once into the IDF value it corresponds
        # to, so the hot path compares floats rather than recomputing shares.
        n = max(len(self.vectors), 1)
        self.idf_floor = math.log((n + 1) / (self.max_df_share * n + 1)) + 1.0

        # -- inverted index over the informative terms only ------------------
        # Postings are two parallel arrays rather than a list of tuples. A
        # (int, float) tuple costs about 80 bytes once its members are counted;
        # the arrays hold the same pair in 8. Across ~20,000 postings that is
        # the difference between 1.6 MB and 160 KB, and it changes nothing
        # about the traversal.
        built: Dict[str, Tuple[List[int], List[float]]] = {}
        for i, vec in enumerate(self.vectors):
            for term, weight in vec.items():
                if self.idf.get(term, 0.0) < self.idf_floor:
                    continue
                ids, weights = built.setdefault(term, ([], []))
                ids.append(i)
                weights.append(weight)
        self.postings: Dict[str, Tuple[array.array, array.array]] = {
            term: (array.array("i", ids), array.array("f", weights))
            for term, (ids, weights) in built.items()
        }
        self.pruned_terms = sum(1 for v in self.idf.values()
                                if v < self.idf_floor)

        # -- int8 semantic embeddings ---------------------------------------
        self.embeddings: List[Optional[Quantised]] = []
        #: The leading slice of each embedding, kept separately so the coarse
        #: scan touches a quarter of the data.
        self.coarse: List[array.array] = []
        if self.model is not None:
            indexed = [d for d in docs if d.indexable]
            self.embeddings = [self.model.embed_quantised(d.text)
                               for d in indexed]
            self.coarse = [e.codes[:COARSE_DIMS] for e in self.embeddings]
        self.stats = SearchStats()

    # -- stage one ----------------------------------------------------------
    def _candidates(self, qvec: Dict[str, float],
                    stats: SearchStats) -> Dict[int, float]:
        terms = [(t, w) for t, w in qvec.items()
                 if self.idf.get(t, 0.0) >= self.idf_floor]
        terms.sort(key=lambda tw: -self.idf.get(tw[0], 0.0))
        acc: Dict[int, float] = {}
        for term, qw in terms[:MAX_QUERY_TERMS]:
            postings = self.postings.get(term)
            if not postings:
                continue
            ids, weights = postings
            stats.postings_visited += len(ids)
            for i, dw in zip(ids, weights):
                acc[i] = acc.get(i, 0.0) + qw * dw
        return acc

    # -- stage two ----------------------------------------------------------
    def _exact(self, i: int, qvec: Dict[str, float]) -> Tuple[float, List[str]]:
        vec = self.vectors[i]
        score = 0.0
        matched: List[str] = []
        for term, qw in qvec.items():
            dw = vec.get(term)
            if dw is not None:
                score += qw * dw
                matched.append(term)
        return score, matched

    def search(self, query: str, k: int = 10) -> List[Hit]:
        qvec = self.query_vector(query)
        stats = SearchStats()
        if not qvec and self.model is None:
            self.stats = stats
            return []

        acc = self._candidates(qvec, stats)

        # The semantic layer is a **recall** device, not a scoring device.
        #
        # That is a measured decision, not a preference. Blending lexical and
        # semantic scores on every query cost 4-7x in latency, reproduced less
        # of the exact ranking (94% against 97%) and did not improve recall on
        # hand-written paraphrases (0.79 against 0.81). On a corpus where each
        # subject is phrased consistently there is no synonymy to bridge, so
        # the latent space can only blur a signal that is already sharp.
        #
        # What it *can* do is find documents lexical scoring cannot see at all.
        # So the query embedding is not even computed unless the lexical stage
        # fails to fill the result set, and when it is, the semantic score only
        # breaks ties and ranks the candidates that lexical scoring never
        # proposed. Queries the lexical stage can answer get exactly the
        # ranking they would have got without a semantic layer at all.
        qemb = None
        if self.model is not None and self.always_semantic:
            qemb = self.model.embed_quantised(query)
        if self.model is not None and len(acc) < k:
            qemb = qemb or self.model.embed_quantised(query)
            stats.dense_scan = True
            stats.documents_scanned = len(self.coarse)
            probe = qemb.codes[:COARSE_DIMS]
            scored = []
            for i, code in enumerate(self.coarse):
                acc_i = 0
                for a, b in zip(probe, code):
                    acc_i += a * b
                scored.append((acc_i, i))
            scored.sort(reverse=True)
            for _, i in scored[: self.rerank_depth]:
                acc.setdefault(i, 0.0)

        stats.candidates = len(acc)
        top = sorted(acc.items(), key=lambda kv: -kv[1])[: self.rerank_depth]
        stats.reranked = len(top)

        hits: List[Hit] = []
        for i, _ in top:
            lexical, matched = self._exact(i, qvec)
            semantic = 0.0
            if qemb is not None and i < len(self.embeddings) \
                    and self.embeddings[i] is not None:
                semantic = max(0.0, qemb.dot(self.embeddings[i]))
            score = lexical + self.semantic_tiebreak * semantic
            if score > 0:
                hits.append(Hit(self.doc_ids[i], score, lexical=lexical,
                                semantic=semantic, matched_terms=matched))
        self.stats = stats
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return hits[:k]

    def semantic_bytes(self) -> int:
        """Bytes held by the quantised embeddings, excluding object overhead."""
        return sum(e.nbytes() for e in self.embeddings if e is not None)

    def memory_bytes(self) -> int:
        return (_deep_size(self.vectors) + _deep_size(self.idf)
                + _deep_size(self.postings) + _deep_size(self.embeddings))

    def profile(self) -> dict:
        return {
            "index": self.name,
            "documents": self.size,
            "excluded_by_routing": len(self.excluded),
            "vocabulary": len(self.idf),
            "posting_lists": len(self.postings),
            "terms_pruned_below_idf_floor": self.pruned_terms,
            "idf_floor": round(self.idf_floor, 4),
            "max_df_share": self.max_df_share,
            "semantic_bytes": self.semantic_bytes(),
            "rerank_depth": self.rerank_depth,
            "semantic_tiebreak": self.semantic_tiebreak,
            "semantic_role": ("blended into every score"
                              if self.always_semantic
                              else "recall fallback and tie-break only"),
            "semantic_dimensions": self.model.k if self.model else 0,
            "semantic_enabled": self.model is not None,
        }


def build_documents(records: Iterable[dict]) -> List[Document]:
    """Turn ledger records into indexable documents.

    The indexed text is the masked message plus its subject group's title. The
    title is included because a follow-up like "Any update on this?" carries
    almost no searchable content of its own -- what makes it findable is the
    subject it was attached to in Part 2.
    """
    docs: List[Document] = []
    for rec in records:
        text = rec.get("masked_text", "")
        title = rec.get("group_title")
        if title:
            text = f"{text} {title}"
        docs.append(Document(
            doc_id=rec["message_id"],
            text=text,
            indexable=bool(rec.get("indexable", True)),
            meta=rec,
        ))
    return docs
