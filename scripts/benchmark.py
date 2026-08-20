"""Benchmark the retrieval index: baseline against optimised.

    python scripts/benchmark.py --input data/messages.csv data/l2_messages.csv \
        --out outputs/benchmark_report.json

What is being compared
----------------------
`v1-exact-lexical`   TF-IDF cosine scored against every document. No
                     approximation; also the reference ranking for quality.
`v2-hybrid-pruned`   Pruned inverted index for candidate generation, int8 LSA
                     embeddings, exact rerank of the top candidates.
`v2-lexical-only`    v2 with the semantic layer switched off. Included so the
                     inverted index and the embeddings can be credited
                     separately instead of as one undifferentiated "optimised"
                     -- and because it is what showed the semantic layer was
                     not earning its cost as a scorer.

How it is measured
------------------
* Every configuration answers the *same* queries on the *same* documents in
  the same process, after a warm-up pass, with `time.perf_counter`. Timings
  are reported as median and p95 over repetitions, not as a mean, because a
  mean over a handful of runs mostly measures the slowest outlier.
* Latency is measured at two corpus sizes. The larger one is the corpus
  replicated with distinct IDs -- which is a valid way to measure how the two
  approaches diverge with scale, and is **not** used for quality, where
  duplicating documents would make every metric meaningless.
* Quality uses relevance labels derived from the subject groups: a message is
  relevant to a query if it belongs to the group that query is about. That is
  a real ground truth for retrieval, and it is worth being explicit about what
  it does and does not test -- it measures whether retrieval finds the
  messages the grouper considers related. It does not independently verify the
  grouping.
* Query sets are separated on purpose. Literal queries use the subject's own
  words. Paraphrases are written by hand to avoid them, and are split into
  ones whose words *are* in the model's vocabulary and ones whose words are
  not, because those two cases have entirely different outcomes and averaging
  them hides the finding.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mint import ledger as L  # noqa: E402
from mint.embed import load_default as load_semantic  # noqa: E402
from mint.retrieve import (Document, ExactLexicalIndex,  # noqa: E402
                           HybridIndex, _deep_size)
from mint.subject import content_tokens  # noqa: E402

#: Paraphrases written by hand for this benchmark. Each maps to the subject it
#: is about. They deliberately avoid the subject's own wording, which is the
#: only way to test whether retrieval is doing anything beyond string overlap.
PARAPHRASES: List[Tuple[str, str]] = [
    ("what is outstanding on the checklist for privacy",
     "Review the privacy checklist"),
    ("has anyone written back to the client yet",
     "Reply to the client email"),
    ("the book borrowed from the library needs renewing",
     "Renew the library book"),
    ("where has the coursework upload got to",
     "Upload the assignment"),
    ("the form for joining that still needs completing",
     "Complete the onboarding form"),
    ("the presentation that had to be sent again",
     "Send the revised presentation"),
    ("checking the results the model produced",
     "Review the model results"),
    ("the tracker for the project that needs updating",
     "Update the project tracker"),
    ("making copies of the project files so they are safe",
     "Back up the project files"),
    ("recording the video for the demo",
     "Prepare the demo video"),
    # Deliberately out of vocabulary: none of "utility", "coursework fee",
    # "adviser" or "helpline" occurs in the corpus at all.
    ("settle the utility payment", "Pay the electricity bill"),
    ("catch up with my adviser", "Mentor catch-up"),
    ("ring the support helpline", "Call the service centre"),
    ("sign off the timesheet", "Submit the weekly report"),
]


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1)
    return ordered[max(idx, 0)]


def time_queries(index, queries: Sequence[str], repeats: int) -> Dict[str, float]:
    for q in queries[:3]:                       # warm-up
        index.search(q, 10)
    samples: List[float] = []
    for q in queries:
        for _ in range(repeats):
            start = time.perf_counter()
            index.search(q, 10)
            samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "queries": len(queries),
        "samples": len(samples),
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms": round(percentile(samples, 0.95), 4),
        "mean_ms": round(statistics.fmean(samples), 4),
        "max_ms": round(max(samples), 4),
    }


def evaluate(index, cases: Sequence[Tuple[str, List[str]]],
             k: int = 10) -> Dict[str, float]:
    """recall@k, MRR and nDCG@k over labelled queries."""
    recalls, rrs, ndcgs, misses = [], [], [], 0
    for query, relevant in cases:
        gold = set(relevant)
        if not gold:
            continue
        hits = [h.doc_id for h in index.search(query, k)]
        found = [i for i, doc in enumerate(hits) if doc in gold]
        recalls.append(len(found) / min(len(gold), k))
        rrs.append(1.0 / (found[0] + 1) if found else 0.0)
        dcg = sum(1.0 / math.log2(i + 2) for i in found)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
        ndcgs.append(dcg / ideal if ideal else 0.0)
        if not found:
            misses += 1
    n = max(len(recalls), 1)
    return {
        "queries": len(recalls),
        "recall_at_10": round(sum(recalls) / n, 4),
        "mrr": round(sum(rrs) / n, 4),
        "ndcg_at_10": round(sum(ndcgs) / n, 4),
        "queries_with_no_relevant_hit": misses,
    }


def agreement(reference, candidate, queries: Sequence[str],
              k: int = 10) -> Dict[str, float]:
    """How much of the exact index's ranking the optimised one reproduces."""
    overlaps, top1 = [], 0
    for query in queries:
        ref = [h.doc_id for h in reference.search(query, k)]
        cand = [h.doc_id for h in candidate.search(query, k)]
        if not ref:
            continue
        overlaps.append(len(set(ref) & set(cand)) / len(ref))
        if cand and ref and cand[0] == ref[0]:
            top1 += 1
    n = max(len(overlaps), 1)
    return {"recall_at_10_vs_exact": round(sum(overlaps) / n, 4),
            "top1_agreement": round(top1 / n, 4),
            "queries": len(overlaps)}


def build_documents(ledger: L.Ledger, multiplier: int = 1) -> List[Document]:
    docs: List[Document] = []
    for copy in range(multiplier):
        for rec in ledger.records:
            group = ledger.group(rec.group_id) if rec.group_id else None
            text = rec.masked_text
            if group is not None:
                text = f"{text} {group.title}"
            doc_id = rec.message_id if copy == 0 else f"C{copy}_{rec.message_id}"
            docs.append(Document(doc_id=doc_id, text=text,
                                 indexable=rec.route.indexable))
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", nargs="+",
                    default=["data/messages.csv", "data/l2_messages.csv"])
    ap.add_argument("--demo")
    ap.add_argument("--queries", help="extra queries, used for latency only")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--scale", type=int, default=8,
                    help="replication factor for the scaling run")
    ap.add_argument("--out", "-o", default="outputs/benchmark_report.json")
    args = ap.parse_args()

    paths = [(Path(p).name, p if Path(p).is_absolute() else ROOT / p)
             for p in args.input]
    if args.demo:
        demo = Path(args.demo)
        paths.append((demo.name, demo if demo.is_absolute() else ROOT / demo))
    for _, path in paths:
        if not Path(path).exists():
            print(f"error: input not found: {path}", file=sys.stderr)
            return 1

    print("building the ledger ...")
    t0 = time.perf_counter()
    ledger = L.run(paths, build_index=False)
    ingest_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  {len(ledger.records)} messages in {ingest_ms:.0f} ms")

    model = load_semantic()
    if model is None:
        print("error: models/semantic.json is missing; run "
              "scripts/build_index.py first", file=sys.stderr)
        return 1

    docs = build_documents(ledger)

    # ---- query sets -------------------------------------------------------
    by_title = {g.title.lower(): g for g in ledger.groups}
    labelled_literal: List[Tuple[str, List[str]]] = []
    for group in ledger.groups:
        if len(group.members) < 3:
            continue
        labelled_literal.append((group.title, list(group.message_ids)))

    # Split on *content* words, not on every word. Splitting on raw tokens put
    # "settle the utility payment" in the in-vocabulary bucket because "the"
    # is in the vocabulary, which is exactly backwards: the question is whether
    # the model has ever seen the words that carry the meaning.
    labelled_para_in, labelled_para_out = [], []
    vocab = model.vocabulary
    for query, subject in PARAPHRASES:
        group = by_title.get(subject.lower())
        if group is None:
            continue
        content = content_tokens(query)
        known = sum(1 for w in content if w in vocab)
        in_vocab = content and known / len(content) >= 0.5
        target = labelled_para_in if in_vocab else labelled_para_out
        target.append((query, list(group.message_ids)))

    latency_queries = [q for q, _ in labelled_literal[:20]] \
        + [q for q, _ in PARAPHRASES]
    if args.queries:
        qpath = Path(args.queries)
        qpath = qpath if qpath.is_absolute() else ROOT / qpath
        if qpath.exists():
            text = qpath.read_text(encoding="utf-8-sig").lstrip("﻿")
            latency_queries += [r["query"] for r in csv.DictReader(io.StringIO(text))
                                if r.get("query")]

    # ---- build ------------------------------------------------------------
    configs = {}
    print("building indexes ...")
    for name, factory in (
        ("v1-exact-lexical", lambda: ExactLexicalIndex(docs)),
        ("v2-hybrid-pruned", lambda: HybridIndex(docs, model)),
        ("v2-lexical-only", lambda: HybridIndex(docs, model, use_semantic=False)),
        # The design that was measured and rejected: semantic scores blended
        # into every query rather than used as a fallback. Kept in the report
        # so the decision can be checked rather than taken on trust.
        ("v2-semantic-always", lambda: HybridIndex(docs, model,
                                                   semantic_tiebreak=0.5,
                                                   always_semantic=True)),
    ):
        start = time.perf_counter()
        index = factory()
        configs[name] = {"index": index,
                         "build_ms": round((time.perf_counter() - start) * 1000, 2)}
        print(f"  {name:<20s} built in {configs[name]['build_ms']:.0f} ms")

    v1 = configs["v1-exact-lexical"]["index"]
    v2 = configs["v2-hybrid-pruned"]["index"]
    v2lex = configs["v2-lexical-only"]["index"]
    v2always = configs["v2-semantic-always"]["index"]

    # ---- size -------------------------------------------------------------
    semantic_path = ROOT / "models" / "semantic.json"
    dense_float_bytes = len(v2.embeddings) * model.k * 8
    sizes = {
        "v1-exact-lexical": {
            "in_memory_bytes": v1.memory_bytes(),
            "document_vectors_bytes": _deep_size(v1.vectors),
        },
        "v2-hybrid-pruned": {
            "in_memory_bytes": v2.memory_bytes(),
            "document_vectors_bytes": _deep_size(v2.vectors),
            "postings_bytes": _deep_size(v2.postings),
            "int8_embeddings_bytes": v2.semantic_bytes(),
            "same_embeddings_as_float64_bytes": dense_float_bytes,
            "embedding_compression_ratio":
                round(dense_float_bytes / max(v2.semantic_bytes(), 1), 2),
        },
        "v2-lexical-only": {
            "in_memory_bytes": v2lex.memory_bytes(),
            "postings_bytes": _deep_size(v2lex.postings),
        },
        "v2-semantic-always": {
            "in_memory_bytes": v2always.memory_bytes(),
        },
        "semantic_model_artifact_bytes": (semantic_path.stat().st_size
                                          if semantic_path.exists() else 0),
        "semantic_model_metadata": model.metadata,
    }

    # ---- latency ----------------------------------------------------------
    print(f"timing {len(latency_queries)} queries x {args.repeats} repeats ...")
    latency = {name: time_queries(cfg["index"], latency_queries, args.repeats)
               for name, cfg in configs.items()}

    scaled = {}
    if args.scale > 1:
        print(f"scaling run: corpus x{args.scale} ...")
        big_docs = build_documents(ledger, multiplier=args.scale)
        big_v1 = ExactLexicalIndex(big_docs)
        big_v2 = HybridIndex(big_docs, model)
        scaled = {
            "documents": len(big_docs),
            "note": ("the corpus replicated with distinct IDs. This measures "
                     "how the two designs diverge with corpus size and is not "
                     "used for any quality number"),
            "v1-exact-lexical": time_queries(big_v1, latency_queries,
                                             max(args.repeats // 3, 5)),
            "v2-hybrid-pruned": time_queries(big_v2, latency_queries,
                                             max(args.repeats // 3, 5)),
        }
        del big_v1, big_v2, big_docs

    # ---- quality ----------------------------------------------------------
    print("scoring retrieval quality ...")
    quality = {}
    for name, cfg in configs.items():
        index = cfg["index"]
        quality[name] = {
            "literal_queries": evaluate(index, labelled_literal),
            "paraphrase_in_vocabulary": evaluate(index, labelled_para_in),
            "paraphrase_out_of_vocabulary": evaluate(index, labelled_para_out),
        }
    fidelity = {
        "v2-hybrid-pruned": agreement(v1, v2, latency_queries),
        "v2-lexical-only": agreement(v1, v2lex, latency_queries),
        "v2-semantic-always": agreement(v1, v2always, latency_queries),
    }

    def speedup(a: str, b: str, block) -> float:
        return round(block[a]["median_ms"] / max(block[b]["median_ms"], 1e-9), 2)

    report = {
        "device": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "note": ("single process, no parallelism, no warm OS cache "
                     "control. Absolute numbers are specific to this machine; "
                     "the ratios are the comparable part"),
        },
        "corpus": {
            "messages": len(ledger.records),
            "indexed_documents": v2.size,
            "excluded_by_privacy_routing": len(v2.excluded),
            "batches": [b.to_dict() for b in ledger.batches],
            "ingest_ms": round(ingest_ms, 1),
        },
        "method": {
            "repeats_per_query": args.repeats,
            "latency_queries": len(latency_queries),
            "timer": "time.perf_counter, per-call, median and p95 reported",
            "relevance_labels": ("a message is relevant to a query if it "
                                 "belongs to the subject group that query is "
                                 "about; groups with fewer than three members "
                                 "are excluded"),
            "paraphrases": ("written by hand for this benchmark, split by "
                            "whether their words occur in the semantic model's "
                            "vocabulary at all"),
        },
        "build_ms": {name: cfg["build_ms"] for name, cfg in configs.items()},
        "size": sizes,
        "latency": latency,
        "latency_scaled": scaled,
        "quality": quality,
        "ranking_fidelity_vs_exact": fidelity,
        "finding": (
            "The semantic layer does not earn its cost as a scorer on this "
            "corpus. Blending it into every query (v2-semantic-always) is "
            "slower than lexical scoring, reproduces less of the exact "
            "ranking, and does not improve recall on hand-written "
            "paraphrases. Every subject here is phrased with a consistent "
            "vocabulary, so there is little synonymy for a latent space to "
            "bridge and it mostly blurs an already-sharp signal. The shipped "
            "index therefore uses it only where lexical scoring returns too "
            "few candidates to fill the result set, which costs nothing on "
            "the queries lexical scoring can already answer."
        ),
        "headline": {
            "speedup_v2_vs_v1": speedup("v1-exact-lexical", "v2-hybrid-pruned",
                                        latency),
            "speedup_v2_vs_v1_scaled": (
                speedup("v1-exact-lexical", "v2-hybrid-pruned", scaled)
                if scaled else None),
            "embedding_compression_ratio":
                sizes["v2-hybrid-pruned"]["embedding_compression_ratio"],
            "memory_ratio_v2_over_v1": round(
                sizes["v2-hybrid-pruned"]["in_memory_bytes"]
                / max(sizes["v1-exact-lexical"]["in_memory_bytes"], 1), 2),
        },
    }

    dest = Path(args.out)
    dest = dest if dest.is_absolute() else ROOT / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # ---- print ------------------------------------------------------------
    bar = "=" * 78
    print()
    print(bar)
    print("BENCHMARK")
    print(bar)
    print(f"  device   : {report['device']['platform']}")
    print(f"  corpus   : {report['corpus']['messages']} messages, "
          f"{report['corpus']['indexed_documents']} indexed "
          f"({report['corpus']['excluded_by_privacy_routing']} withheld by "
          f"privacy routing)")
    print(f"  ingest   : {report['corpus']['ingest_ms']:.0f} ms end to end")
    print()
    print(f"  {'configuration':<20s}{'build':>9s}{'median':>10s}{'p95':>10s}"
          f"{'memory':>11s}")
    for name in configs:
        size = sizes.get(name, {}).get("in_memory_bytes", 0)
        print(f"  {name:<20s}{report['build_ms'][name]:>8.0f}m"
              f"{latency[name]['median_ms']:>9.3f}m{latency[name]['p95_ms']:>9.3f}m"
              f"{size/1024:>10.0f}K")
    print(f"\n  speedup, {report['corpus']['indexed_documents']} docs : "
          f"{report['headline']['speedup_v2_vs_v1']}x")
    if scaled:
        print(f"  speedup, {scaled['documents']} docs : "
              f"{report['headline']['speedup_v2_vs_v1_scaled']}x")
    print(f"  int8 embeddings vs float64      : "
          f"{report['headline']['embedding_compression_ratio']}x smaller")
    print(f"  v2 memory vs v1                 : "
          f"{report['headline']['memory_ratio_v2_over_v1']}x")
    print()
    print(f"  {'quality (recall@10)':<28s}{'literal':>10s}"
          f"{'para(in-vocab)':>16s}{'para(oov)':>12s}")
    for name in configs:
        q = quality[name]
        print(f"  {name:<28s}{q['literal_queries']['recall_at_10']:>10.3f}"
              f"{q['paraphrase_in_vocabulary']['recall_at_10']:>16.3f}"
              f"{q['paraphrase_out_of_vocabulary']['recall_at_10']:>12.3f}")
    print()
    for name, value in fidelity.items():
        print(f"  {name} reproduces {value['recall_at_10_vs_exact']:.1%} of the "
              f"exact index's top 10 (top-1 agreement "
              f"{value['top1_agreement']:.1%})")
    print()
    print(f"  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
