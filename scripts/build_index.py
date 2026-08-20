"""Train the LSA projection used for semantic retrieval and export it.

    python scripts/build_index.py --input data/messages.csv data/l2_messages.csv

Steps, in the order they matter:

1. Load every message and **mask it immediately**, exactly as `scripts/train.py`
   does. The projection is fitted on masked text only, so no secret can reach
   the exported vocabulary. That is asserted, not assumed.
2. Fit TF-IDF (1-2 grams, sublinear TF) and a truncated SVD.
3. Sparsify: keep each term's largest-magnitude components and drop the rest,
   which is what makes the per-message forward pass cheap enough to run inside
   a request.
4. Quantise the surviving loadings to int8, one scale per component.
5. Verify the pure-Python forward pass still agrees with scikit-learn, and
   report how much the sparsification and quantisation cost.

The artifact contains n-gram keys, IDF weights and integer loadings. It
contains no message text, and `scripts/train.py`'s vocabulary check is reused
verbatim to prove it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mint.embed import SemanticModel  # noqa: E402
from mint.normalize import tokenize  # noqa: E402
from mint.sensitive import scan  # noqa: E402
from scripts.train import assert_vocabulary_is_clean  # noqa: E402

RANDOM_STATE = 20260820


def load_masked(paths) -> list[str]:
    out: list[str] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8-sig").lstrip("﻿")
        rows = list(csv.DictReader(io.StringIO(text)))
        rows.sort(key=lambda r: (r.get("timestamp", ""), r.get("message_id", "")))
        out.extend(scan(r["message"]).masked_text for r in rows)
    return out


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(an * bn, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", nargs="+",
                    default=["data/messages.csv", "data/l2_messages.csv"])
    ap.add_argument("--k", type=int, default=128, help="latent dimensions")
    ap.add_argument("--top-components", type=int, default=32,
                    help="components retained per term (0 = keep all)")
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--out", "-o", default="models/semantic.json")
    args = ap.parse_args()

    paths = [p if Path(p).is_absolute() else ROOT / p for p in args.input]
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        print(f"error: input not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    docs = load_masked(paths)
    print(f"loaded {len(docs)} messages (masked at load time)")

    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False,
                          ngram_range=(1, 2), sublinear_tf=True,
                          min_df=args.min_df, norm="l2", smooth_idf=True,
                          token_pattern=None)
    X = vec.fit_transform(docs)
    vocabulary = {t: int(i) for t, i in vec.vocabulary_.items()}
    assert_vocabulary_is_clean(vocabulary)
    print(f"vocabulary: {len(vocabulary)} features -- secret-leak check PASSED")

    k = min(args.k, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=k, random_state=RANDOM_STATE).fit(X)
    explained = float(svd.explained_variance_ratio_.sum())
    # components_ is (k, V); the projection applied to a document is its
    # transpose, so each row below is one term's loading vector.
    P = svd.components_.T.astype(np.float64)          # (V, k)
    print(f"SVD: k={k}, explained variance {explained:.4f}")

    dense_emb = X @ P                                  # reference embeddings

    # ---- sparsify ---------------------------------------------------------
    keep = args.top_components or k
    if keep < k:
        order = np.argsort(-np.abs(P), axis=1)[:, :keep]
        mask = np.zeros_like(P, dtype=bool)
        np.put_along_axis(mask, order, True, axis=1)
        Ps = np.where(mask, P, 0.0)
    else:
        Ps = P
    sparse_emb = X @ Ps
    sparse_agree = float(np.mean(cosine_rows(dense_emb, sparse_emb)))
    print(f"sparsified to {keep}/{k} components per term; mean cosine against "
          f"the dense projection = {sparse_agree:.4f}")

    # ---- quantise ---------------------------------------------------------
    peaks = np.max(np.abs(Ps), axis=0)
    scales = np.where(peaks > 0, peaks / 127.0, 1.0)
    Q = np.rint(Ps / scales).astype(np.int16)
    Q = np.clip(Q, -127, 127)
    quant_emb = X @ (Q.astype(np.float64) * scales)
    quant_agree = float(np.mean(cosine_rows(dense_emb, quant_emb)))
    print(f"quantised to int8; mean cosine against the dense projection = "
          f"{quant_agree:.4f}")

    rows: dict[str, list] = {}
    for i in range(Q.shape[0]):
        nz = np.nonzero(Q[i])[0]
        rows[str(i)] = [[int(c), int(Q[i][c])] for c in nz]
    density = sum(len(r) for r in rows.values()) / max(len(rows), 1)

    artifact = {
        "vocabulary": vocabulary,
        "idf": [float(v) for v in vec.idf_],
        "k": int(k),
        "ngram_max": 2,
        "sublinear_tf": True,
        "component_scale": [float(s) for s in scales],
        "rows": rows,
        "metadata": {
            "trained_on": "masked message text only",
            "n_documents": len(docs),
            "vocabulary_size": len(vocabulary),
            "explained_variance": round(explained, 4),
            "top_components_per_term": keep,
            "mean_components_per_term": round(density, 2),
            "cosine_vs_dense_after_sparsification": round(sparse_agree, 4),
            "cosine_vs_dense_after_quantisation": round(quant_agree, 4),
            "note": ("Contains no message text -- only tokenised n-gram keys, "
                     "IDF weights and integer SVD loadings."),
        },
    }

    dest = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, separators=(",", ":"), sort_keys=True)
    size = dest.stat().st_size
    artifact["metadata"]["artifact_bytes"] = size
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, separators=(",", ":"), sort_keys=True)
    print(f"wrote {dest} ({size/1024:.0f} KB, {density:.1f} components/term)")

    # ---- verify the runtime reproduces the reference ----------------------
    model = SemanticModel(artifact)
    sample = docs[:: max(1, len(docs) // 200)]
    deltas = []
    for text in sample:
        pure = np.array(model.embed(text))
        ref = np.asarray((vec.transform([text]) @ P))[0]
        n1, n2 = np.linalg.norm(pure), np.linalg.norm(ref)
        deltas.append(float(pure @ ref / (n1 * n2)) if n1 and n2 else 1.0)
    worst = min(deltas)
    print(f"pure-Python vs scikit-learn on {len(sample)} messages: "
          f"mean cosine {np.mean(deltas):.4f}, worst {worst:.4f}")
    if worst < 0.90:
        raise SystemExit("ABORT: the runtime projection diverges from the "
                         "reference by more than the sparsification budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
