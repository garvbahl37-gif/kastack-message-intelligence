"""Train the statistical classifier and export it as a plain JSON artifact.

Pipeline
--------
1.  Load the corpus and **mask it immediately**. The model is trained on masked
    text only, so no secret can ever reach the vocabulary. That is enforced,
    not assumed -- see `assert_vocabulary_is_clean` below.
2.  Generate labels by *weak supervision*: the high-precision frames in
    `mint/rules.py` label the messages they are confident about and abstain on
    the rest. The corpus has no ground-truth labels, so this is how a
    supervised model gets a training signal at all.
3.  Fit TF-IDF (1-2 grams, sublinear TF) + multinomial logistic regression.
4.  Evaluate honestly (see below).
5.  Export vocabulary, IDF, coefficients and intercepts to
    `models/classifier.json` for the pure-Python runtime.

Why the evaluation is grouped by template
-----------------------------------------
A plain random split would be close to meaningless here: the 900 messages are
generated from 115 templates, so a random test set is full of sentences whose
near-identical twins are in the training set. The model would score ~100% by
memorising, and the number would tell us nothing.

So the cross-validation is grouped by template: every message from a given
template lands in the same fold. The model is therefore always tested on
sentence *forms it has never seen*. That is the number worth quoting, and it is
substantially harder than the random-split number (which is also reported, for
contrast).

The second honest check is against `eval/gold_labels.csv` -- my own hand
judgements. I wrote both the rules and those labels, so rule-vs-gold agreement
is not independent evidence; model-vs-gold on held-out templates is.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mint import taxonomy as T  # noqa: E402
from mint.normalize import tokenize  # noqa: E402
from mint.rules import weak_label  # noqa: E402
from mint.sensitive import scan  # noqa: E402

RANDOM_STATE = 20260812

# Shapes that must never appear in the exported vocabulary. If any matches a
# vocabulary entry, a secret survived masking and the build must fail.
#
# These are deliberately *generic*. An earlier version also listed the specific
# literals seen in this corpus, which was self-defeating: a hard-coded list of
# real secret fragments is itself a disclosure, and it would have shipped those
# fragments to a public repository inside the very check meant to prevent that.
# Shape-based rules catch the same cases and generalise to unseen data.
FORBIDDEN_VOCAB = [
    re.compile(r"\d{4,}"),                          # long digit runs
    re.compile(r"(?:sk|pk|tok|ghp|gho|xox)[_-]\w+"),  # known secret prefixes
    re.compile(r"[A-Za-z]{3,}[#$%^&*!][A-Za-z0-9]*"),  # password-like symbol mixes
    re.compile(r"\b[A-Z]{2,}-\d+(?:-[A-Z0-9]+)+"),   # ID / recovery-code shapes
]


def load_corpus(path: Path) -> list[dict]:
    """Load messages and mask them before anything else touches the text."""
    with path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    # Chronological order is part of the brief; sort explicitly rather than
    # trusting file order.
    rows.sort(key=lambda r: (r["timestamp"], r["message_id"]))

    out = []
    for r in rows:
        result = scan(r["message"])
        out.append(
            {
                "message_id": r["message_id"],
                "timestamp": r["timestamp"],
                "sender": r["sender"],
                "masked": result.masked_text,
                "risk": result.overall_risk,
            }
        )
    return out


def template_group(masked_text: str) -> str:
    """Group key for GroupKFold: the template a message was generated from."""
    from mint.normalize import canonical

    t = canonical(masked_text)
    t = re.sub(r"\d{4}-\d{2}-\d{2}", "<DATE>", t)
    t = re.sub(r"\d{1,2}:\d{2}", "<TIME>", t)
    t = re.sub(r"\d+", "<N>", t)
    return t


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenize,
                    lowercase=False,       # tokenize() already lowercases
                    preprocessor=None,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                    norm="l2",
                    smooth_idf=True,
                    token_pattern=None,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=4.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def assert_vocabulary_is_clean(vocabulary: dict[str, int]) -> None:
    """Fail the build if any secret-looking string reached the vocabulary."""
    offenders = [
        term for term in vocabulary
        if any(p.search(term) for p in FORBIDDEN_VOCAB)
    ]
    if offenders:
        raise SystemExit(
            "ABORT: sensitive-looking terms leaked into the model vocabulary: "
            + ", ".join(sorted(offenders)[:20])
        )


def main() -> int:
    src = ROOT / "data" / "messages.csv"
    if not src.exists():
        print(f"error: {src} not found.", file=sys.stderr)
        return 1

    corpus = load_corpus(src)
    print(f"loaded {len(corpus)} messages (masked at load time)")

    # ---- weak supervision -------------------------------------------------
    labelled = []
    for row in corpus:
        y = weak_label(row["masked"], row["sender"])
        if y is not None:
            labelled.append((row["masked"], y, template_group(row["masked"])))

    X = [t for t, _, _ in labelled]
    y = np.array([c for _, c, _ in labelled])
    groups = np.array([g for _, _, g in labelled])

    print(f"weak supervision labelled {len(X)}/{len(corpus)} messages "
          f"({len(X)/len(corpus):.0%}) across {len(set(groups))} templates")
    for cat, n in sorted(Counter(y).items()):
        print(f"    {cat:24s} {n:4d}")

    # ---- honest evaluation: unseen templates ------------------------------
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    grouped_true, grouped_pred = [], []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        # A fold is only usable if the training half saw every class.
        if len(set(y[train_idx])) < len(set(y)):
            continue
        pipe = build_pipeline()
        pipe.fit([X[i] for i in train_idx], y[train_idx])
        grouped_pred.extend(pipe.predict([X[i] for i in test_idx]))
        grouped_true.extend(y[test_idx])

    grouped_acc = float(np.mean(np.array(grouped_true) == np.array(grouped_pred)))

    # ---- contrast: the easy (misleading) random split ---------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rand_true, rand_pred = [], []
    for train_idx, test_idx in skf.split(X, y):
        pipe = build_pipeline()
        pipe.fit([X[i] for i in train_idx], y[train_idx])
        rand_pred.extend(pipe.predict([X[i] for i in test_idx]))
        rand_true.extend(y[test_idx])
    random_acc = float(np.mean(np.array(rand_true) == np.array(rand_pred)))

    print()
    print("=" * 68)
    print("CROSS-VALIDATION")
    print("=" * 68)
    print(f"  grouped by template (tested on UNSEEN sentence forms) : {grouped_acc:.4f}")
    print(f"  random stratified split (same forms in train & test)  : {random_acc:.4f}")
    print()
    print("Held-out-template report:")
    print(classification_report(grouped_true, grouped_pred, digits=3, zero_division=0))

    labels_sorted = sorted(set(grouped_true))
    cm = confusion_matrix(grouped_true, grouped_pred, labels=labels_sorted)
    print("Confusion matrix (rows = gold, cols = predicted):")
    width = max(len(c) for c in labels_sorted) + 2
    print(" " * width + "".join(f"{c[:10]:>12s}" for c in labels_sorted))
    for name, row in zip(labels_sorted, cm):
        print(f"{name:<{width}}" + "".join(f"{v:>12d}" for v in row))

    # ---- fit the shipping model on everything -----------------------------
    pipe = build_pipeline()
    pipe.fit(X, y)
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    vocabulary = {term: int(i) for term, i in vec.vocabulary_.items()}
    assert_vocabulary_is_clean(vocabulary)
    print(f"\nvocabulary: {len(vocabulary)} features -- secret-leak check PASSED")

    artifact = {
        "classes": [str(c) for c in clf.classes_],
        "vocabulary": vocabulary,
        "idf": [float(v) for v in vec.idf_],
        "coef": [[float(v) for v in row] for row in clf.coef_],
        "intercept": [float(v) for v in clf.intercept_],
        "ngram_max": 2,
        "sublinear_tf": True,
        "metadata": {
            "trained_on": "masked message text only",
            "n_training_messages": len(X),
            "n_templates": len(set(groups)),
            "label_source": "weak supervision from mint/rules.py",
            "grouped_cv_accuracy": round(grouped_acc, 4),
            "random_cv_accuracy": round(random_acc, 4),
            "model_categories": T.MODEL_CATEGORIES,
            "note": (
                "Contains no message text -- only tokenised n-gram keys, IDF "
                "weights and learned coefficients."
            ),
        },
    }

    dest = ROOT / "models" / "classifier.json"
    dest.parent.mkdir(exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=1, sort_keys=True)
    size_kb = dest.stat().st_size / 1024
    print(f"wrote {dest} ({size_kb:.0f} KB)")

    # ---- verify the pure-Python runtime reproduces sklearn ----------------
    from mint.model import TfidfLogisticModel

    runtime = TfidfLogisticModel(artifact)
    sk_probs = pipe.predict_proba(X)
    max_delta = 0.0
    for text, sk_row in zip(X, sk_probs):
        pure = dict(runtime.predict(text).distribution)
        for cls, p in zip(clf.classes_, sk_row):
            max_delta = max(max_delta, abs(pure[str(cls)] - float(p)))
    print(f"pure-Python vs scikit-learn: max probability delta = {max_delta:.2e}")
    if max_delta > 1e-3:
        raise SystemExit("ABORT: pure-Python inference diverges from scikit-learn")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
