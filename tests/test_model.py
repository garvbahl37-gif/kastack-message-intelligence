"""The pure-Python runtime must reproduce scikit-learn exactly.

This is the test that lets the deployed service drop scikit-learn entirely. If
it ever fails, the exported artifact and the training pipeline have diverged
and the deployment is silently serving a different model than the one that was
evaluated.
"""

from __future__ import annotations

import math

import pytest

from mint.model import TfidfLogisticModel, load_default

sklearn = pytest.importorskip("sklearn", reason="scikit-learn is a dev-only dep")

SAMPLES = [
    "Please submit the weekly report by 2026-09-05.",
    "Calendar update: family dinner, 2026-09-19 at 10:00, the library.",
    "Get 25% off selected headphones this weekend. Use code SAVE30.",
    "Remember that I prefer morning meetings.",
    "The cafeteria closes at 6 PM.",
    "Can you review the privacy checklist before 2026-09-09?",
    "Let us meet sometime next week.",
    "A sentence the model has certainly never seen, with zorble in it.",
]


@pytest.fixture(scope="module")
def model():
    m = load_default()
    if m is None:
        pytest.skip("models/classifier.json not present; run scripts/train.py")
    return m


def test_artifact_contains_no_message_text(model):
    """Vocabulary entries are n-gram keys, not sentences."""
    for term in model.vocabulary:
        assert len(term.split()) <= model.ngram_max


def test_artifact_vocabulary_has_no_secret_shapes(model):
    import re
    bad = re.compile(r"\d{4,}|(?:sk|pk|tok|ghp|gho)[_-]\w+")
    offenders = [t for t in model.vocabulary if bad.search(t)]
    assert not offenders, f"secret-shaped vocabulary entries: {offenders[:10]}"


@pytest.mark.parametrize("text", SAMPLES)
def test_probabilities_are_a_valid_distribution(model, text):
    pred = model.predict(text)
    total = sum(p for _, p in pred.distribution)
    assert math.isclose(total, 1.0, abs_tol=1e-3)
    assert all(0.0 <= p <= 1.0 for _, p in pred.distribution)
    assert pred.label == pred.distribution[0][0]


def test_unknown_tokens_do_not_crash_and_stay_uncertain(model):
    pred = model.predict("zzzz qqqq xxxx")
    assert pred.label in model.classes
    assert pred.confidence < 0.9


def test_explanations_are_real_linear_contributions(model):
    text = "Please submit the weekly report by 2026-09-05."
    feats = model.explain(text, top_k=5)
    assert feats
    # Every reported feature must actually be in the vocabulary...
    for name, _ in feats:
        assert name in model.vocabulary
    # ...and contributions must be sorted strongest-first.
    values = [v for _, v in feats]
    assert values == sorted(values, reverse=True)


def test_matches_sklearn_when_retrained_on_the_same_data():
    """Refit a small pipeline, export it, and compare the two forward passes."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from mint.normalize import tokenize

    texts = SAMPLES * 4
    labels = ["a", "b", "c", "d", "e", "a", "b", "c"] * 4

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, lowercase=False,
                                  ngram_range=(1, 2), sublinear_tf=True,
                                  norm="l2", smooth_idf=True, token_pattern=None)),
        ("clf", LogisticRegression(C=4.0, max_iter=2000, random_state=0)),
    ])
    pipe.fit(texts, labels)
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]

    runtime = TfidfLogisticModel({
        "classes": [str(c) for c in clf.classes_],
        "vocabulary": {t: int(i) for t, i in vec.vocabulary_.items()},
        "idf": [float(v) for v in vec.idf_],
        "coef": [[float(v) for v in row] for row in clf.coef_],
        "intercept": [float(v) for v in clf.intercept_],
        "ngram_max": 2,
        "sublinear_tf": True,
    })

    sk = pipe.predict_proba(SAMPLES)
    for text, row in zip(SAMPLES, sk):
        mine = dict(runtime.predict(text).distribution)
        for cls, p in zip(clf.classes_, row):
            assert abs(mine[str(cls)] - float(p)) < 1e-3, (
                f"divergence on {text!r} / {cls}"
            )
    assert np.allclose(sk.sum(axis=1), 1.0)
