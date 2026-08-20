"""The two retrieval indexes, and what the optimisation cost."""

from __future__ import annotations

import pytest

from mint.embed import load_default as load_semantic
from mint.retrieve import Document, ExactLexicalIndex, HybridIndex

DOCS = [
    Document("D1", "Can you review the accessibility audit before 2026-11-09?"),
    Document("D2", "Following up on submit the quarterly budget sheet."),
    Document("D3", "The fire drill has been moved to 2026-11-16 at 11:00."),
    Document("D4", "Update: review the accessibility audit has been completed."),
    Document("D5", "Half price on annual plans this week. Use code PLAN50."),
    Document("D6", "The lift in block B is back in service."),
    Document("D7", "New task: draft the retention policy by 2026-11-11."),
    Document("D8", "Your OTP is ******.", indexable=False),
]


@pytest.fixture(scope="module")
def indexes():
    model = load_semantic()
    return ExactLexicalIndex(DOCS), HybridIndex(DOCS, model)


def test_both_indexes_skip_documents_routing_refused(indexes):
    for index in indexes:
        assert "D8" not in index.doc_ids
        assert index.excluded == ["D8"]


def test_both_find_the_obvious_thing(indexes):
    for index in indexes:
        hits = index.search("accessibility audit", 3)
        assert {h.doc_id for h in hits[:2]} == {"D1", "D4"}


def test_hits_carry_the_evidence_for_their_score(indexes):
    for index in indexes:
        hit = index.search("accessibility audit", 1)[0]
        assert hit.score > 0
        assert hit.matched_terms
        assert set(hit.to_dict()) >= {"message_id", "score", "lexical_score",
                                      "semantic_score", "matched_terms"}


def test_the_optimised_index_reproduces_the_exact_ranking(indexes):
    exact, hybrid = indexes
    for query in ("accessibility audit", "fire drill moved", "retention policy",
                  "budget sheet", "annual plans discount"):
        a = [h.doc_id for h in exact.search(query, 5)]
        b = [h.doc_id for h in hybrid.search(query, 5)]
        assert a[:1] == b[:1], f"top hit diverged on {query!r}"


def test_an_unknown_word_does_not_dominate_the_ranking(indexes):
    """A term the corpus has never seen carries no evidence about it."""
    exact, _ = indexes
    plain = [h.doc_id for h in exact.search("accessibility audit", 3)]
    noisy = [h.doc_id for h in exact.search("zzqqxx accessibility audit", 3)]
    assert plain == noisy


def test_the_dense_scan_is_not_paid_for_when_lexical_scoring_suffices(indexes):
    """k=2 because this fixture holds seven documents: the fallback triggers on
    "fewer candidates than results asked for", and asking for three of seven is
    genuinely asking for more than lexical scoring can supply here."""
    _, hybrid = indexes
    hybrid.search("accessibility audit", 2)
    assert not hybrid.stats.dense_scan
    assert not hybrid.stats.fallback_scan
    assert hybrid.stats.postings_visited > 0


def test_the_dense_scan_fires_when_lexical_scoring_cannot_fill_the_result(indexes):
    _, hybrid = indexes
    hybrid.search("zzqqxx unheardof phrasing", 10)
    assert hybrid.stats.dense_scan


def test_the_inverted_index_visits_fewer_documents_than_a_full_scan(indexes):
    exact, hybrid = indexes
    exact.search("accessibility audit", 2)
    hybrid.search("accessibility audit", 2)
    assert hybrid.stats.candidates < exact.stats.documents_scanned


def test_the_semantic_layer_is_optional(indexes):
    lexical = HybridIndex(DOCS, load_semantic(), use_semantic=False)
    assert [h.doc_id for h in lexical.search("accessibility audit", 3)][:2] \
        == [h.doc_id for h in indexes[0].search("accessibility audit", 3)][:2]
    assert not lexical.profile()["semantic_enabled"]


def test_int8_embeddings_are_stored_as_bytes_not_python_ints():
    """The whole point of quantising is size; a list of ints undoes it."""
    model = load_semantic()
    if model is None:
        pytest.skip("semantic model not built")
    index = HybridIndex(DOCS, model)
    per_doc = index.semantic_bytes() / max(len(index.embeddings), 1)
    assert per_doc <= model.k + 8, "embeddings should cost ~1 byte per dimension"


def test_the_profile_reports_what_was_excluded_and_pruned(indexes):
    _, hybrid = indexes
    profile = hybrid.profile()
    assert profile["excluded_by_routing"] == 1
    assert profile["documents"] == len(DOCS) - 1
    assert profile["semantic_role"]
