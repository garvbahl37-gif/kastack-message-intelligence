"""The assistant: routing to intents, citing evidence, and refusing."""

from __future__ import annotations

import pytest

from mint.assistant import Assistant, route_intent
from tests.conftest import group_titled


@pytest.fixture(scope="module")
def bot(ledger):
    return Assistant(ledger)


@pytest.mark.parametrize("query,intent", [
    ("Which demo messages must be blocked from external processing?",
     "blocked_messages"),
    ("Which message requires confirmation before processing?",
     "needs_confirmation"),
    ("Which existing task became critical in the demo data?", "became_priority"),
    ("Why was this item marked as critical?", "explain_priority"),
    ("What deadlines have changed?", "deadline_changes"),
    ("Are there any conflicting messages about the same event?", "conflicts"),
    ("What meetings were rescheduled?", "rescheduled"),
    ("Which tasks have been completed?", "by_status"),
    ("What tasks should I complete today?", "due_today"),
    ("Which critical or high-priority tasks are still pending?",
     "pending_by_priority"),
    ("What is the latest status of the budget sheet?", "latest_status"),
    ("Show all messages related to the fire drill.", "messages_about"),
    ("Was the vendor contract countersigned by the legal team?",
     "yes_no_verification"),
])
def test_questions_reach_the_right_intent(query, intent):
    assert route_intent(query).name == intent


def test_a_verb_is_not_a_status(bot):
    """'What tasks should I complete today?' asks what is outstanding."""
    answer = bot.answer("What tasks should I complete today?")
    assert answer.intent == "due_today"


def test_every_answer_carries_evidence_scores_and_a_selection_reason(bot):
    for query in ("Which tasks have been completed?",
                  "What deadlines have changed?",
                  "Which critical or high-priority tasks are still pending?"):
        answer = bot.answer(query).to_dict()
        assert answer["supporting_message_ids"]
        assert answer["reason"]
        assert answer["privacy_route"]
        for item in answer["evidence"]:
            assert item["why_selected"]
            assert item["relevance_score"] >= 0
        for hit in answer["retrieved"]:
            assert 0 <= hit["score"] <= 2


def test_structured_answers_are_complete_sets_not_top_matches(bot, ledger):
    """A group answer must contain every message in the group."""
    drill = group_titled(ledger, "fire drill")
    answer = bot.answer("Show all messages related to the fire drill.")
    assert set(answer.supporting_message_ids) == set(drill.message_ids)


def test_the_rescheduled_answer_merges_changes_rather_than_echoing_the_last(bot):
    answer = bot.answer("Which meeting was rescheduled and what is its latest "
                        "schedule?")
    assert "2026-11-16" in answer.answer, "the date came from the earlier message"
    assert "15:45" in answer.answer, "the time came from the later one"


def test_completed_and_cancelled_are_both_reported(bot):
    answer = bot.answer("Which tasks or meetings were completed or cancelled?")
    assert "accessibility audit" in answer.answer.lower()
    assert "design critique" in answer.answer.lower()


def test_a_hedge_is_reported_rather_than_resolved(bot):
    answer = bot.answer("What is the latest status of the accessibility audit?")
    assert answer.intent == "latest_status"
    assert "hedge" in answer.answer.lower()


def test_blocked_content_is_named_but_never_quoted(bot, ledger):
    answer = bot.answer("Which messages must be blocked from external "
                        "processing?")
    assert answer.supporting_message_ids
    for item in answer.evidence:
        assert item.withheld
        assert item.masked_text is None
    assert answer.route.route == "blocked"


def test_confirmation_gated_answers_say_so(bot):
    answer = bot.answer("Which messages require confirmation?")
    assert answer.route.route == "confirm_required"
    assert answer.notes


def test_confirming_unlocks_the_gated_answer(bot):
    answer = bot.answer("Which messages require confirmation?", confirmed=True)
    assert answer.route.route == "local_only"


# -- refusal ---------------------------------------------------------------
def test_a_question_nothing_answers_is_refused(bot):
    """The corpus contains the same question; similarity is not an answer."""
    answer = bot.answer("Was the vendor contract countersigned by the legal team?")
    assert not answer.sufficient
    assert "not enough evidence" in answer.answer.lower()
    assert "T_022" in answer.supporting_message_ids


def test_a_question_about_nothing_at_all_is_refused(bot):
    answer = bot.answer("Did the warehouse inspection pass in Reykjavik?")
    assert not answer.sufficient


def test_refusal_still_explains_itself(bot):
    answer = bot.answer("Was the vendor contract countersigned by the legal team?")
    assert answer.reason
    assert answer.confidence > 0


def test_a_yes_no_question_with_an_assertion_behind_it_is_answered(bot):
    answer = bot.answer("Has the accessibility audit been completed?")
    assert answer.sufficient
    assert "completed" in answer.answer.lower()


def test_a_weak_single_word_resolution_is_declared(ledger):
    """Resolving on one shared term is the weakest link the matcher allows."""
    bot = Assistant(ledger)
    answer = bot.answer("Show all messages related to the audit.")
    if answer.group_ids:
        assert "single shared term" in answer.answer or answer.confidence < 0.95


def test_no_answer_invents_a_message_id(bot, ledger):
    known = {r.message_id for r in ledger.records}
    for query in ("Which tasks have been completed?", "What deadlines have changed?",
                  "Which meeting was rescheduled and what is its latest schedule?",
                  "Which critical or high-priority tasks are still pending?",
                  "Are there any conflicting messages about the same event?"):
        answer = bot.answer(query)
        assert set(answer.supporting_message_ids) <= known
