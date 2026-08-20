"""Privacy-aware routing, and the properties that make it more than a label."""

from __future__ import annotations

import pytest

from mint import routing as R
from mint.sensitive import scan


@pytest.mark.parametrize("text,route,policy", [
    ("Your OTP is 552317. It expires in five minutes.", R.BLOCKED, "P1-CREDENTIAL"),
    ("Use password RiverStone#42 to sign in.", R.BLOCKED, "P1-CREDENTIAL"),
    ("Integration token: tok_test_9KQ22.", R.BLOCKED, "P1-CREDENTIAL"),
    ("The test card number is 5555 4444 3333 1111.", R.BLOCKED, "P1-CREDENTIAL"),
    ("My identification number is ID-4417-KK.", R.LOCAL_ONLY, "P2-IDENTITY"),
    ("My recent test result says low iron.", R.LOCAL_ONLY, "P2-IDENTITY"),
    ("You can contact me on 98765 43210.", R.CONFIRM_REQUIRED, "P3-CONTACT"),
    ("Deliver the spare laptop to 4 Hillside Road, Pune.",
     R.CONFIRM_REQUIRED, "P3-CONTACT"),
    ("The staff canteen now opens at eight.", R.LOCAL_ONLY, "P0-DEFAULT"),
])
def test_routes_follow_the_l1_recommendation(text, route, policy):
    decision = R.route_message("M1", scan(text))
    assert (decision.route, decision.policy_id) == (route, policy)


def test_the_strictest_finding_sets_the_route():
    """A message with both a phone number and an OTP is blocked, not gated."""
    text = "Your OTP is 552317 and you can contact me on 98765 43210."
    decision = R.route_message("M1", scan(text))
    assert decision.route == R.BLOCKED


def test_blocked_messages_are_never_indexable_or_quotable():
    decision = R.route_message("M1", scan("Your OTP is 552317."))
    assert not decision.indexable and not decision.quotable
    assert not decision.exportable
    assert R.INDEX in decision.denied_operations


def test_external_sending_is_denied_for_every_message_including_harmless_ones():
    for text in ("The staff canteen now opens at eight.", "Your OTP is 552317."):
        decision = R.route_message("M1", scan(text))
        assert R.SEND_EXTERNAL in decision.denied_operations


def test_confirmation_gated_messages_stay_searchable_in_masked_form():
    decision = R.route_message("M1", scan("You can contact me on 98765 43210."))
    assert decision.indexable, "refusing to index contact details would be theatre"
    assert not decision.quotable
    assert decision.confirmation_prompt


def test_an_answer_inherits_the_strictest_route_of_its_evidence():
    decisions = {
        "A": R.route_message("A", scan("The canteen opens at eight.")),
        "B": R.route_message("B", scan("You can contact me on 98765 43210.")),
        "C": R.route_message("C", scan("Your OTP is 552317.")),
    }
    assert R.route_answer(["A"], decisions).route == R.LOCAL_ONLY
    assert R.route_answer(["A", "B"], decisions).route == R.CONFIRM_REQUIRED
    assert R.route_answer(["A", "B", "C"], decisions).route == R.BLOCKED


def test_confirmation_unlocks_only_what_it_was_given_for():
    decisions = {"B": R.route_message("B", scan("Contact me on 98765 43210.")),
                 "C": R.route_message("C", scan("Your OTP is 552317."))}
    assert R.route_answer(["B"], decisions, confirmed=True).route == R.LOCAL_ONLY
    assert R.route_answer(["C"], decisions, confirmed=True).route == R.BLOCKED


def test_blocked_evidence_is_reported_as_existing_not_hidden():
    decisions = {"C": R.route_message("C", scan("Your OTP is 552317."))}
    verdict = R.route_answer(["C"], decisions)
    assert verdict.withheld_message_ids == ["C"]
    assert "withheld" in verdict.rationale


def test_the_policy_table_covers_every_sensitivity_type():
    from mint.sensitive import RECOMMENDED_ACTION
    covered = {t for row in R.policy_table() for t in row["sensitivity_types"]}
    assert covered == set(RECOMMENDED_ACTION)


# -- the structural property, end to end -----------------------------------
def test_blocked_messages_never_enter_the_search_index(ledger):
    blocked = {r.message_id for r in ledger.records if r.route.route == R.BLOCKED}
    assert blocked, "the fixture must contain something blocked"
    assert blocked.isdisjoint(set(ledger.index.doc_ids))
    assert set(ledger.index.excluded) == blocked


def test_no_query_can_surface_a_blocked_message(ledger):
    """Not filtered out of results -- never a candidate in the first place."""
    blocked = {r.message_id for r in ledger.records if r.route.route == R.BLOCKED}
    for query in ("OTP", "token", "password", "integration token local test",
                  "552317", "expires in five minutes"):
        hits = {h.doc_id for h in ledger.index.search(query, 25)}
        assert hits.isdisjoint(blocked)


def test_the_routing_output_file_withholds_blocked_content(ledger):
    for row in ledger.routing_records():
        if row["route"] == R.BLOCKED:
            assert row["masked_text"] is None
            assert not row["indexed"]
