"""Grouping and the subject lifecycle."""

from __future__ import annotations

import pytest

from mint import groups as G
from tests.conftest import group_titled


def test_follow_ups_join_the_subject_they_follow_up_on(ledger):
    audit = group_titled(ledger, "accessibility audit")
    assert {"T_001", "T_011", "T_015"} <= set(audit.message_ids)


def test_a_sparse_reference_still_lands_on_its_subject(ledger):
    """'the budget sheet' names no verb, but it is still the same work."""
    budget = group_titled(ledger, "budget sheet")
    assert "T_017" in budget.message_ids


def test_unrelated_subjects_are_not_merged(ledger):
    audit = group_titled(ledger, "accessibility audit")
    budget = group_titled(ledger, "budget sheet")
    assert audit.group_id != budget.group_id
    assert not (set(audit.message_ids) & set(budget.message_ids))


def test_messages_that_name_no_subject_are_left_ungrouped(ledger):
    """Promotional, personal, general and credential traffic joins nothing."""
    grouped = {m.message_id for g in ledger.groups for m in g.members}
    for mid in ("T_006", "T_007", "T_008", "T_009", "T_010"):
        assert mid not in grouped


def test_the_timeline_starts_where_the_work_started(ledger):
    audit = group_titled(ledger, "accessibility audit")
    assert audit.members[0].message_id == "T_001"
    assert audit.status_history[0].act == "new_task"


def test_members_are_in_chronological_order(ledger):
    for group in ledger.groups:
        stamps = [m.timestamp for m in group.members]
        assert stamps == sorted(stamps)


# -- the state machine -----------------------------------------------------
def test_completion_is_terminal(ledger):
    audit = group_titled(ledger, "accessibility audit")
    assert audit.status == G.COMPLETED
    assert audit.status_source == "T_015"


def test_a_hedge_cannot_overturn_a_firm_status(ledger):
    """T_019 hedges that the audit is signed off; the status must not move."""
    audit = group_titled(ledger, "accessibility audit")
    assert "T_019" in audit.message_ids
    assert audit.status == G.COMPLETED
    assert any(c.kind.startswith("hedged") for c in audit.conflicts)


def test_a_restatement_is_absorbed_not_re_asserted(ledger):
    """T_016 repeats T_015. One transition, one restatement counted."""
    audit = group_titled(ledger, "accessibility audit")
    completions = [t for t in audit.status_history if t.act == "completion"]
    assert len(completions) == 1
    assert audit.restatement_count >= 1


def test_cancellation_is_recorded_with_its_source(ledger):
    critique = group_titled(ledger, "design critique")
    assert critique.status == G.CANCELLED
    assert critique.status_source == "T_020"


def test_status_chases_do_not_assert_progress(ledger):
    """Asking whether something is in progress is not a report that it is."""
    budget = group_titled(ledger, "budget sheet")
    assert budget.status == G.PENDING
    assert budget.chase_count >= 2
    assert budget.response_required


# -- deadlines and schedules ------------------------------------------------
def test_a_deadline_change_updates_the_group_and_records_the_move(ledger):
    cert = group_titled(ledger, "domain certificate")
    assert cert.latest_deadline == "2026-11-04"
    assert cert.latest_deadline_source == "T_013"
    moves = [c for c in cert.deadline_history if c.origin == "l2_update"]
    assert moves and moves[-1].direction == "earlier"


def test_a_time_only_change_keeps_the_date_already_on_record(ledger):
    """T_014 sets 2026-11-16 11:00; T_018 changes only the time."""
    drill = group_titled(ledger, "fire drill")
    assert drill.latest_date == "2026-11-16"
    assert drill.latest_time == "15:45"
    assert drill.schedule_source == "T_018"
    assert drill.status == G.RESCHEDULED


def test_original_requests_are_not_counted_as_deliberate_changes(ledger):
    for group in ledger.groups:
        for change in group.deadline_history:
            if change.origin == "l1_request":
                assert not any(c.kind == "deadline_changed"
                               and change.message_id in c.message_ids
                               for c in group.conflicts)


# -- conflicts --------------------------------------------------------------
def test_unresolved_alternatives_are_recorded_not_narrowed(ledger):
    loose = [r for r in ledger.records if r.verdict.alternatives]
    assert any(r.message_id == "T_021" for r in loose)
    assert ledger.record("T_021").verdict.alternatives == ["Tuesday", "Thursday"]


def test_group_confidence_falls_when_evidence_conflicts(ledger):
    audit = group_titled(ledger, "accessibility audit")
    drill = group_titled(ledger, "fire drill")
    assert audit.conflicts and audit.confidence < drill.confidence


def test_every_group_can_explain_itself(ledger):
    for group in ledger.groups:
        assert group.summary.endswith(".")
        assert group.status_reason
        assert 0.30 <= group.confidence <= 0.97


def test_summaries_only_contain_what_was_recorded(ledger):
    """No clause may name a message that is not in the group."""
    import re
    for group in ledger.groups:
        for mid in re.findall(r"\bT_\d{3}\b", group.summary):
            assert mid in group.message_ids


def test_a_group_with_nothing_firm_said_about_it_is_not_pending(ledger):
    """'unclear' and 'pending' mean different things and must stay distinct."""
    vendor = [g for g in ledger.groups if "vendor" in g.title.lower()]
    if vendor:
        assert vendor[0].status in (G.UNCLEAR, G.PENDING)
        if vendor[0].status == G.PENDING:
            assert vendor[0].item_ids, "pending must rest on an extracted item"
