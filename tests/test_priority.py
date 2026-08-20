"""The priority engine: bands, signals, corroboration and updates."""

from __future__ import annotations

import pytest

from mint import groups as G
from mint import priority as P
from tests.conftest import group_titled


def test_every_signal_has_a_weight_a_family_and_an_explanation():
    for name, spec in P.SIGNALS.items():
        assert spec.name == name
        assert spec.family
        assert spec.explain and not spec.explain.endswith(".")


def test_every_decision_carries_the_brief_s_required_fields(ledger):
    for row in ledger.priority_records():
        assert row["message_id"] and row["item_id"]
        assert row["priority"] in P.BANDS
        assert row["reason"]
        assert isinstance(row["signals"], list) and row["signals"]
        assert 0.0 < row["confidence"] <= 1.0


def test_the_reason_is_assembled_from_signals_that_actually_fired(ledger):
    cert = group_titled(ledger, "domain certificate")
    decision = ledger.priorities[cert.group_id]
    for signal in decision.signals:
        if signal.weight > 0:
            continue
    top = max(decision.signals, key=lambda s: s.weight)
    assert top.detail in decision.reason


def test_an_urgent_overdue_chased_task_reaches_critical(ledger):
    cert = group_titled(ledger, "domain certificate")
    decision = ledger.priorities[cert.group_id]
    assert decision.priority == P.CRITICAL
    assert "explicit_urgency" in decision.signal_names


def test_closed_work_is_not_a_priority(ledger):
    for needle, signal in (("accessibility audit", "status_completed"),
                           ("design critique", "status_cancelled")):
        group = group_titled(ledger, needle)
        decision = ledger.priorities[group.group_id]
        assert decision.priority == P.LOW
        assert signal in decision.signal_names


def test_critical_requires_two_families_of_evidence():
    """One very loud signal must not on its own demand that a human drop
    everything -- that is the 'not from one keyword' rule, enforced."""
    # Overdue *and* restated as due tomorrow: two signals, both of them
    # deadline signals, and nobody has said it is urgent or chased it once.
    lone = G.MessageGroup(
        group_id="X", title="t", kind="task",
        signature=G.signature("a lone overdue thing"),
        status=G.PENDING, latest_deadline="2026-01-01",
        latest_deadline_source="M1", pending_relative_deadline="tomorrow",
        pending_relative_source="M2", confidence=0.9)
    decision = P.score_group(lone, "2026-06-01 00:00:00", category="task")
    assert decision.score >= P.THRESHOLDS[0][1]
    assert decision.priority == P.HIGH
    assert decision.gated_from_critical
    assert "corroboration" in decision.reason


def test_priority_updates_when_a_later_message_changes_the_deadline(
        ledger, one_batch):
    """The same subject, before and after the batch that escalates it."""
    before = one_batch.priorities[group_titled(one_batch, "domain certificate").group_id]
    after = ledger.priorities[group_titled(ledger, "domain certificate").group_id]
    assert after.score > before.score
    assert after.priority == P.CRITICAL


def test_a_change_is_attributed_to_a_message_or_to_elapsed_time(ledger):
    for changes in ledger.priority_history.values():
        for change in changes:
            assert change.trigger in ("initial", "message", "elapsed_time")
            if change.trigger == "message":
                assert change.trigger_message_ids
            elif change.trigger == "elapsed_time":
                assert not change.trigger_message_ids


def test_score_movement_inside_a_band_is_still_recorded():
    early = P.PriorityDecision(P.CRITICAL, 4.7, 0.8, "", as_of="t1")
    late = P.PriorityDecision(P.CRITICAL, 9.3, 0.8, "", as_of="t2")
    change = P.track(early, late, ["M9"])
    assert change is not None and change.kind == "escalation"
    assert change.delta == pytest.approx(4.6)


def test_noise_below_the_movement_threshold_is_not_reported():
    a = P.PriorityDecision(P.HIGH, 3.1, 0.8, "", as_of="t1")
    b = P.PriorityDecision(P.HIGH, 3.4, 0.8, "", as_of="t2")
    assert P.track(a, b, ["M9"]) is None


def test_confidence_falls_when_the_deadline_was_never_resolved():
    base = dict(kind="task", signature=G.signature("some subject"),
                status=G.PENDING, confidence=0.9)
    firm = G.MessageGroup(group_id="A", title="t", latest_deadline="2026-06-05",
                          latest_deadline_source="M1", **base)
    vague = G.MessageGroup(group_id="B", title="t",
                           pending_relative_deadline="tomorrow",
                           pending_relative_source="M2", **base)
    now = "2026-06-04 00:00:00"
    assert (P.score_group(vague, now).confidence
            < P.score_group(firm, now).confidence)


def test_reference_time_is_the_newest_message_not_the_clock(ledger):
    """The same input must always produce the same output."""
    assert ledger.as_of == ledger.batches[-1].last_seen
    for decision in ledger.priorities.values():
        assert decision.as_of == ledger.as_of
