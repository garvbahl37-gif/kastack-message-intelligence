"""The ledger: ordering, snapshots, output files and the L1 relationship."""

from __future__ import annotations

import json

import pytest

from mint import groups as G
from mint import ledger as L
from mint import priority as P
from mint import routing as R
from mint import taxonomy as T
from tests.conftest import BATCH_ONE, BATCH_TWO, group_titled


def test_batches_are_processed_in_the_order_given(ledger):
    names = [b.name for b in ledger.batches]
    assert names == ["batch_one.csv", "batch_two.csv"]
    order = [r.message_id for r in ledger.records]
    assert order == sorted(order), "this fixture is already chronological"


def test_batch_order_beats_timestamp_order():
    """A later batch is processed after an earlier one even if it is older."""
    older = "message_id,timestamp,sender,message\nZ_1,2020-01-01 00:00:00,A,Hi.\n"
    ledger = L.run([("first.csv", BATCH_ONE), ("second.csv", older)])
    assert ledger.records[-1].message_id == "Z_1"
    assert ledger.as_of == "2020-01-01 00:00:00"


def test_l1_still_runs_first_and_its_results_are_kept(ledger):
    for record in ledger.records:
        assert record.classification.category in T.CATEGORIES
        assert record.classification.reason
    assert any(r.item and r.item.type == "task" for r in ledger.records)
    assert any(r.item and r.item.type == "event" for r in ledger.records)


def test_the_record_has_nowhere_to_put_raw_text(ledger):
    assert not hasattr(ledger.records[0], "raw")
    assert "raw" not in ledger.records[0].to_dict()


def test_nothing_is_extracted_from_a_message_routing_blocked(ledger):
    for record in ledger.records:
        if record.route.route == R.BLOCKED:
            assert record.item is None


def test_a_snapshot_is_taken_at_every_batch_boundary(ledger):
    assert len(ledger.snapshots) == len(ledger.batches)
    for snapshot, batch in zip(ledger.snapshots, ledger.batches):
        assert snapshot.as_of == batch.last_seen


def test_priority_history_only_grows_forward(ledger):
    for changes in ledger.priority_history.values():
        stamps = [c.as_of for c in changes]
        assert stamps == sorted(stamps)


def test_the_reference_time_advances_with_each_batch(one_batch, ledger):
    assert one_batch.as_of < ledger.as_of


def test_an_ungrouped_actionable_item_still_gets_a_priority(ledger):
    """A subject with no identity is scored alone rather than forced into a
    group it does not belong in."""
    rows = [r for r in ledger.priority_records() if r["group_id"] is None]
    for row in rows:
        assert row["priority"] in P.BANDS
        assert row["needs_review"], "a lone subject is weaker evidence"


def test_items_from_one_subject_share_its_priority(ledger):
    by_group = {}
    for row in ledger.priority_records():
        if row["group_id"]:
            by_group.setdefault(row["group_id"], set()).add(row["priority"])
    for group_id, bands in by_group.items():
        assert len(bands) == 1, f"{group_id} gave its items different bands"


def test_lookups_resolve(ledger):
    record = ledger.records[0]
    assert ledger.record(record.message_id) is record
    group = ledger.groups[0]
    assert ledger.group(group.group_id) is group
    assert ledger.group_of(group.members[0].message_id) is group


def test_subject_resolution_uses_the_grouper_s_own_matcher(ledger):
    match = ledger.resolve_subject("the budget sheet")
    assert match is not None
    assert match[0] is group_titled(ledger, "budget sheet")


# -- outputs ---------------------------------------------------------------
@pytest.fixture(scope="module")
def written(ledger, tmp_path_factory):
    out = tmp_path_factory.mktemp("outputs")
    paths = L.write_outputs(ledger, out, answers=[{"query": "q", "answer": "a"}])
    return {p.name: p for p in paths}


def test_all_required_output_files_are_written(written):
    for name in ("priority_decisions.json", "message_groups.json",
                 "privacy_routing.json", "priority_snapshots.json",
                 "l2_summary.json", "message_groups.csv",
                 "priority_decisions.csv", "assistant_answers.json"):
        assert name in written, f"{name} missing"
        assert written[name].stat().st_size > 0


def test_the_priority_file_matches_the_brief_s_shape(written):
    rows = json.loads(written["priority_decisions.json"].read_text())
    required = {"message_id", "item_id", "priority", "reason", "signals",
                "confidence"}
    for row in rows:
        assert required <= set(row)


def test_the_group_file_matches_the_brief_s_shape(written):
    rows = json.loads(written["message_groups.json"].read_text())
    required = {"group_id", "title", "related_message_ids", "related_item_ids",
                "status", "latest_deadline", "summary", "confidence"}
    for row in rows:
        assert required <= set(row)
        assert row["status"] in G.STATUSES


def test_the_routing_file_carries_the_policy_it_applied(written):
    payload = json.loads(written["privacy_routing.json"].read_text())
    assert payload["policy_table"]
    for row in payload["decisions"]:
        assert row["route"] in R.ROUTES
        assert row["policy_id"]
        assert row["rationale"]


def test_the_summary_reports_what_was_withheld(written):
    summary = json.loads(written["l2_summary.json"].read_text())
    assert summary["excluded_from_index"] == summary["route_counts"]["blocked"]
    assert summary["as_of"]
