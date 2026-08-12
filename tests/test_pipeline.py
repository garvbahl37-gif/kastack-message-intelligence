"""End-to-end pipeline behaviour, plus the model/rule contract."""

from __future__ import annotations

import io

import pytest

from mint import pipeline, rules
from mint import taxonomy as T
from mint.model import load_default

CSV = """message_id,timestamp,sender,message
MSG_003,2026-09-03 10:00:00,Neha,Please submit the weekly report by 2026-09-05.
MSG_001,2026-09-01 08:00:00,Meera,Calendar update: family dinner, 2026-09-19 at 10:00, the library.
MSG_002,2026-09-02 09:00:00,Promotions,Get 25% off selected headphones. Use code SAVE30.
MSG_005,2026-09-05 12:00:00,Riya,Your OTP is 884422. It expires in 10 minutes.
MSG_004,2026-09-04 11:00:00,Tara,Remember that I prefer morning meetings.
MSG_006,2026-09-06 13:00:00,Kabir,The cafeteria closes at 6 PM.
"""


@pytest.fixture(scope="module")
def result():
    return pipeline.run(io.StringIO(CSV))


def test_messages_are_processed_in_chronological_order(result):
    stamps = [m.timestamp for m in result.messages]
    assert stamps == sorted(stamps)
    assert [m.message_id for m in result.messages] == [
        "MSG_001", "MSG_002", "MSG_003", "MSG_004", "MSG_005", "MSG_006",
    ]


def test_all_six_categories_are_reachable(result):
    got = {m.classification.category for m in result.messages}
    assert got == {
        T.MEETING_EVENT, T.PROMOTIONAL, T.ACTION_REQUIRED,
        T.PERSONAL_INFORMATION, T.SENSITIVE_INFORMATION, T.GENERAL_INFORMATION,
    }


def test_every_classification_has_a_reason_and_bounded_confidence(result):
    for m in result.messages:
        c = m.classification
        assert c.reason.strip(), f"{m.message_id} has no reason"
        assert 0.0 < c.confidence <= 0.99, f"{m.message_id}: {c.confidence}"
        assert c.category in T.CATEGORIES


def test_sensitive_message_keeps_its_topical_reading_as_secondary(result):
    otp = next(m for m in result.messages if m.message_id == "MSG_005")
    assert otp.classification.category == T.SENSITIVE_INFORMATION
    assert otp.classification.secondary_category is not None
    assert otp.classification.agreement == "deterministic_detector"


def test_only_actionable_messages_produce_items(result):
    sources = {i.source_message_id for i in result.items}
    assert sources == {"MSG_001", "MSG_003"}
    assert {i.type for i in result.items} == {"event", "task"}


def test_summary_counts_are_internally_consistent(result):
    s = result.summary()
    assert s["total_messages"] == len(result.messages) == 6
    assert sum(s["categories"].values()) == 6
    assert s["tasks_extracted"] + s["events_extracted"] == len(result.items)
    assert s["sensitive_messages"] == len(result.sensitive_report())


def test_missing_columns_are_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="missing required column"):
        pipeline.run(io.StringIO("id,text\n1,hello\n"))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="no data rows"):
        pipeline.run(io.StringIO("message_id,timestamp,sender,message\n"))


def test_bom_prefixed_csv_is_handled():
    res = pipeline.run(io.StringIO("﻿" + CSV))
    assert len(res.messages) == 6


def test_csv_passed_as_a_plain_string_is_treated_as_content_not_a_path():
    """Regression: a str longer than NAME_MAX must never be probed as a path.

    The upload endpoint hands `run()` the decoded file body as a str. Asking
    the filesystem whether that body is a filename raises ENAMETOOLONG on Linux
    once it exceeds 255 bytes -- which is exactly what a real CSV does.
    """
    big = CSV + "".join(
        f"MSG_{i:03d},2026-09-{(i % 28) + 1:02d} 10:00:00,Neha,"
        f"Please submit report {i} by 2026-10-01.\n"
        for i in range(100, 200)
    )
    assert len(big) > 255
    res = pipeline.run(big)
    assert len(res.messages) == 106


def test_short_existing_path_as_a_string_still_loads_the_file(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text(CSV, encoding="utf-8")
    assert len(pipeline.run(str(p)).messages) == 6


# --- the rule/model contract ------------------------------------------------
def test_rules_abstain_rather_than_guess_on_unfamiliar_text():
    verdict = rules.classify("Zorble the quixotic frobnitz.")
    assert verdict.category is None
    assert verdict.confidence == 0.0


def test_weak_supervision_refuses_hedged_frames():
    assert rules.weak_label("The review could be Friday afternoon.") is None
    assert rules.weak_label("Please submit the weekly report by 2026-09-05.") == \
        T.ACTION_REQUIRED


def test_system_still_classifies_with_no_model_available():
    """The model is an enhancement, not a dependency.

    Passing model=None forces the degraded path a fresh checkout hits before
    `scripts/train.py` has ever been run.
    """
    from mint.classifier import classify as classify_message
    from mint.sensitive import scan

    text = "Please submit the weekly report by 2026-09-05."
    out = classify_message("MSG_X", text, "Neha", scan(text), model=None)
    assert out.category == T.ACTION_REQUIRED
    assert out.agreement == "rule_only"
    assert out.confidence > 0.5


@pytest.mark.skipif(load_default() is None, reason="model not trained yet")
def test_agreement_between_voices_raises_confidence_above_either_alone():
    from mint.classifier import classify as classify_message
    from mint.sensitive import scan

    text = "Please submit the weekly report by 2026-09-05."
    both = classify_message("MSG_X", text, "Neha", scan(text))
    assert both.agreement == "rule_and_model_agree"
    assert both.confidence > both.rule_confidence
    assert both.confidence > both.model_confidence


@pytest.mark.skipif(load_default() is None, reason="model not trained yet")
def test_disagreement_is_flagged_and_damped():
    from mint.classifier import classify as classify_message
    from mint.sensitive import scan

    text = "The report may be needed tomorrow."
    out = classify_message("MSG_X", text, "Neha", scan(text))
    assert out.agreement == "disagreement"
    assert out.needs_review
    assert out.confidence <= 0.72
