"""Speech acts: what a message does to the subject it names."""

from __future__ import annotations

import pytest

from mint import acts as A


@pytest.mark.parametrize("text,act,subject", [
    ("Can you share an update on review the privacy checklist?",
     A.STATUS_QUERY, "review the privacy checklist"),
    ("Following up on pay the electricity bill; is it in progress?",
     A.STATUS_QUERY, "pay the electricity bill"),
    ("Please confirm whether you started to upload the assignment.",
     A.STATUS_QUERY, "upload the assignment"),
    ("Any progress on the item concerning the assignment?",
     A.STATUS_QUERY, "assignment"),
    ("Update: finish the test cases has been completed successfully.",
     A.COMPLETION, "finish the test cases"),
    ("Confirmed: email the signed document has been completed.",
     A.COMPLETION, "email the signed document"),
    ("You can cancel call the service centre; it is no longer required.",
     A.CANCELLATION, "call the service centre"),
    ("The team stand-up has been cancelled.", A.CANCELLATION, "team stand-up"),
    ("New task: compare two embedding models by 2026-10-01.",
     A.NEW_TASK, "compare two embedding models"),
    ("This is another status request about complete the onboarding form, "
     "not a new task.", A.STATUS_QUERY, "complete the onboarding form"),
])
def test_frames_capture_act_and_subject(text, act, subject):
    v = A.detect(text)
    assert v.act == act
    assert v.subject_phrase == subject


def test_reschedule_captures_the_new_slot():
    v = A.detect("The family dinner has been moved to 2026-09-29 at 10:00. "
                 "Please use the new schedule.")
    assert v.act == A.RESCHEDULE
    assert (v.new_date, v.new_time) == ("2026-09-29", "10:00")


def test_a_time_only_change_asserts_no_date():
    """The date must stay absent so the group keeps the one already on record."""
    v = A.detect("The date for internship orientation stays the same, but the "
                 "time is now 17:30.")
    assert v.act == A.RESCHEDULE
    assert v.new_date is None and v.new_time == "17:30"


def test_relative_deadlines_are_kept_raw_and_never_resolved():
    v = A.detect("The deadline to confirm the interview slot is now tomorrow "
                 "at 10 AM. This is urgent.")
    assert v.act == A.DEADLINE_CHANGE
    assert v.new_date is None, "a relative phrase must not become a date"
    assert v.new_date_raw == "tomorrow"
    assert v.new_time == "10:00"
    assert v.urgent


def test_a_hedge_downgrades_a_firm_act():
    v = A.detect("Confirm the interview slot might already be finished, but I "
                 "cannot confirm it.")
    assert v.hedged and v.act == A.AMBIGUOUS_UPDATE


def test_mutually_exclusive_options_are_captured_not_chosen():
    v = A.detect("The deadline may be Monday, or it may be Wednesday. Wait for "
                 "the official update.")
    assert v.act == A.AMBIGUOUS_UPDATE
    assert v.alternatives == ["Monday", "Wednesday"]
    assert v.subject_phrase is None, "no subject is named, so none may be invented"


def test_de_escalation_is_not_urgency():
    """'no longer urgent' contains 'urgent'; only one of them may fire."""
    v = A.detect("This may no longer be urgent.")
    assert v.de_escalated and not v.urgent


def test_conflict_markers_are_detected():
    v = A.detect("Please note that confirm the interview slot is due on "
                 "2026-09-28, although the earlier message listed another date.")
    assert v.act == A.DEADLINE_CHANGE and v.conflict_marker


@pytest.mark.parametrize("prefix", ["Follow-up: ", "Additional update: ",
                                    "Follow-up: Additional update: "])
def test_restatement_prefixes_are_recorded_not_ignored(prefix):
    base = "update: prepare the demo video has been completed successfully."
    v = A.detect(prefix + base)
    assert v.act == A.COMPLETION and v.repetition
    assert v.subject_phrase == A.detect(base).subject_phrase


def test_messages_that_act_on_nothing_are_informational():
    for text in ("The office cafeteria menu has been updated.",
                 "A new wallpaper is available in the employee portal."):
        assert A.detect(text).act == A.INFORMATIONAL


def test_an_unmatched_question_is_an_open_question():
    v = A.detect("Was the compliance form approved by the finance director?")
    assert v.act == A.OPEN_QUESTION


def test_l1_extraction_promotes_a_request_and_drops_its_subject_phrase():
    """The fallback phrase is the whole sentence; L1's title is better."""
    raw = A.detect("Can you review the privacy checklist before 2026-09-09?")
    promoted = A.from_l1_extraction(raw, "task")
    assert promoted.act == A.NEW_TASK
    assert promoted.frame == "l1_extraction"
    assert promoted.subject_phrase is None


def test_l1_extraction_never_overwrites_a_real_l2_act():
    raw = A.detect("Update: finish the test cases has been completed successfully.")
    assert A.from_l1_extraction(raw, "task").act == A.COMPLETION
