"""Tests for Part 2 -- and above all for what the extractor refuses to do."""

from __future__ import annotations

import pytest

from mint import taxonomy as T
from mint.extract import extract

SENT = "2026-09-01 09:00:00"


def task(text, mid="MSG_X", sender="Ishaan", roster=("Maya", "Ishaan", "Tara")):
    return extract(mid, text, sender, SENT, T.ACTION_REQUIRED, 1, roster)


def event(text, mid="MSG_Y", sender="Meera", roster=("Maya", "Meera")):
    return extract(mid, text, sender, SENT, T.MEETING_EVENT, 1, roster)


# --- the core promise: nothing is invented ---------------------------------
def test_relative_date_is_not_resolved_into_the_date_field():
    item = task("The report may be needed tomorrow.")
    assert item.date is None
    assert item.date_raw == "tomorrow"
    assert item.date_status == "unresolved_relative"
    assert "date" in item.unresolved_fields


def test_relative_date_suggestion_is_offered_but_kept_separate():
    item = task("The report may be needed tomorrow.")
    # The suggestion exists so a human can act on it...
    assert item.date_suggestion == "2026-09-02"
    # ...but it is never promoted into the asserted field.
    assert item.date is None


def test_vague_time_reference_stays_unresolved():
    item = event("The review could be Friday afternoon.")
    assert item.date is None
    assert item.date_status == "unresolved_relative"
    assert item.date_raw.lower().startswith("friday")


def test_absent_date_is_missing_not_guessed():
    item = task("Could you send it soon?")
    assert item.date is None
    assert item.time is None
    assert item.person is None
    assert set(item.unresolved_fields) >= {"date", "time", "person"}


def test_person_is_null_when_nobody_is_named():
    item = task("Please submit the weekly report by 2026-09-05.")
    assert item.person is None
    assert item.person_status == "missing"


def test_sender_is_never_promoted_into_the_person_field():
    item = task("Please submit the weekly report by 2026-09-05.", sender="Ishaan")
    assert item.person is None
    assert item.source_sender == "Ishaan"


def test_venue_words_are_not_mistaken_for_people():
    item = event("The project review is scheduled for 2026-09-09 at 14:00 in Meeting Room A.")
    assert item.person is None
    assert item.location == "Meeting Room A"


# --- what it does extract ---------------------------------------------------
def test_task_with_deadline():
    item = task("Can you review the privacy checklist before 2026-09-09?")
    assert item.type == "task"
    assert item.title == "Review the privacy checklist"
    assert item.date == item.deadline == "2026-09-09"
    assert item.source_message_id == "MSG_X"


@pytest.mark.parametrize("text,title,date", [
    ("Please submit the weekly report by 2026-09-05.", "Submit the weekly report", "2026-09-05"),
    ("I need you to renew the library book by 2026-09-08.", "Renew the library book", "2026-09-08"),
    ("Don't forget to pay the electricity bill; deadline is 2026-09-09.",
     "Pay the electricity bill", "2026-09-09"),
    ("Prepare the demo video is due on 2026-09-10.", "Prepare the demo video", "2026-09-10"),
])
def test_task_frames(text, title, date):
    item = task(text)
    assert item.title == title
    assert item.deadline == date


@pytest.mark.parametrize("text,title,date,time,loc", [
    ("Calendar update: family dinner, 2026-09-19 at 10:00, the library.",
     "Family dinner", "2026-09-19", "10:00", "Library"),
    ("Reminder: mentor catch-up happens on 2026-09-16 at 11:00 in the city clinic.",
     "Mentor catch-up", "2026-09-16", "11:00", "City clinic"),
    ("Please join the AI workshop on 2026-09-13, 12:00 at Google Meet.",
     "AI workshop", "2026-09-13", "12:00", "Google Meet"),
    ("The product demo is scheduled for 2026-09-12 at 11:00 in Zoom.",
     "Product demo", "2026-09-12", "11:00", "Zoom"),
])
def test_event_frames(text, title, date, time, loc):
    item = event(text)
    assert item.type == "event"
    assert item.title == title
    assert item.date == date
    assert item.time == time
    assert item.location == loc


def test_time_is_normalised_to_24_hour_zero_padded():
    item = event("The project review is scheduled for 2026-09-02 at 9:00 in Zoom.")
    assert item.time == "09:00"


def test_named_person_is_extracted_when_actually_addressed():
    item = task("Please call Maya when you are free.")
    assert item.person == "Maya"


# --- priority ---------------------------------------------------------------
@pytest.mark.parametrize("date,expected", [
    ("2026-09-02", "high"),     # 1 day out
    ("2026-09-04", "high"),     # 3 days out
    ("2026-09-06", "medium"),   # 5 days out
    ("2026-09-20", "low"),      # 19 days out
])
def test_priority_tracks_deadline_proximity(date, expected):
    item = task(f"Please submit the weekly report by {date}.")
    assert item.priority == expected
    assert item.priority_reason


def test_explicit_urgency_overrides_proximity():
    item = task("Please urgently submit the weekly report by 2026-09-30.")
    assert item.priority == "high"
    assert "urgen" in item.priority_reason


def test_every_item_carries_a_priority_reason():
    for text in ["Could you send it soon?",
                 "Please submit the weekly report by 2026-09-05.",
                 "Let us meet sometime next week."]:
        item = task(text)
        assert item.priority_reason, f"no reason given for: {text}"


# --- scope ------------------------------------------------------------------
@pytest.mark.parametrize("category", [
    T.GENERAL_INFORMATION, T.PERSONAL_INFORMATION,
    T.PROMOTIONAL, T.SENSITIVE_INFORMATION,
])
def test_non_actionable_categories_produce_nothing(category):
    assert extract("MSG_Z", "The cafeteria closes at 6 PM.", "Neha",
                   SENT, category, 1, ()) is None
