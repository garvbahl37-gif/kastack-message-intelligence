"""Shared fixtures.

The L2 tests run against a synthetic corpus written here rather than against
the supplied dataset. Two reasons, and both matter: the dataset must not be
committed, and a test that only passes on one corpus is a test of that corpus
rather than of the code. Every lifecycle shape the L2 brief lists appears
below, in an order chosen so that chronology is doing real work.
"""

from __future__ import annotations

import pytest

from mint import ledger as L

#: Two batches. The second is processed after the first, and several of its
#: messages are only meaningful in the light of the first.
BATCH_ONE = """message_id,timestamp,sender,message
T_001,2026-11-02 09:00:00,Priya,Can you review the accessibility audit before 2026-11-09?
T_002,2026-11-02 09:30:00,Devansh,Please submit the quarterly budget sheet by 2026-11-06.
T_003,2026-11-02 10:00:00,Nadia,I need you to renew the domain certificate by 2026-11-05.
T_004,2026-11-02 10:30:00,Ops Desk,Reminder: fire drill happens on 2026-11-14 at 09:30 in the atrium.
T_005,2026-11-02 11:00:00,Priya,"Calendar update: design critique, 2026-11-12 at 14:00, Studio 3."
T_006,2026-11-02 11:30:00,Marketing,Half price on annual plans this week. Use code PLAN50.
T_007,2026-11-02 12:00:00,Devansh,The lift in block B is back in service.
T_008,2026-11-02 12:30:00,Private Message,Your OTP is 552317. It expires in five minutes.
T_009,2026-11-02 13:00:00,Private Message,You can contact me on 70155 28643 after six.
T_010,2026-11-02 13:30:00,Private Message,My recent test result says low iron.
"""

BATCH_TWO = """message_id,timestamp,sender,message
T_011,2026-11-03 09:00:00,Nadia,Can you share an update on review the accessibility audit?
T_012,2026-11-03 09:30:00,Devansh,Following up on submit the quarterly budget sheet; is it in progress?
T_013,2026-11-03 10:00:00,Nadia,"The deadline to renew the domain certificate is now 2026-11-04, earlier than previously planned. Treat this as urgent."
T_014,2026-11-03 10:30:00,Ops Desk,The fire drill has been moved to 2026-11-16 at 11:00. Please use the new schedule.
T_015,2026-11-03 11:00:00,Priya,Update: review the accessibility audit has been completed successfully.
T_016,2026-11-03 11:30:00,Priya,Follow-up: update: review the accessibility audit has been completed successfully.
T_017,2026-11-03 12:00:00,Devansh,Any progress on the item concerning the budget sheet?
T_018,2026-11-03 12:30:00,Ops Desk,"The date for fire drill stays the same, but the time is now 15:45."
T_019,2026-11-03 13:00:00,Priya,"The audit might already be signed off, but I am not completely sure."
T_020,2026-11-03 13:30:00,Ops Desk,The design critique has been cancelled.
T_021,2026-11-03 14:00:00,Project Lead,"The deadline may be Tuesday, or it may be Thursday. Wait for the official update."
T_022,2026-11-03 14:30:00,Unknown Sender,Was the vendor contract countersigned by the legal team?
T_023,2026-11-03 15:00:00,Private Message,Integration token: tok_test_9KQ22. Use it only locally.
T_024,2026-11-03 15:30:00,Operations,"Deliver the spare laptop to 4 Hillside Road, Pune."
T_025,2026-11-03 16:00:00,Mentor,New task: draft the retention policy by 2026-11-11.
"""


@pytest.fixture(scope="session")
def ledger() -> L.Ledger:
    return L.run([("batch_one.csv", BATCH_ONE), ("batch_two.csv", BATCH_TWO)])


@pytest.fixture(scope="session")
def one_batch() -> L.Ledger:
    """The first batch alone, for before/after comparisons."""
    return L.run([("batch_one.csv", BATCH_ONE)])


def group_titled(ledger: L.Ledger, needle: str):
    """Find a group by a fragment of its title. Fails loudly if ambiguous."""
    matches = [g for g in ledger.groups if needle.lower() in g.title.lower()]
    assert matches, f"no group matching {needle!r}"
    assert len(matches) == 1, f"{needle!r} matched {[g.title for g in matches]}"
    return matches[0]
