"""Tests for detection and masking.

Every secret used here is invented in-file. Nothing from the supplied dataset
appears in this repository, so these tests are safe to publish.
"""

from __future__ import annotations

import re

import pytest

from mint.sensitive import MASK_TOKEN, mask, scan

CASES = [
    ("Your OTP is 731209. It expires in 5 minutes.", "one_time_password", "high"),
    ("Use password RedCanyon#77 to sign in.", "password", "high"),
    ("My card number is 5555 4444 3333 2222.", "payment_card", "high"),
    ("Please note my bank account number 771903355012.", "bank_account", "high"),
    ("The temporary access token is tok_demo_Z9Q11W.", "auth_token", "high"),
    ("My account recovery code is RC-51-QP-08.", "account_recovery_code", "high"),
    ("My identification number is ID-9931-AB.", "government_id", "high"),
    ("My recent test result says low haemoglobin.", "health_information", "high"),
    ("My home address is 7 Hill Street, Pune-11.", "postal_address", "medium"),
    ("You can contact me on 91234 56789.", "phone_number", "medium"),
]


@pytest.mark.parametrize("text,expected_type,expected_risk", CASES)
def test_detects_type_and_risk(text, expected_type, expected_risk):
    result = scan(text)
    assert result.is_sensitive
    assert expected_type in result.types
    assert result.overall_risk == expected_risk


@pytest.mark.parametrize("text,expected_type,_risk", CASES)
def test_secret_value_is_gone_from_masked_text(text, expected_type, _risk):
    result = scan(text)
    assert MASK_TOKEN in result.masked_text
    # Every alphanumeric run of 4+ characters that was in the secret must be
    # absent. Words from the surrounding sentence are allowed to remain.
    secret_span = text[result.findings[0].span[0]:result.findings[0].span[1]]
    for chunk in re.findall(r"[A-Za-z0-9]{4,}", secret_span):
        assert chunk not in result.masked_text, f"{chunk!r} survived masking"


def test_mask_is_fixed_width_so_it_leaks_no_length():
    short = scan("Your OTP is 1234.").masked_text
    long = scan("Your OTP is 999888777666555.").masked_text
    assert short == long == f"Your OTP is {MASK_TOKEN}."


def test_masking_is_surgical_and_keeps_context():
    out = mask("Your OTP is 731209. It expires in 5 minutes.")
    assert out == f"Your OTP is {MASK_TOKEN}. It expires in 5 minutes."
    # The trailing "5 minutes" is not a secret and must survive.
    assert "5 minutes" in out


def test_finding_object_cannot_carry_the_secret():
    finding = scan("Use password RedCanyon#77 to sign in.").findings[0]
    serialised = repr(finding)
    assert "RedCanyon" not in serialised


def test_credential_mention_without_a_value_is_low_risk():
    result = scan("I will send the login details separately.")
    assert result.types == ["credential_mention"]
    assert result.overall_risk == "low"
    assert result.overall_action == "safe_to_process_locally"
    # Nothing to redact, so the text is returned untouched.
    assert MASK_TOKEN not in result.masked_text


def test_ordinary_messages_are_not_flagged():
    for text in [
        "The client discussion is scheduled for 2026-09-12 at 11:00 in Zoom.",
        "Please submit the weekly report by 2026-09-05.",
        "Get 25% off selected headphones this weekend. Use code SAVE30.",
        "The cafeteria closes at 6 PM.",
        "Remember that I prefer morning meetings.",
        "Free delivery on orders above 500 rupees.",
    ]:
        assert not scan(text).is_sensitive, f"false positive on: {text}"


def test_multiple_secrets_in_one_message_are_all_masked():
    result = scan("Your OTP is 445566 and my card number is 4444 3333 2222 1111.")
    assert {"one_time_password", "payment_card"} <= set(result.types)
    assert result.masked_text.count(MASK_TOKEN) == 2
    assert "445566" not in result.masked_text
    assert "4444" not in result.masked_text


def test_strictest_action_wins_when_types_are_mixed():
    result = scan("My home address is 7 Hill Street. Your OTP is 445566.")
    assert result.overall_risk == "high"
    assert result.overall_action == "do_not_store"


def test_token_prefix_is_caught_without_a_keyword():
    result = scan("Deploy with sk-live-9f2b7c41aa and restart.")
    assert "auth_token" in result.types
    assert "sk-live-9f2b7c41aa" not in result.masked_text
