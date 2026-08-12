"""End-to-end proof that no secret reaches any artifact the system produces.

Two complementary checks:

1.  A synthetic corpus with secrets this file invents is pushed through the
    real pipeline, and every generated artifact is searched for them. This runs
    everywhere, including CI, and needs no dataset.
2.  If the supplied dataset happens to be present locally, the generated output
    files are additionally scanned with *generic* secret shapes (long digit
    runs, token prefixes). No literal from the dataset is written here, so this
    file stays publishable.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from mint import pipeline

ROOT = Path(__file__).resolve().parents[1]

# Invented secrets, deliberately distinctive so a substring search is decisive.
PLANTED = [
    "884422", "MoonGate#41", "6011 5566 7788 9900", "tok_demo_QQ7781",
    "RC-77-ZZ-14", "ID-4417-KK", "998877 665544", "19 Orchid Lane, Kochi-88",
]

SYNTHETIC_CSV = """message_id,timestamp,sender,message
MSG_T01,2026-09-01 08:00:00,Riya,Your OTP is 884422. It expires in 10 minutes.
MSG_T02,2026-09-01 09:00:00,Riya,Use password MoonGate#41 to sign in to the portal.
MSG_T03,2026-09-01 10:00:00,Dev,My card number is 6011 5566 7788 9900.
MSG_T04,2026-09-01 11:00:00,Dev,The temporary access token is tok_demo_QQ7781.
MSG_T05,2026-09-01 12:00:00,Riya,My account recovery code is RC-77-ZZ-14.
MSG_T06,2026-09-01 13:00:00,Dev,My identification number is ID-4417-KK.
MSG_T07,2026-09-01 14:00:00,Riya,You can contact me on 998877 665544.
MSG_T08,2026-09-01 15:00:00,Dev,My home address is 19 Orchid Lane, Kochi-88.
MSG_T09,2026-09-01 16:00:00,Riya,Please submit the weekly report by 2026-09-05.
MSG_T10,2026-09-01 17:00:00,Dev,The project review is scheduled for 2026-09-09 at 14:00 in Zoom.
"""


@pytest.fixture(scope="module")
def result():
    return pipeline.run(io.StringIO(SYNTHETIC_CSV))


def test_masked_text_on_every_record_is_clean(result):
    blob = json.dumps([m.to_row() for m in result.messages])
    for secret in PLANTED:
        assert secret not in blob, f"{secret!r} leaked into the message records"


def test_classification_output_is_clean(result):
    blob = json.dumps(result.classifications())
    for secret in PLANTED:
        assert secret not in blob, f"{secret!r} leaked into Part 1 output"


def test_extraction_output_is_clean(result):
    blob = json.dumps([i.to_dict() for i in result.items])
    for secret in PLANTED:
        assert secret not in blob, f"{secret!r} leaked into Part 2 output"


def test_sensitive_report_is_clean(result):
    """The report about the secrets must not contain the secrets."""
    blob = json.dumps(result.sensitive_report())
    for secret in PLANTED:
        assert secret not in blob, f"{secret!r} leaked into Part 3 output"
    assert len(result.sensitive_report()) == 8


def test_written_files_are_clean(result, tmp_path):
    written = pipeline.write_outputs(result, tmp_path)
    for path in written:
        text = path.read_text(encoding="utf-8")
        for secret in PLANTED:
            assert secret not in text, f"{secret!r} leaked into {path.name}"


def test_processed_message_has_no_raw_field(result):
    """The record passed around downstream must be structurally incapable of
    holding raw text -- not merely empty of it."""
    fields = set(vars(result.messages[0]))
    assert "raw" not in fields and "message" not in fields
    assert "masked_text" in fields


# ---------------------------------------------------------------------------
# Optional: scan real generated outputs with generic secret shapes.
# ---------------------------------------------------------------------------
GENERIC_SECRET_SHAPES = [
    (re.compile(r"\b\d{6,}\b"), "a run of 6+ digits"),
    (re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b"), "a card-shaped digit group"),
    (re.compile(r"\b(?:sk|pk|tok|ghp|gho)[_-][A-Za-z0-9]{4,}\b"), "a token prefix"),
]

OUTPUTS = ROOT / "outputs"


@pytest.mark.skipif(not OUTPUTS.exists(), reason="no generated outputs present")
@pytest.mark.parametrize("name", [
    "classified_messages.json", "extracted_items.json",
    "sensitive_report.json", "classified_messages.csv",
])
def test_generated_outputs_contain_no_secret_shapes(name):
    path = OUTPUTS / name
    if not path.exists():
        pytest.skip(f"{name} has not been generated")
    text = path.read_text(encoding="utf-8")
    # Strip ISO timestamps and dates first; those are legitimately full of digits.
    text = re.sub(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}:\d{2})?", " ", text)
    for pattern, description in GENERIC_SECRET_SHAPES:
        found = pattern.findall(text)
        assert not found, f"{name} contains {description}: {sorted(set(found))[:5]}"
