"""Subject identity: what may be treated as the same thing, and what may not."""

from __future__ import annotations

import pytest

from mint.subject import SubjectSpace, content_tokens, signature

SUBJECTS = [
    "Review the privacy checklist", "Review the model results",
    "Upload the assignment", "Submit the weekly report",
    "Call the service centre", "Call Maya when you are free",
    "Prepare the demo video", "Prepare the offline inference demo",
    "Complete the onboarding form", "Update the project tracker",
    "Back up the project files", "Create a latency chart",
    "Internship orientation", "Team stand-up", "Sprint planning",
    "Study-group session", "Design review", "Project review",
]


@pytest.fixture(scope="module")
def space() -> SubjectSpace:
    return SubjectSpace([signature(s) for s in SUBJECTS])


@pytest.mark.parametrize("probe,subject", [
    ("the assignment", "Upload the assignment"),
    ("the model results", "Review the model results"),
    ("model-results review", "Review the model results"),
    ("the onboarding form", "Complete the onboarding form"),
    ("study-group", "Study-group session"),
    ("internship orientation", "Internship orientation"),
])
def test_partial_references_match(space, probe, subject):
    score, shared = space.match(signature(probe), signature(subject))
    assert score > 0, f"{probe!r} should reach {subject!r}"
    assert shared


@pytest.mark.parametrize("a,b,why", [
    ("Review the privacy checklist", "Review the model results",
     "they share only the cheap word 'review'"),
    ("Prepare the offline inference demo", "Prepare the demo video",
     "two thirds overlap is not containment and not a restatement"),
    ("Call Maya when you are free", "Call the service centre",
     "'call' alone does not identify a subject"),
    ("Design review", "Project review", "different subjects, same activity noun"),
    ("Team stand-up", "Sprint planning", "nothing in common"),
    ("Create a latency chart", "Review the model results", "unrelated"),
])
def test_unrelated_subjects_stay_apart(space, a, b, why):
    score, _ = space.match(signature(a), signature(b))
    assert score == 0.0, f"{a!r} and {b!r} must not match: {why}"


def test_a_single_cheap_word_cannot_carry_a_match(space):
    """The failure the brief names: grouping on one shared common word."""
    score, _ = space.match(signature("review"), signature("Review the model results"))
    assert score == 0.0


def test_a_single_rare_word_can_carry_a_match(space):
    """...but a word that names one subject is allowed to identify it."""
    score, shared = space.match(signature("assignment"),
                                signature("Upload the assignment"))
    assert score > 0 and shared == ["assignment"]


def test_generic_heads_alone_are_not_an_identity():
    assert not signature("meeting")
    assert not signature("the session")
    assert signature("study-group session")


def test_morphology_is_folded():
    assert content_tokens("results") == content_tokens("result")
    assert content_tokens("onboarding form") == ["onboard", "form"]
    # Hyphenated compounds contribute their parts as well as themselves.
    assert set(content_tokens("model-results")) >= {"model", "result"}


def test_idf_scales_with_the_subject_vocabulary(space):
    """A token naming one subject must outrank one naming many."""
    assert space.idf("assignment") > space.idf("review")
    assert space.distinctive_df >= 1
