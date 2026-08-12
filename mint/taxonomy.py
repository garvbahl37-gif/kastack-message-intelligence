"""The category taxonomy and the reasoning behind it.

Six categories are required by the brief. Five of them describe what a message
*is*; the sixth -- ``sensitive_information`` -- describes how the message must
be *handled*. Those are different axes, and a message can sit on both:

    "For my profile, my home address is <a private street address>."

is simultaneously personal information and sensitive information.

Decision: when both axes fire, ``sensitive_information`` wins the single
``category`` slot, because that is the label that changes the system's
behaviour (redact, restrict storage, never forward). The topical label is not
thrown away -- it is kept in ``secondary_category`` so no signal is lost.
"""

from __future__ import annotations

ACTION_REQUIRED = "action_required"
MEETING_EVENT = "meeting_or_event"
PERSONAL_INFORMATION = "personal_information"
GENERAL_INFORMATION = "general_information"
PROMOTIONAL = "promotional"
SENSITIVE_INFORMATION = "sensitive_information"

CATEGORIES = [
    ACTION_REQUIRED,
    MEETING_EVENT,
    PERSONAL_INFORMATION,
    GENERAL_INFORMATION,
    PROMOTIONAL,
    SENSITIVE_INFORMATION,
]

DESCRIPTIONS = {
    ACTION_REQUIRED: (
        "The sender asks the recipient to do something, or states a deliverable "
        "with a deadline."
    ),
    MEETING_EVENT: (
        "A meeting, appointment or event is announced, scheduled, proposed or "
        "recalled -- something to attend rather than something to produce."
    ),
    PERSONAL_INFORMATION: (
        "A durable personal fact or preference about the sender that is worth "
        "remembering but is not a secret or an identifier."
    ),
    GENERAL_INFORMATION: (
        "A neutral factual update with no action, no event and no personal or "
        "sensitive content."
    ),
    PROMOTIONAL: (
        "Marketing content: offers, discounts, coupon codes, upsells."
    ),
    SENSITIVE_INFORMATION: (
        "The message body contains a credential, financial identifier, personal "
        "identifier, health detail or private contact detail."
    ),
}

# Categories the statistical model is trained to predict. The model is
# deliberately *not* trained on sensitive_information: that label is decided by
# the deterministic detector in `sensitive.py`, because a probabilistic guess is
# the wrong tool for a decision about whether to redact something.
MODEL_CATEGORIES = [
    ACTION_REQUIRED,
    MEETING_EVENT,
    PERSONAL_INFORMATION,
    GENERAL_INFORMATION,
    PROMOTIONAL,
]
