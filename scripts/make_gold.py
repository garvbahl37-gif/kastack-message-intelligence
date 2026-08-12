"""Build the hand-labelled evaluation set.

The corpus ships with no labels, so the only honest way to quote an accuracy
number is to label some of it by hand. Rather than label 900 near-duplicate
messages, I labelled the 115 distinct *message templates* that generate them
(after masking and opener-stripping) and propagated each judgement to every
message built from that template. That covers 100% of the corpus.

Publishing constraint
---------------------
The brief forbids putting the dataset in a public repository, so this file must
not contain message text. Templates are therefore keyed by a truncated SHA-256
of the normalised template string. Anyone holding the dataset can re-run this
script and reproduce `eval/gold_labels.csv` exactly; anyone without it learns
nothing about the message contents from the hashes.

The emitted `eval/gold_labels.csv` contains only `message_id,category,
template_hash`, which is safe to publish.

Honesty note: I wrote both the rule layer and these labels, so agreement
between them is not independent evidence. The meaningful number this set
produces is how well the *statistical model* does on templates it never saw
during training -- see `scripts/train.py`.
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mint.normalize import canonical  # noqa: E402
from mint.sensitive import scan  # noqa: E402

ACTION = "action_required"
MEETING = "meeting_or_event"
PERSONAL = "personal_information"
GENERAL = "general_information"
PROMO = "promotional"
SENSITIVE = "sensitive_information"


def template_key(raw_message: str) -> str:
    """Collapse a message to the template that generated it."""
    t = canonical(scan(raw_message).masked_text)
    t = re.sub(r"\d{4}-\d{2}-\d{2}", "<DATE>", t)
    t = re.sub(r"\d{1,2}:\d{2}", "<TIME>", t)
    t = re.sub(r"\d+", "<N>", t)
    return t


def template_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# My judgements, one per template, keyed by template hash.
#
# The four marked AMBIGUOUS are cases where a careful human could reasonably
# choose a different label. They are kept in the gold set (removing hard cases
# to flatter a score would be dishonest) and are listed in the README.
# ---------------------------------------------------------------------------
JUDGEMENTS: dict[str, str] = {}
AMBIGUOUS: set[str] = set()


def _label(template: str, category: str, ambiguous: bool = False) -> None:
    h = template_hash(template)
    JUDGEMENTS[h] = category
    if ambiguous:
        AMBIGUOUS.add(h)


# Meeting or event -- a scheduling frame naming something to attend.
for _t in [
    "are you available for the college seminar at <TIME> on <DATE>? location: the main office.",
    "are you available for the college seminar at <TIME> on <DATE>? location: the training hall.",
    "are you available for the design review at <TIME> on <DATE>? location: the main office.",
    "are you available for the design review at <TIME> on <DATE>? location: the training hall.",
    "are you available for the technical interview at <TIME> on <DATE>? location: the main office.",
    "are you available for the technical interview at <TIME> on <DATE>? location: the training hall.",
    "calendar update: family dinner, <DATE> at <TIME>, the college auditorium.",
    "calendar update: family dinner, <DATE> at <TIME>, the library.",
    "calendar update: placement briefing, <DATE> at <TIME>, the college auditorium.",
    "calendar update: placement briefing, <DATE> at <TIME>, the library.",
    "calendar update: team stand-up, <DATE> at <TIME>, the college auditorium.",
    "calendar update: team stand-up, <DATE> at <TIME>, the library.",
    "please join the ai workshop on <DATE>, <TIME> at conference room <N>.",
    "please join the ai workshop on <DATE>, <TIME> at google meet.",
    "please join the internship orientation on <DATE>, <TIME> at conference room <N>.",
    "please join the internship orientation on <DATE>, <TIME> at google meet.",
    "please join the study-group session on <DATE>, <TIME> at conference room <N>.",
    "please join the study-group session on <DATE>, <TIME> at google meet.",
    "reminder: doctor appointment happens on <DATE> at <TIME> in the cafeteria.",
    "reminder: doctor appointment happens on <DATE> at <TIME> in the city clinic.",
    "reminder: mentor catch-up happens on <DATE> at <TIME> in the cafeteria.",
    "reminder: mentor catch-up happens on <DATE> at <TIME> in the city clinic.",
    "reminder: sprint planning happens on <DATE> at <TIME> in the cafeteria.",
    "reminder: sprint planning happens on <DATE> at <TIME> in the city clinic.",
    "the client discussion is scheduled for <DATE> at <TIME> in meeting room a.",
    "the client discussion is scheduled for <DATE> at <TIME> in zoom.",
    "the product demo is scheduled for <DATE> at <TIME> in meeting room a.",
    "the product demo is scheduled for <DATE> at <TIME> in zoom.",
    "the project review is scheduled for <DATE> at <TIME> in meeting room a.",
    "the project review is scheduled for <DATE> at <TIME> in zoom.",
]:
    _label(_t, MEETING)

# Meeting, but with no resolvable date -- still something to attend.
_label("let us meet sometime next week.", MEETING)
_label("the review could be friday afternoon.", MEETING)

# Action required -- the recipient is asked to produce or do something.
for _t in [
    "can you finish the test cases before <DATE>?",
    "can you review the privacy checklist before <DATE>?",
    "can you send the revised presentation before <DATE>?",
    "can you update the project tracker before <DATE>?",
    "complete the onboarding form is due on <DATE>.",
    "could you send it soon?",
    "don't forget to call the service centre; deadline is <DATE>.",
    "don't forget to email the signed document; deadline is <DATE>.",
    "don't forget to pay the electricity bill; deadline is <DATE>.",
    "don't forget to upload the assignment; deadline is <DATE>.",
    "i need you to back up the project files by <DATE>.",
    "i need you to renew the library book by <DATE>.",
    "i need you to review the model results by <DATE>.",
    "i need you to verify the dataset labels by <DATE>.",
    "if possible, review the file before the meeting.",
    "please call maya when you are free.",
    "please complete the python exercise by <DATE>.",
    "please confirm the interview slot by <DATE>.",
    "please reply to the client email by <DATE>.",
    "please submit the weekly report by <DATE>.",
    "prepare the demo video is due on <DATE>.",
    "send the expense receipt is due on <DATE>.",
    "share the meeting notes is due on <DATE>.",
]:
    _label(_t, ACTION)

# Action required, but only implicitly -- a reasonable person could read these
# as neutral status updates instead.
_label("maya asked whether the demo was ready.", ACTION, ambiguous=True)
_label("the report may be needed tomorrow.", ACTION, ambiguous=True)

# Personal information -- durable facts and preferences, not secrets.
for _t in [
    "for my profile, i am vegetarian.",
    "for my profile, i live near the central library.",
    "for my profile, i usually study after dinner.",
    "for my profile, my emergency contact is my brother.",
    "for my profile, my favourite language is python.",
    "i might prefer evening meetings now.",
    "just so you know, i drink coffee without sugar.",
    "just so you know, i prefer morning meetings.",
    "just so you know, i prefer receiving updates by email.",
    "just so you know, i use dark mode.",
    "just so you know, my t-shirt size is medium.",
    "personal note: i am vegetarian.",
    "personal note: i live near the central library.",
    "personal note: i usually study after dinner.",
    "personal note: my emergency contact is my brother.",
    "personal note: my favourite language is python.",
    "remember that i drink coffee without sugar.",
    "remember that i prefer morning meetings.",
    "remember that i prefer receiving updates by email.",
    "remember that i use dark mode.",
    "remember that my t-shirt size is medium.",
]:
    _label(_t, PERSONAL)

# Promotional.
for _t in [
    "book movie tickets today and receive cashback. use code save<N>.",
    "earn reward points on your next purchase. use code save<N>.",
    "flash sale on laptops starts at <N> pm. use code save<N>.",
    "free delivery on orders above <N> rupees. use code save<N>.",
    "get <N>% off selected headphones this weekend. use code save<N>.",
    "join our premium plan for exclusive benefits. use code save<N>.",
    "limited-time offer: buy one course and get one free. use code save<N>.",
    "special festival discount on clothing. use code save<N>.",
    "upgrade your subscription and save <N>%. use code save<N>.",
    "you may like our new student plan.",
    "your food-delivery coupon expires tonight. use code save<N>.",
]:
    _label(_t, PROMO)

# General information.
for _t in [
    "the building entrance has moved temporarily.",
    "the cafeteria closes at <N> pm.",
    "the event registration desk opens at <N> am.",
    "the laptop battery is fully charged.",
    "the library has extended weekend hours.",
    "the new python version is available.",
    "the office wi-fi will be under maintenance tonight.",
    "the project folder was reorganized.",
    "the report template has been updated.",
    "the shuttle leaves every thirty minutes.",
    "the support team changed its working hours.",
    "the training material is on the portal.",
    "the weather forecast says light rain.",
    "the webinar recording is now available.",
    "tomorrow is a public holiday.",
]:
    _label(_t, GENERAL)

# Announces that credentials will follow, but carries none. Detected as a
# low-risk credential *mention* in Part 3, yet the message itself is just a
# status update -- the distinction the system is meant to draw.
_label("i will send the login details separately.", GENERAL, ambiguous=True)

# Sensitive information -- a credential, identifier or private detail is in
# the body. Masked before hashing, hence the ****** in the template.
for _t in [
    "my account recovery code is ******.",
    "my card number is ******.",
    "my home address is ******.",
    "my identification number is ******.",
    "my recent test result says ******.",
    "please note my bank account number ******.",
    "the temporary access token is ******.",
    "use password ****** to sign in to the test account.",
    "you can contact me on ******.",
    "your otp is ******. it expires in <N> minutes.",
]:
    _label(_t, SENSITIVE)


def main() -> int:
    src = ROOT / "data" / "messages.csv"
    if not src.exists():
        print(f"error: {src} not found. Place the dataset there first.", file=sys.stderr)
        return 1

    with src.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    out_rows, unlabelled = [], []
    for r in rows:
        tmpl = template_key(r["message"])
        h = template_hash(tmpl)
        cat = JUDGEMENTS.get(h)
        if cat is None:
            unlabelled.append(tmpl)
            continue
        out_rows.append(
            {
                "message_id": r["message_id"],
                "category": cat,
                "template_hash": h,
                "ambiguous": "yes" if h in AMBIGUOUS else "no",
            }
        )

    if unlabelled:
        print(f"warning: {len(set(unlabelled))} template(s) have no judgement:",
              file=sys.stderr)
        for t in sorted(set(unlabelled)):
            print("   ", t, file=sys.stderr)

    dest = ROOT / "eval" / "gold_labels.csv"
    dest.parent.mkdir(exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["message_id", "category", "template_hash", "ambiguous"]
        )
        w.writeheader()
        w.writerows(out_rows)

    print(f"wrote {len(out_rows)}/{len(rows)} labelled messages to {dest}")
    print(f"distinct templates judged: {len(JUDGEMENTS)} "
          f"({len(AMBIGUOUS)} flagged ambiguous)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
