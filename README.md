# Local Message Intelligence

A message-triage system that classifies messages into six categories, extracts
tasks and events without inventing anything, and detects and masks sensitive
information before any other stage of the pipeline runs.

Everything runs locally. There is not a single outbound network call in the
`mint` package — no message text is sent to any external service, AI or
otherwise.

**Live demo:** https://kastack-message-intelligence.vercel.app

---

## Contents

- [What it does](#what-it-does)
- [Results](#results)
- [How message classification works](#how-message-classification-works)
- [How tasks and events are extracted](#how-tasks-and-events-are-extracted)
- [How sensitive information is detected and masked](#how-sensitive-information-is-detected-and-masked)
- [Running it](#running-it)
- [Project layout](#project-layout)
- [Assumptions](#assumptions)
- [Limitations](#limitations)
- [AI-tool usage declaration](#ai-tool-usage-declaration)

---

## What it does

```
CSV row
   │
   ▼
┌──────────────────────┐   the ONLY stage that sees raw text
│  scan & mask         │   detects credentials/identifiers, redacts the value
└──────────┬───────────┘
           │  masked text only from here on
           ├────────────────────────────────┐
           ▼                                ▼
┌──────────────────────┐          ┌──────────────────────┐
│  rule frames         │          │  TF-IDF + logistic   │
│  high precision      │          │  regression, trained │
│  abstains when unsure│          │  on masked text only │
└──────────┬───────────┘          └──────────┬───────────┘
           └───────────┬────────────────────┘
                       ▼
            ┌──────────────────────┐
            │  arbitration         │  agreement → confidence up
            │  → category          │  conflict  → confidence down + review flag
            │  → confidence        │
            │  → reason            │
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │  task/event extract  │  nulls where nothing was stated
            └──────────────────────┘
```

Three independent mechanisms, deliberately given different authority:

| Layer | Kind | Authority |
|---|---|---|
| Sensitive detector | Deterministic regex | **Veto.** Overrides everything |
| Rule frames | Hand-written grammar patterns | High precision, abstains freely |
| Linear model | Trained TF-IDF + logistic regression | Always has an opinion |

The detector is not a vote — it is a veto. Whether to redact something is a
safety decision, and safety decisions should not be made by a softmax.

---

## Results

Measured on the 900-message assessment corpus.

### Classification against a hand-labelled gold set

The corpus ships without labels, so I labelled it myself. Rather than label 900
near-duplicate messages, I labelled the **115 distinct templates** that generate
them and propagated each judgement — covering 100% of the corpus.
(`scripts/make_gold.py`, output in `eval/gold_labels.csv`.)

| Metric | Result |
|---|---|
| Overall accuracy | **97.8%** (880 / 900) |
| Excluding 3 templates I flagged as genuinely ambiguous | **100.0%** (870 / 870) |
| On those ambiguous templates | 33.3% (10 / 30) |
| **Accuracy on messages the system did _not_ flag for review** | **100.0%** (850 / 850) |

The last row is the one I care about most: **all 20 errors fall inside the 50
messages the system flagged as needing review.** The system does not merely make
few mistakes — it knows which of its answers not to trust.

| Category | Precision | Recall | F1 | n |
|---|---|---|---|---|
| action_required | 0.960 | 0.960 | 0.960 | 250 |
| meeting_or_event | 1.000 | 1.000 | 1.000 | 170 |
| personal_information | 1.000 | 1.000 | 1.000 | 110 |
| general_information | 0.938 | 0.938 | 0.938 | 160 |
| promotional | 1.000 | 1.000 | 1.000 | 110 |
| sensitive_information | 1.000 | 1.000 | 1.000 | 100 |

### The model on its own, tested on sentence forms it has never seen

| Cross-validation scheme | Accuracy |
|---|---|
| **Grouped by template** — every message from a template lands in the same fold | **95.5%** |
| Random stratified split | 100.0% |

The 100% is the misleading number, and it is reported precisely so it can be
dismissed. With 900 messages generated from 115 templates, a random test set is
full of sentences whose near-identical twins are in the training set; the model
scores perfectly by memorising. Grouping by template forces it to classify
wording it has genuinely never seen. **95.5% is the honest figure.**

The residual errors are interpretable: held-out `general_information` templates
that mention a time or a place ("The cafeteria closes at 6 PM") drift toward
`meeting_or_event` when the model has never seen that template. The rule layer
gets those right, which is exactly why the hybrid exists.

### Extraction and detection

| | |
|---|---|
| Tasks extracted | 250 |
| Events extracted | 170 |
| Items with at least one unresolved field | 405 |
| Sensitive messages detected | 110 (100 carrying a value + 10 bare credential mentions) |
| Distinct sensitivity types | 11 |
| Tests | 93 passing |

### An honesty note about these numbers

I wrote the rule layer *and* the gold labels, so agreement between them is not
independent evidence — it partly measures my own consistency. The genuinely
independent number is the grouped-CV score, where a model trained by an
algorithm is tested on templates withheld from it. I have not tuned anything
against a held-out test set that I then report as a headline figure, and the
three ambiguous templates are kept in the gold set rather than quietly dropped
to flatter the score.

---

## How message classification works

### The six categories

| Category | Definition |
|---|---|
| `action_required` | The sender asks the recipient to do something, or states a deliverable with a deadline |
| `meeting_or_event` | Something to **attend** — announced, scheduled, proposed or recalled |
| `personal_information` | A durable personal fact or preference; not a secret or an identifier |
| `general_information` | A neutral factual update, with no action, event, personal or sensitive content |
| `promotional` | Marketing: offers, discounts, coupon codes, upsells |
| `sensitive_information` | The body contains a credential, financial/personal identifier, health detail or private contact detail |

**One deliberate taxonomy decision.** Five of these describe what a message *is*;
`sensitive_information` describes how it must be *handled*. Those are different
axes, and a message can sit on both — "For my profile, my home address is …" is
simultaneously personal and sensitive. When both fire, `sensitive_information`
wins the single `category` slot, because that is the label that changes system
behaviour (redact, restrict storage, never forward). The topical reading is not
discarded: it is kept in `secondary_category`.

I did not add or merge any of the six required categories.

### Layer 1 — rule frames, not keywords

Each rule is a named linguistic **frame**: a grammatical shape that carries
intent. The word "meeting" appearing anywhere does not make a message a meeting;
the frame `the X is scheduled for Y` does.

```python
("scheduled_copula", r"\b(?:is|are|was|will\s+be)\s+scheduled\s+(?:for|on|at)\b"),
("occurrence_verb",  r"\b(?:happens|takes\s+place|will\s+be\s+held)\s+(?:on|at|in)\b"),
("availability_probe", r"\bare\s+you\s+(?:available|free)\s+for\b"),
```

This is what keeps precision high on messages that share vocabulary across
categories:

| Message | Category | Why |
|---|---|---|
| "The webinar recording is now available." | `general_information` | "webinar" is present, but no scheduling frame is |
| "The event registration desk opens at 9 AM." | `general_information` | Facility hours, not something to attend |
| "Share the meeting notes is due on 2026-09-14." | `action_required` | Deadline frame beats the word "meeting" |
| "Please confirm the interview slot by 2026-09-05." | `action_required` | Asks you to act; no date/time/place for the interview itself |
| "I might prefer evening meetings now." | `personal_information` | First-person preference frame |

Frames are evaluated in precedence order — promotional, meeting, action,
personal, general — because marketing copy borrows the grammar of every other
category ("Join our premium plan", "Offer ends 2026-09-10").

Crucially, the rule layer **abstains** when nothing matches. Abstention is
different from "confidently general information", and the arbitration step
treats it differently.

### Layer 2 — a trained classifier

TF-IDF (1–2 grams, sublinear TF) into multinomial logistic regression
(`C=4.0`, balanced class weights). Trained on **masked text only**.

The corpus has no labels, so training labels come from **weak supervision**: the
high-precision frames label the 760 messages they are confident about
(unhedged, confidence ≥ 0.84) and abstain on the remaining 140. The model then
generalises beyond the frames.

Tokenisation folds dates, times, numbers and mask tokens into placeholders
(`datetoken`, `timetoken`, `numtoken`, `maskedtoken`). Without this the model
would learn that `2026-09-09` means "deadline" — memorising this corpus instead
of learning a transferable signal.

The model is trained on **five** categories, not six. `sensitive_information` is
excluded by design: it belongs to the deterministic detector.

### Layer 3 — arbitration, and where confidence comes from

Confidence is computed, not decorative:

| Situation | Confidence | Review flag |
|---|---|---|
| Detector fires | 0.98 (fixed) | no |
| Rule and model agree | `max(rule, model) + 0.04`, capped at 0.99 | only if hedged |
| Rules abstained, model alone | `model × 0.88`, capped at 0.88 | if < 0.60 |
| Model unavailable, rules alone | `rule × 0.92` | if < 0.60 |
| **They disagree** | `0.45 + 0.5 × spread`, clamped to [0.30, 0.72] | **always** |

Agreement between two independent methods is real evidence and earns a bounded
bonus. Disagreement is real evidence too, and costs. Nothing is ever assigned
1.0 — a classifier that claims certainty is not being honest.

Every decision carries a written reason naming the frame that fired, the text
that triggered it, and what the model thought:

```
The rule layer matched the calendar marker frame ("calendar update:"), and the
trained classifier independently agreed at p=0.99.
```

```
The two methods disagreed: the rule layer matched the hedged action frame
("may be needed") suggesting action_required (0.50), while the trained
classifier preferred general_information (p=0.77). Resolved to
general_information as the more confident reading; flagged for review because
action_required is a defensible alternative.
```

The system also degrades gracefully: with no trained model present it still
classifies every message using rules alone, at reduced confidence.

---

## How tasks and events are extracted

Only `action_required` and `meeting_or_event` messages are considered — the
other four carry nothing schedulable by definition.

Extraction is frame-based, so the title comes out as `Review the privacy
checklist` rather than as the whole sentence:

```python
("request_with_deadline",
 r"\b(?:can|could|would|will)\s+you\s+(?P<title>.+?)\s+(?:by|before)\s+(?P<date>\d{4}-\d{2}-\d{2})"),
("calendar_entry",
 r"^calendar\s+(?:update|invite|entry)\s*:\s*(?P<title>[^,]+?)\s*,\s*(?P<date>…)\s+at\s+(?P<time>…)\s*,\s*(?P<loc>.+?)\.?$"),
```

### Nothing is invented

This is the constraint the module is built around.

**Dates.** Only an *absolute* date in the text fills the `date` field. A message
saying "tomorrow" leaves `date` as `null`, preserves the phrase in `date_raw`,
and sets `date_status` to `unresolved_relative`:

```json
{
  "item_id": "EVENT_0007", "type": "event", "title": "Review",
  "date": null, "date_raw": "Friday afternoon",
  "date_status": "unresolved_relative", "date_suggestion": null,
  "time": null, "person": null, "priority": "medium",
  "priority_reason": "an event is stated but its date could not be resolved",
  "unresolved_fields": ["date", "time", "person"],
  "source_message_id": "MSG_0037"
}
```

Where a relative phrase *could* be resolved against the message timestamp
("tomorrow" → the next day), the result is offered in a separate
`date_suggestion` field and never promoted into `date`. Resolving it would
usually be right — but "usually right" is exactly the silent inference this
system exists to avoid. The suggestion is a proposal for a human; the `date`
field is an assertion, and it stays empty.

**People.** `person` is filled only when a known participant is actually named,
or when a capitalised name appears in a position that implies a person ("call
**Maya**", "**Nadia** asked"). A bare capitalised word is not enough — an early
version happily returned `"person": "Meeting"` out of "Meeting Room A".
The sender is recorded separately in `source_sender` and is **never** promoted
into `person`: "who sent this" and "who is involved" are different questions.

**Priority** is derived from deadline proximity and always states its reason:

| Condition | Priority |
|---|---|
| Explicit urgency wording ("urgent", "asap") | `high` |
| Deadline ≤ 3 days after the message | `high` |
| 4–7 days | `medium` |
| > 7 days | `low` |
| No resolvable date (event) | `medium` |
| No resolvable date (task) | `low` |

```json
{ "priority": "high", "priority_reason": "the deadline is 3 day(s) after the message" }
```

Every item lists its own gaps in `unresolved_fields`, so incomplete extractions
are visible rather than silently plausible.

---

## How sensitive information is detected and masked

### Detected types

| Type | Risk | Recommended action |
|---|---|---|
| `one_time_password` | high | `do_not_store` |
| `password` | high | `do_not_store` |
| `auth_token` | high | `do_not_store` |
| `account_recovery_code` | high | `do_not_store` |
| `payment_card` | high | `do_not_store` |
| `bank_account` | high | `do_not_store` |
| `government_id` | high | `do_not_send_to_external_service` |
| `health_information` | high | `do_not_send_to_external_service` |
| `postal_address` | medium | `ask_for_confirmation` |
| `phone_number` | medium | `ask_for_confirmation` |
| `credential_mention` | low | `safe_to_process_locally` |

Risk reflects harm if leaked. Live credentials are `do_not_store` because
retaining them has no upside. Identity and health data are `high` but
`do_not_send_to_external_service` rather than `do_not_store`: they may
legitimately need local retention, but must never cross a network boundary.
Health data is rated high as special-category personal data under GDPR Art. 9
and India's DPDP Act, not because it is a credential.

### Detection is keyword-anchored

A bare number is not a secret; a number introduced by "OTP is" is. Anchoring on
the surrounding language is what keeps false positives near zero in a corpus
full of dates, times and prices.

```python
Detector(
    "one_time_password",
    r"\b(?:OTP|one[\s-]?time\s+(?:password|code|pin)|verification\s+code)\b"
    r"(?:\s+(?:is|:|=))?\s*(\d[\d\s\-]{2,}\d)",
    "a one-time authentication code is present in the message body",
)
```

Exactly one capturing group — the span to redact. The anchoring keywords sit
*outside* it, so they survive masking and remain available to the classifier.
Well-known secret prefixes (`sk-`, `tok_`, `ghp_`, `xoxb-`) are additionally
caught with no keyword at all.

### `credential_mention`: detected, but not risky

```
"I will send the login details separately."
```

This refers to credentials but contains none. It is reported in the Part-3
output at **low** risk with `safe_to_process_locally`, and classified as
`general_information` — because there is nothing to redact and nothing to
protect. Distinguishing "mentions a secret" from "contains a secret" is the
point of that type.

### Masking

Masking is **surgical** — only the matched value span is replaced, so the
sentence stays readable and downstream classification still works:

```
Your OTP is 731209. It expires in 5 minutes.     <- invented example
Your OTP is ******. It expires in 5 minutes.
```

Note that "5 minutes" survives: it is a number, but not the secret.

The mask has **fixed width** regardless of the original value's length. A mask
that mirrored the length would leak it — a small but real side channel:

```python
assert scan("Your OTP is 1234.").masked_text \
    == scan("Your OTP is 99988877766.").masked_text
```

### The property that makes the rest of the system safe

Masking happens **first**, before anything else runs, and the raw text is never
stored on the record the program passes around. `ProcessedMessage` has no `raw`
field — it cannot leak a secret it is structurally incapable of holding, and
`SensitiveFinding` likewise stores a character span, never a value.

That is why no downstream stage — classifier, extractor, JSON writer, HTTP
response, log line, the web UI — needs its own redaction logic. By the time any
of them run, there is nothing left to redact.

Three independent checks enforce this:

1. `tests/test_no_leakage.py` pushes a synthetic corpus containing invented
   secrets through the real pipeline and asserts none appears in any produced
   artifact — including the sensitive report itself.
2. The same file scans real generated outputs for *generic* secret shapes
   (6+ digit runs, card groupings, token prefixes) after stripping timestamps.
3. `scripts/train.py` aborts the build if any secret-shaped string reaches the
   model vocabulary.

In the browser, the mask token renders as a redaction bar whose text node is
removed entirely — there is nothing under it to select, copy, screenshot or read
out of the DOM. Hovering names the *type*; it never reveals the value.

---

## Running it

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

Place the dataset at `data/messages.csv` (it is gitignored and must not be
committed).

```bash
# 1. Build the hand-labelled gold set        -> eval/gold_labels.csv
python scripts/make_gold.py

# 2. Train and export the model              -> models/classifier.json
python scripts/train.py

# 3. Run the pipeline                        -> outputs/*.json
python scripts/run_pipeline.py --evaluate --ids data/mandatory_demo_ids.csv

# 4. Tests
python -m pytest

# 5. Web app
uvicorn api.index:app --reload --port 8000
```

Generated outputs (all containing masked text only):

| File | Contents |
|---|---|
| `outputs/classified_messages.json` | Part 1 — category, confidence, reason, full evidence trail |
| `outputs/extracted_items.json` | Part 2 — tasks and events with unresolved fields marked |
| `outputs/sensitive_report.json` | Part 3 — type, risk, masked text, recommended action |
| `outputs/summary.json` | Aggregate counts |
| `outputs/classified_messages.csv` | Flat view of the above |

### Deployment

The service deploys to Vercel as a single Python function.

Inference at request time is **pure Python** (`mint/model.py`): scikit-learn
trains the model, then exports vocabulary, IDF weights, coefficients and
intercepts to a plain JSON artifact, and ~80 lines of standard-library Python
reproduce the forward pass. `tests/test_model.py` asserts agreement with
scikit-learn's own `predict_proba` to within 1e-3.

This buys three things: the runtime installs one dependency instead of a ~100 MB
scientific stack; the artifact is a diffable, reviewable file of numbers rather
than an opaque pickle that executes code on load; and because the forward pass
is right there, each prediction decomposes into the exact token contributions
that produced it — which is what the UI's evidence panel shows.

`.vercelignore` excludes `data/`, `outputs/` and `eval/`, so the dataset and
everything derived from it never reach the hosting platform. The deployed demo
ships only the self-authored `sample_data/sample_messages.csv`.

---

## Project layout

```
mint/                     the system (no network calls anywhere)
  sensitive.py            detection + masking — the only module that sees raw text
  rules.py                linguistic frames; weak-supervision label source
  model.py                pure-Python TF-IDF + logistic-regression inference
  classifier.py           arbitration between detector, rules and model
  extract.py              task/event extraction with strict null handling
  pipeline.py             orchestration; mask-first ordering
  normalize.py            opener stripping, tokenisation
  taxonomy.py             the six categories and the reasoning behind them
api/index.py              FastAPI service (thin — all logic lives in mint/)
web/index.html            single-page UI, no external assets
scripts/                  make_gold.py, train.py, run_pipeline.py
tests/                    93 tests
eval/gold_labels.csv      hand labels — IDs and categories only, no message text
models/classifier.json    exported model artifact
sample_data/              synthetic sample written for the public demo
data/, outputs/           gitignored — the dataset and its derivatives
```

---

## Assumptions

1. **Chronological order is enforced, not assumed.** `read_rows` sorts by
   `(timestamp, message_id)` rather than trusting file order.
2. **One category per message**, as the brief specifies, with
   `sensitive_information` winning ties and the topical label preserved in
   `secondary_category`.
3. **The participant roster is derived from the `sender` column** of whatever
   file is loaded, so person-matching adapts to a new dataset. Non-human senders
   (`Promotions`, `HR Team`, `Project Lead`) are excluded from it.
4. **Timestamps are the send time**, and are the only reference point offered
   for relative dates — and only as a labelled suggestion.
5. **All times are naive local time.** No timezone appears in the data, so none
   is inferred.
6. **The dataset is fictional**, but every sensitive-looking value is treated as
   if it were real.

---

## Limitations

**Corpus.** The 900 messages come from 115 templates. Real inboxes are far more
varied, and I would expect accuracy to drop on genuinely free-form text. The
grouped-CV design is my attempt to measure this honestly rather than hide it,
but held-out templates from the same generator are still easier than the real
world.

**The gold set is mine.** I wrote the rules and the labels, so their agreement
partly measures my own consistency rather than correctness. A second annotator
and an inter-annotator agreement score would be the fix.

**Genuinely ambiguous messages.** Three templates resist confident labelling,
and the system gets 10/30 of them right:

- *"I will send the login details separately."* — a status update, or an action?
  I labelled it `general_information`; the model says `action_required` at 0.43,
  driven by the token "send". Flagged for review.
- *"The report may be needed tomorrow."* — I labelled it `action_required`; the
  model says `general_information` at 0.58. The two layers disagree, which is
  the correct behaviour for a sentence that hedges twice.
- *"Maya asked whether the demo was ready."* — a relayed request. Classified
  `action_required`, but the extracted title is the mechanically faithful and
  awkward `"Demo was ready"`; a better title would require inventing a verb.

**English only, and rule-bound.** Frames are English-specific. Novel phrasings
for a known intent will fall through to the model alone, at reduced confidence.

**Detection is recall-limited by its anchors.** A secret with no keyword and no
recognised prefix — a bare 6-digit number on its own line — is not detected. My
own leak test caught a real instance of this during development: the phone
detector assumed 5+5 digit grouping and missed a 6+6 format, which is why the
pattern is now liberal about grouping once the keyword anchor has established
what the value is.

**Confidence is calibrated by construction, not by fitting.** The bonuses and
penalties are principled but hand-chosen. Proper calibration (Platt scaling or
isotonic regression against held-out labels) would be the next step.

**No temporal reasoning across messages.** Each message is classified in
isolation. A thread where "the review" is scheduled in one message and moved in
another produces two unlinked items.

### Possible improvements

- Local sentence embeddings (e.g. a MiniLM run offline) as a third voice, to
  catch paraphrases the frames miss without sending anything externally.
- Active learning: the `needs_review` queue is already the ideal labelling
  batch — every current error is in it.
- Cross-message coreference to merge duplicate and updated events.
- Proper probability calibration once a second annotator exists.
- Extending detection with checksum validation (Luhn for cards, Verhoeff for
  Aadhaar) to raise precision on unanchored numeric values.

---

## AI-tool usage declaration

**AI tools were used substantially in building this project, and disclosing that
accurately matters more than appearing to have done it unaided.**

**Tool used:** Anthropic's Claude (Claude Code, the agentic CLI), run locally
against the Claude API.

**What it was used for:**
- Exploratory analysis of the dataset's structure (recovering the 115 underlying
  templates, which shaped the whole evaluation design).
- Drafting the implementation of every module in `mint/`, the FastAPI service,
  the web UI, the test suite and this README.
- Designing the evaluation methodology, including the grouped-by-template
  cross-validation and the review-flag coverage metric.

**What it was *not* used for:**
- **No message text was ever sent to any AI service for classification.** The
  system contains no API calls. Classification, extraction and detection are
  performed entirely by the local regex frames and the local logistic-regression
  model in this repository.
- No labels were produced by an LLM. The 115 template judgements in
  `scripts/make_gold.py` are my own.
- The trained model is a scikit-learn logistic regression fitted on this corpus.
  There is no LLM anywhere in the inference path.

**Sample message text was shown to the AI tool during development** — while
analysing dataset structure and writing tests. This was a considered trade-off,
made because the data is explicitly fictional. The rule the brief sets — never
send raw messages to an external AI service *as the system's method of
processing them* — is honoured absolutely: the shipped system is fully offline.
Had the data been real, I would have developed against a synthetic fixture like
`sample_data/sample_messages.csv` instead.

**Understanding.** I can explain every design decision here — why the detector
vetoes rather than votes, why the model is trained on masked text, why the
random-split score is reported only to be dismissed, why `date_suggestion` is
kept separate from `date`, and why fixed-width masking matters. The reasoning is
documented inline throughout the source, not only in this file.

**Other libraries:** scikit-learn and NumPy (training only), FastAPI and
python-multipart (serving), pytest (tests). No pre-trained language models, no
external inference APIs.
