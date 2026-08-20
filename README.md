# Local Message Intelligence

A message-triage system in two layers.

**L1** classifies messages into six categories, extracts tasks and events
without inventing anything, and detects and masks sensitive information before
any other stage of the pipeline runs.

**L2** extends it from *reading messages* to *tracking subjects*. It works out
which messages are about the same thing, follows each subject through being
raised, chased, rescheduled, contradicted, completed or cancelled, prioritises
what is still open and explains why, answers questions about all of it, and
enforces a privacy route on every message before any of that can quote it.

Everything runs locally. There is not a single outbound network call in the
`mint` package — no message text is sent to any external service, AI or
otherwise.

| | |
|---|---|
| **Live demo — L2** | https://kastack-message-intelligence.vercel.app/l2 |
| **Live demo — L1** | https://kastack-message-intelligence.vercel.app |
| **Repository** | https://github.com/garvbahl37-gif/kastack-message-intelligence |
| **Walkthrough — L2** | [`docs/DEMO_SCRIPT_L2.md`](docs/DEMO_SCRIPT_L2.md) |
| **Walkthrough — L1** | [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) |

The hosted demo loads a synthetic sample written for this repository. The
assessment datasets are not committed here and are not deployed; use **Upload
batches** to analyse real files, which are processed in memory for one request
and never written to disk.

---

## Contents

**L1 — reading a message**

- [What it does](#what-it-does)
- [Results](#results)
- [How message classification works](#how-message-classification-works)
- [How tasks and events are extracted](#how-tasks-and-events-are-extracted)
- [How sensitive information is detected and masked](#how-sensitive-information-is-detected-and-masked)

**L2 — tracking a subject**

- [How L2 extends L1](#how-l2-extends-l1)
- [How related messages are identified](#how-related-messages-are-identified)
- [How priority is calculated and updated](#how-priority-is-calculated-and-updated)
- [How semantic retrieval works](#how-semantic-retrieval-works)
- [How the assistant answers, and when it refuses](#how-the-assistant-answers-and-when-it-refuses)
- [How privacy-aware routing works](#how-privacy-aware-routing-works)
- [What was optimised, and how it was benchmarked](#what-was-optimised-and-how-it-was-benchmarked)

**Both**

- [Running it](#running-it)
- [Project layout](#project-layout)
- [Assumptions](#assumptions)
- [Limitations](#limitations)
- [AI-tool usage declaration](#ai-tool-usage-declaration)

---

# L1 — reading a message

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

### At a glance

| L1 — reading a message | |
|---|---|
| Classification accuracy vs a hand-labelled gold set | **97.8%** (880 / 900) |
| Excluding three genuinely ambiguous templates | **100%** (870 / 870) |
| Classifier alone, on sentence forms it has never seen | **95.5%** |
| Sensitive messages detected and masked | 110 / 110 |

| L2 — tracking a subject | |
|---|---|
| Messages processed (900 L1 + 180 L2 + 24 demo) | 1,104 in **205 ms** |
| Subject groups | 63, covering 557 messages |
| Contradictions recorded | 20 groups, 11 of them contested |
| Retrieval, optimised vs baseline | **7.4× faster** at 1,029 docs, **19.5×** at 8,832 |
| …at what cost to ranking | none measurable — 99.8% of the exact top 10, identical recall |
| int8 embeddings vs float64 | **8× smaller** |
| Demonstration queries answered correctly | 8 / 8, including the one that must be refused |
| Tests | 260 |

Everything below is measured, and where a number is unflattering it is reported
rather than dropped.

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

# L2 — tracking a subject

## How L2 extends L1

L1 is a pipeline: read a row, mask it, classify it, extract from it, move on.
Every message is judged alone, and that is the right shape for the questions L1
was asked.

It is the wrong shape for L2's questions. "Which critical tasks are still
pending?" cannot be answered one message at a time, because whether a task is
still pending depends on a message that arrived three days later. So L2 keeps a
**ledger** ([`mint/ledger.py`](mint/ledger.py)): a running account of every
subject, updated message by message in chronological order.

The L1 pipeline still runs, first and unchanged:

```
read batch → mask → classify (L1) → extract (L1) → detect act →
assign to subject group → route for privacy → snapshot priorities →
(all batches done) → build the retrieval index
```

Five things are new, and each one is a module:

| Module | Question it answers |
|---|---|
| [`mint/subject.py`](mint/subject.py) | Are these two messages about the same thing? |
| [`mint/acts.py`](mint/acts.py) | What does this message *do* to the thing it names? |
| [`mint/groups.py`](mint/groups.py) | What is true about this subject now? |
| [`mint/priority.py`](mint/priority.py) | What should be dealt with first, and why? |
| [`mint/routing.py`](mint/routing.py) | What is this message allowed to be used for? |
| [`mint/embed.py`](mint/embed.py), [`mint/retrieve.py`](mint/retrieve.py) | Which messages match this query? |
| [`mint/assistant.py`](mint/assistant.py) | What is the answer, and is there enough evidence for one? |

Three properties of the L1 design carried straight into L2 and are worth naming,
because they are why the extension was cheap:

- **Masking still happens first.** The act detector, the grouper, the summariser
  and the search index all receive masked text because there is nothing else in
  the record to receive. No new redaction step was needed anywhere.
- **The record still has no field for raw text.** `LedgerRecord`, like
  `ProcessedMessage` before it, is structurally incapable of holding a secret.
- **L1's extraction is reused as evidence, not repeated.** When L1 decides a
  message is actionable and pulls a task out of it, `acts.from_l1_extraction`
  treats that as the act which *raised* the subject. That is the hinge between
  the two systems: without it, a subject's timeline would appear to begin with
  somebody chasing it, and its original deadline would have no source message.

### Batches, and why order is not sorting

The brief requires the L2 messages to be processed after the L1 messages. Rows
are sorted **within** a batch and batches are concatenated **in the order
given**. Merging everything and sorting by timestamp would usually produce the
same sequence, but "usually" is not a guarantee, and one out-of-order timestamp
would silently rewrite history.

### What it produces

On the 900 L1 messages plus the 180 L2 messages plus the 24 demonstration
messages — 1,104 in total, processed in about 205 ms:

| | |
|---|---|
| Subject groups | 63 |
| Messages belonging to a subject | 557 |
| Messages belonging to none | 547 (promotional, personal, general and credential traffic that names no tracked subject) |
| Status | 39 pending · 12 rescheduled · 6 completed · 6 cancelled |
| Priority | 7 critical · 13 high · 11 medium · 32 low |
| Groups with a recorded contradiction | 20, of which 11 are contested |
| Privacy routes | 1,004 local · 25 confirmation required · 75 blocked |
| Excluded from the search index | 75 — every blocked message |
| Mean group confidence | 0.937 |

---

## How related messages are identified

Part 2 of the brief, and the requirement that shapes it is this one: *messages
should not be grouped only because they contain one common word*.

The unit of identity is a **subject signature** — the content words that survive
once the conversational scaffolding is stripped away.

```
"Can you share an update on review the privacy checklist?"
"Please confirm whether you started to review the privacy checklist."
"The deadline to review the privacy checklist is now 2026-09-30."
                    ↓
              {review, privacy, checklist}
```

So does the L1 task whose extracted title was *"Review the privacy checklist"* —
which is how an L2 follow-up finds the L1 task it follows up on.

### Why not string equality, and why not word overlap

String equality fails because real follow-ups are lossy. *"Any progress on the
item concerning the assignment?"* never repeats the verb. *"Has the material for
our earlier model-results review been handled?"* reorders the words and
hyphenates them.

Plain word overlap fails in the other direction, and this corpus is full of the
failure: *"Review the privacy checklist"* and *"Review the model results"* share
**review**, and they are unrelated.

So a match has to qualify in one of exactly two ways, and the two are different
shapes of the same relation:

**(a) Sub-phrase.** Nearly all of the shorter signature's informative mass is
covered by the longer one — *"the assignment"* inside *"upload the assignment"*.
Formally, `shared_idf / min(mass_a, mass_b) ≥ 0.92`.

**(b) Restatement.** The two vocabularies are near-identical — *"model-results
review"* against *"review the model results"*. Formally, cosine `≥ 0.62` over
IDF-weighted tokens.

Anything in the band between them is two different subjects sharing a word or
two. *"Prepare the demo video"* and *"prepare the offline inference demo"* cover
two thirds of each other and are unrelated; admitting that band merged them, and
excluding it is what keeps them apart.

Two further guards:

- **A shared generic head is not a match.** *Session*, *meeting*, *task* describe
  the shape of an activity, never its identity. A signature with no specific
  token at all — *"the meeting"* — has no identity and joins nothing.
- **A single shared token must be rare enough to identify a subject alone.**
  *assignment* names one subject here and may carry a match by itself;
  *review* names eight and may not. The threshold is a share of the subject
  vocabulary, so it scales with the corpus instead of being a constant tuned to
  this one.

Weights are IDF over **distinct subjects**, not over messages: *review* appears
in a handful of subjects but hundreds of messages, and it is the subject count
that says how much identifying power it carries.

### Two passes, and why they are separate

1. Collect every subject signature and build the IDF statistics.
2. Walk the messages **in chronological order**, assigning each to a group and
   advancing that group's state machine.

Keeping them apart is what lets the matcher know *review* is a cheap word
without letting the future leak into the past. The statistics are a property of
the corpus; no message is ever influenced by the content of a later one.

### From messages to a status

Each message performs an **act** on the subject it names, detected by frames in
the same style as L1's rule layer — each frame captures the act and the subject
span in one pass, so the two can never disagree about which words were the
subject.

| Act | Example |
|---|---|
| `new_task` / `new_event` | *"New task: compare two embedding models by 2026-10-01."* |
| `status_query` | *"Can you share an update on review the privacy checklist?"* |
| `progress_update` | *"We have started work on the privacy checklist."* |
| `completion` | *"Update: finish the test cases has been completed successfully."* |
| `cancellation` | *"You can cancel call the service centre; it is no longer required."* |
| `reschedule` | *"The family dinner has been moved to 2026-09-29 at 10:00."* |
| `deadline_change` | *"The deadline to renew the library book is now 2026-10-01."* |
| `ambiguous_update` | *"The report might already be done, but I am not completely sure."* |
| `open_question` | *"Was the compliance form approved by the finance director?"* |
| `informational` | *"The office cafeteria menu has been updated."* |

Four rules govern the state machine, and they matter more than the transition
table:

1. **Only firm acts move the state.** A hedged sentence never produces a firm
   status. If nothing firm has ever been said the status is `unclear` — not
   `pending`, because those mean different things.
2. **Restatements do not re-assert.** *"Follow-up: X has been completed"* repeats
   an earlier claim and must not out-rank something said in between. A repeated
   act is absorbed when the group already carries it, and applied when it does
   not — a follow-up can still be the first time a fact arrives.
3. **Completion and cancellation are terminal.** Later chases and deadline edits
   do not silently reopen the item; they are recorded as contradictions, which
   is what they are.
4. **Contradictions are kept, not resolved away.** A subject reported both
   complete and cancelled comes out `contested`, with its confidence reduced and
   both messages cited. Averaging that disagreement into a clean-looking answer
   would be a lie.

### Deadlines and schedules

- A **deadline change** records `(from, to, direction, source message)`. Original
  requests that merely restated a subject with a different date are recorded
  separately from deliberate changes — restating is not changing, and conflating
  them would report nine conflicts per subject in this corpus.
- A **relative deadline** is never resolved into a date. *"The deadline is now
  tomorrow at 10 AM"* leaves `latest_deadline` at its last **absolute** value and
  sets `pending_relative_deadline` to the phrase, with its source. Both are
  reported. Inventing 2026-10-05 would usually be right, and "usually right" is
  the silent inference this system exists to avoid.
- A **schedule** is a merge, not an overwrite. *"The date stays the same, but the
  time is now 17:30"* keeps the date the earlier message set. The latest
  schedule is the result of applying every change in order.

### Worked example, from the real corpus

`GROUP_006 — Internship orientation`, 15 messages:

```
MSG_0011  2026-09-01  event raised, 2026-09-18 13:00
…         nine further L1 announcements, each with its own date
MSG_0953  2026-09-28  rescheduled → 2026-10-01 12:00
MSG_0955  2026-09-28  rescheduled → 2026-10-03 14:00
DEMO_007  2026-10-04  rescheduled → 2026-10-07 15:00
DEMO_009  2026-10-04  time only   → 17:30      (date preserved)
DEMO_017  2026-10-04  "we may move it; I will confirm later"  → hedged, ignored
                      status: rescheduled · latest: 2026-10-07 at 17:30
```

The last message is the interesting one. It is about the same meeting, it
proposes a change, and it changes nothing — because it does not assert anything.
It is recorded as a contradiction so the uncertainty is visible, and the schedule
stands.

### Summaries

Group summaries are generated from the group's own recorded history — every
clause is a fact something wrote down. No paraphrasing model is involved, and
none could be: the brief forbids sending message text to an external service,
and inventing narrative locally would be the same failure with a shorter network
path. A test asserts that no summary names a message that is not in its group.

---

## How priority is calculated and updated

Part 1 of the brief, and it is explicit that priority must not be "random or
only using one keyword". So this is a **weighted signal model** over 26 named
signals, each with a fixed weight, a family, and a sentence explaining what it
means. The score is their sum, the band is a threshold on the score, and the
reason attached to a decision is assembled from the signals that actually fired.

### Priority is a property of the subject, not of a sentence

*"Can you review the privacy checklist before 2026-09-09?"* is not urgent
because of anything in the sentence. It is urgent because that deadline has
passed and thirteen later messages are still chasing it. So the engine scores the
**group**, and every item extracted from that subject inherits the result — they
are the same work seen from different messages. A message read alone can only
ever be scored on what it says alone; items that could not be grouped are scored
that way and flagged for review.

### The signals

| Family | Signals |
|---|---|
| Deadline | `deadline_overdue` +3.2 · `deadline_today` +2.4 · `deadline_tomorrow` +2.0 · `deadline_within_3_days` +1.4 · `deadline_within_7_days` +0.7 · `deadline_beyond_7_days` −0.5 · `deadline_relative_imminent` +2.0 · `deadline_moved_earlier` +0.8 · `deadline_moved_later` −0.4 · `deadline_conflict` +0.4 · `deadline_unresolved` · `no_deadline` |
| Urgency | `explicit_urgency` +1.8 · `de_escalated` −1.6 |
| Pressure | `chased_repeatedly` +1.4 · `chased_once` +0.4 · `response_required` +0.5 · `chased_after_closure` +0.6 |
| Sender | `authority_sender` +0.5 |
| Sensitivity | `sensitive_high_risk` +0.4 |
| Lifecycle | `status_completed` −6.0 · `status_cancelled` −6.0 · `status_unclear` −0.2 · `status_contested` +0.3 |
| Category | `category_action_required` +0.4 · `category_meeting_event` +0.25 |

Bands: **critical** ≥ 4.6, **high** ≥ 3.0, **medium** ≥ 1.5, otherwise **low**.

Two details are load-bearing:

- **`deadline_relative_imminent` is worth less than an absolute date.** "Due
  tomorrow" is a stated proximity that was never resolved to a date, and the
  confidence attached to the decision is reduced to say so. Using the stated
  proximity is not the same as inventing a date, and the stored deadline stays
  null either way.
- **Closed work scores −6.0**, which is enough to sink any combination of
  deadline and pressure signals. A completed task is not a priority question.

### Critical needs two kinds of evidence

A score can clear 4.6 on deadline signals alone. When it does, the band is held
at **high** and the decision records `gated_from_critical` with a reason. One
very loud signal is not enough to tell a human to drop everything, and this is
what the brief's "not from one keyword" rule means operationally.

### "Now" is the newest message, not the wall clock

Deadline proximity needs a reference point. Using the real current time would
make the same input produce different output on different days — impossible to
check and impossible to demonstrate. The reference is the timestamp of the last
message processed. Load another batch and every deadline moves closer, which is
exactly the update behaviour the brief asks for, and it is reproducible.

A consequence worth stating: **priority can change with no new message at all**,
because time advanced past a deadline. The engine records which of the two
happened, since "this became critical because someone escalated it" and "this
became critical because you ignored it" are different facts and want different
responses.

### How an update looks

A snapshot is taken at every batch boundary. From the real corpus:

```
batch 1  (900 L1 messages, as of 2026-09-24)   no changes yet — first assessment
batch 2  (180 L2 messages, as of 2026-10-02)   28 changes
   GROUP_010  Confirm the interview slot     high → critical   MSG_0906, MSG_0972, …
   GROUP_022  Call the service centre        high → low        completed in MSG_0939
   GROUP_012  Update the project tracker     high → low        cancelled in MSG_0942
batch 3  (24 demo messages, as of 2026-10-05)  12 changes
   GROUP_010  Confirm the interview slot     critical, score 4.7 → 9.3   DEMO_001
   GROUP_013  Upload the assignment          critical → low    DEMO_020 extended it
   GROUP_016  Study-group session            medium → high     time passed
```

Band crossings are not the only change worth reporting. A subject going from
*just inside* critical to *far inside* it has genuinely escalated, so a score
movement of 1.5 or more is recorded even when the band holds — which is what
makes "which task became critical in the demo data?" answerable with
`GROUP_010, 4.7 → 9.3, caused by DEMO_001`.

---

## How semantic retrieval works

Retrieval has to find messages that mean the same thing in different words.
Lexical overlap gets a long way — *"model-results review"* and *"review the
model results"* share three tokens — but it fails exactly where a real user
types a paraphrase.

Two mechanisms carry the meaning, and the first does most of the work:

**1. Subject-signature matching** (above). This is what resolves *"the
assignment"*, *"the model results"* and *"the material for our earlier
model-results review"* onto the right subjects. It is morphology-folded,
IDF-weighted and containment-tested, and it is how the assistant turns a phrase
in a question into a group.

**2. Latent semantic embeddings.** Each message also gets a dense vector from a
truncated SVD over the corpus TF-IDF matrix. Words bridge not by dictionary
lookup but because the corpus used them the same way.

```
1,348-term vocabulary → 128 components (89.6% of the variance)
→ 32 largest components kept per term  (cosine 0.970 against the dense projection)
→ int8, one scale per component        (cosine 0.970 — quantisation costs nothing)
→ models/semantic.json, 412 KB
```

The artifact is n-gram keys, IDF weights and integer loadings. It contains no
message text, and a test asserts that.

### Why LSA and not a sentence transformer

Honestly: a MiniLM-class encoder would embed better. It is not used here, and
the reasons are constraints rather than a claim that LSA is superior.

- The brief forbids sending message text to an external service, so the encoder
  would have to run locally inside a serverless function — PyTorch or ONNX
  Runtime plus ~90 MB of weights, to answer queries over 1,100 short messages.
- The forward pass here is forty lines of standard-library Python, and the
  exported model is 412 KB of integers that can be read and checked. For a
  system whose main claim is that nothing leaves the machine, being able to
  *verify* that by reading a file is worth something.

The limitation this buys is real, is measured below, and is not hidden: LSA
cannot relate two words the corpus never used together, so a genuinely novel
synonym does not bridge.

### The semantic layer is a recall device, not a scoring device

This is the most interesting thing the benchmark found, and it changed the
design.

The first version blended lexical and semantic scores on every query. Measured
against exact lexical scoring it was **slower** (0.372 ms against 0.386 ms
median — it had thrown away the inverted index's advantage), reproduced **less**
of the exact ranking (93.1% of the top 10, against 99.8%), and was **worse** on
hand-written paraphrases (0.792 against 0.850 recall@10).

The reason is a property of this corpus rather than of LSA. Every subject here is
phrased with a consistent vocabulary, so lexical overlap is already close to
exact and there is almost no synonymy for a latent space to bridge — it can only
blur a signal that is already sharp.

So the query embedding is **not computed at all** unless lexical candidate
generation cannot fill the result set. When it is, the semantic score breaks ties
and ranks the candidates lexical scoring never proposed. Queries the lexical
stage can answer get exactly the ranking they would have got with no semantic
layer, and pay nothing for its existence.

The rejected design is kept behind a flag and measured in the report, so the
decision can be checked rather than taken on trust.

---

## How the assistant answers, and when it refuses

Part 3. The central decision is that **retrieval is not the answer path.**

Most of the brief's questions are not "find me a document". *"Which critical or
high-priority tasks are still pending?"* is a filter over the priority table.
*"What meetings were rescheduled?"* is a filter over group status. *"What
deadlines have changed?"* is a read of the deadline history. Answering those by
ranking text would be a worse system that fails in a more interesting way: it
would return plausible messages and get the count wrong.

So a question is routed to an **intent** first, answered exactly from the ledger
where one applies, and handed to hybrid retrieval only when none does. Retrieval
runs alongside every answer regardless, so a relevance score is always shown
next to the structured evidence for the reader to judge.

| Intent | Answers questions like |
|---|---|
| `due_today` | What tasks should I complete today? |
| `pending_by_priority` | Which critical or high-priority tasks are still pending? |
| `by_status` | Which tasks have been completed or cancelled? |
| `rescheduled` | Which meeting was rescheduled and what is its latest schedule? |
| `latest_status` | What is the latest status of the task referenced by DEMO_016? |
| `deadline_changes` | What deadlines have changed? |
| `conflicts` | Which messages contain conflicting or uncertain deadlines? |
| `became_priority` | Which existing task became critical in the demo data? |
| `explain_priority` | Why was this item marked as critical? |
| `blocked_messages` | Which messages must be blocked from external processing? |
| `needs_confirmation` | Which messages require confirmation? |
| `messages_about` | Show all messages related to the project report. |
| `yes_no_verification` | Was the compliance form approved by the finance director? |
| `open_search` | anything else |

Every answer carries: the answer, supporting message IDs, related group and item
IDs, a relevance score per piece of evidence, a sentence on **why that evidence
was selected**, the privacy route it inherited, and a confidence.

### Refusal is a first-class outcome

*"The assistant must not generate an unsupported answer"* is a requirement, so
there is a gate rather than a hope.

The interesting case is `yes_no_verification`. A question of the form *"was X
approved?"* can only be answered by a message that **asserts** an outcome. So the
test is assertion, not similarity:

> **Q.** Was the compliance form approved by the finance director?
>
> **A.** There is not enough evidence to answer this. `GROUP_063` (*Was the
> compliance form approved by the finance director*) collects 1 message on this
> subject and none of them asserts an outcome — nothing reports it done,
> cancelled, rescheduled or given a new deadline. `DEMO_022` raises the subject
> and nothing in the corpus replies.

That question has a near-perfect lexical match in this corpus, because another
message asks exactly the same thing. Returning it would be the most confident
possible way to be wrong.

Answers are also refused when nothing scores above the evidence threshold, and
when a referent cannot be resolved. A resolution that rests on a single shared
word is declared in the answer and its confidence reduced.

---

## How privacy-aware routing works

L1 answered *"is there something sensitive here, and what should be done about
it?"*. L2 has to **act** on that answer, because L2 does things L1 never did: it
builds a search index, retrieves evidence, and composes answers out of messages
the user did not name. Each of those is a new way for a value to travel.

Every message gets a **route**, derived from the recommendation L1 already
produced — so the two layers cannot drift apart, and a sensitivity type added to
`mint/sensitive.py` inherits the route of its action automatically.

| Policy | Route | Applies to | Allowed | Denied |
|---|---|---|---|---|
| `P0-DEFAULT` | local only | nothing sensitive found | classify, extract, index, store masked, quote, export | store raw, send external |
| `P1-CREDENTIAL` | **blocked** | OTP, password, auth token, recovery code, payment card, bank account | classify, store masked | extract, index, quote, export, store raw, send external |
| `P2-IDENTITY` | local only | government ID, health information | classify, extract, index, store masked, quote | export, store raw, send external |
| `P3-CONTACT` | **confirm required** | postal address, phone number | classify, extract, index, store masked | quote, export, store raw, send external |
| `P4-MENTION` | local only | a reference to credentials with no value | everything local | store raw, send external |

Three properties make this more than a label:

**The index is filtered, not the results.** Blocked messages are never added to
the retrieval index, so no query can rank one and no ranking bug can leak one.
Filtering after retrieval would mean the value had already been loaded, compared
and scored before something remembered to hide it. A test searches the index for
the planted secrets themselves and asserts nothing comes back.

**Blocked evidence is reported as existing.** *"There are three messages here I
will not show you, and here is what kind they are"* is a useful answer;
pretending they do not exist is not. The assistant names them by ID and
sensitivity type and withholds the content.

**Confirmation is a real gate, not a warning.** Private addresses and phone
numbers are often exactly what a user meant to search for, so refusing outright
would be theatre. The message stays searchable in masked form; quoting or
exporting it needs an explicit confirmation, which the API and UI both carry.

**The external route does not exist.** `send_to_external_service` is denied for
every message including entirely harmless ones — not because the policy says so,
but because this package contains no network client and no outbound call. The
policy records the fact; it is not what enforces it.

The L2 corpus also disclosed a medical condition and a delivery address through
phrasings none of L1's anchors covered (*"My private medical note mentions…"*,
*"Deliver it to 17 River Park Street…"*), so two detectors gained anchors. L1
accuracy against the gold set is unchanged at 97.8%.

---

## What was optimised, and how it was benchmarked

**The optimised component is the retrieval index.** Both versions are in the
repository and the benchmark measures running code on both sides, rather than
comparing against a description of the one that was replaced.

| | |
|---|---|
| `v1-exact-lexical` | TF-IDF cosine scored against every document. No approximation. Also the reference ranking. |
| `v2-hybrid-pruned` | Pruned inverted index for candidate generation, exact rerank of the top 200, int8 LSA embeddings as a recall fallback. **Shipped.** |
| `v2-lexical-only` | v2 with the semantic layer off, so the inverted index and the embeddings can be credited separately. |
| `v2-semantic-always` | The rejected design: semantic scores blended into every query. |

### Method

Every configuration answers the **same** queries over the **same** documents in
the same process, after a discarded warm-up pass, timed with
`time.perf_counter` per call. Median and p95 are reported rather than a mean,
because a mean over a handful of runs mostly measures the slowest outlier.

Latency is measured at two corpus sizes. The larger is the corpus replicated
with distinct IDs — a valid way to measure how two designs diverge with scale,
and **not** used for any quality number, where duplicating documents would make
every metric meaningless.

Quality uses relevance labels derived from the subject groups: a message is
relevant to a query if it belongs to the group that query is about. It is worth
being explicit about what that tests — whether retrieval finds the messages the
grouper considers related. It does not independently verify the grouping.

Query sets are kept separate on purpose:

- **Literal** (40) — the subject's own words.
- **Paraphrase, in vocabulary** (10) — written by hand to avoid the subject's
  wording, using words the model has seen.
- **Paraphrase, out of vocabulary** (4) — *"settle the utility payment"*,
  *"catch up with my adviser"*, *"ring the support helpline"*, *"sign off the
  timesheet"*. None of those content words occurs in the corpus at all.

Averaging the last two would hide the finding, so they are reported separately.

**Testing device.** Apple silicon (arm64), macOS 26.5, CPython 3.12.13, single
process, no parallelism. Absolute numbers are specific to this machine; the
ratios are the comparable part.

### Results

*1,104 messages ingested end to end in 205 ms; 1,029 indexed, 75 withheld by
privacy routing. 42 queries × 30 repeats.*

| Configuration | Build | Median | p95 | Memory |
|---|---|---|---|---|
| `v1-exact-lexical` | 10 ms | 0.386 ms | 0.547 ms | 2,276 KB |
| **`v2-hybrid-pruned`** | 64 ms | **0.053 ms** | 0.521 ms | 3,329 KB |
| `v2-lexical-only` | 11 ms | 0.046 ms | 0.135 ms | 2,950 KB |
| `v2-semantic-always` | 64 ms | 0.372 ms | 1.139 ms | 3,329 KB |

| | 1,029 docs | 8,832 docs |
|---|---|---|
| `v1-exact-lexical` median | 0.386 ms | 4.127 ms |
| `v2-hybrid-pruned` median | 0.053 ms | 0.212 ms |
| **Speed-up** | **7.4×** | **19.5×** |

The gap widens with corpus size because that is what an inverted index does.
Quoting only the small number would understate it; quoting only the large one
would overstate what this corpus proves, so both are reported.

**Quality — recall@10**

| Configuration | Literal | Paraphrase (in vocab) | Paraphrase (OOV) | Top-10 vs exact |
|---|---|---|---|---|
| `v1-exact-lexical` | 1.000 | 0.850 | 0.500 | — |
| **`v2-hybrid-pruned`** | 1.000 | 0.850 | 0.500 | **99.8%** |
| `v2-lexical-only` | 1.000 | 0.850 | 0.500 | 97.9% |
| `v2-semantic-always` | 1.000 | 0.850 | 0.450 | 93.1% |

The shipped index is **7.4× faster with no measurable quality cost**, and
reproduces 99.8% of the exact index's top 10 with 100% top-1 agreement.

**Size**

| | |
|---|---|
| int8 embeddings | 129 KB (1,053 KB as float64 — **8× smaller**) |
| Posting lists | 674 KB |
| Semantic model artifact | 412 KB (1.4 MB as a dense float projection) |
| v2 in memory vs v1 | **1.46×** |

v2 uses *more* memory, and that is the trade: an inverted index and a set of
embeddings on top of the same document vectors. Reporting a speed-up while
hiding the memory cost would be picking the favourable metric.

### Two size optimisations that were nominal until they were measured

Both were found by measuring rather than reasoning, and both are the kind of
thing that makes "we quantised to int8" untrue in practice:

- **int8 codes in a Python list cost ~4 KB per document** once the integer
  objects are counted — *more* than the float vector they replaced. In an
  `array('b')` they cost 128 bytes. 4.3 MB became 129 KB.
- **A `(doc_id, weight)` posting tuple costs ~80 bytes.** Two parallel arrays
  cost 8. Across ~20,000 postings that is 1.6 MB against 160 KB, and the
  traversal is unchanged.

### Two correctness bugs the benchmark surfaced

- **Document-frequency pruning by share alone** is right at 1,000 documents and
  catastrophic at seven — a term in two of seven is already over the share, so
  almost the whole vocabulary was pruned and lexical-only search returned
  nothing. Pruning now also requires an absolute document count.
- **Candidate generation had no correctness guard.** If every query term was
  pruned or unknown there were no posting lists to walk and the result was
  empty, which made the optimised index quietly *less correct* than the one it
  replaced. It now falls back to the exact scan — which also raised ranking
  fidelity from 97.9% to 99.8%.

The full report, including per-configuration detail and the scaling run, is
[`outputs/benchmark_report.json`](outputs/benchmark_report.json). The hosted demo
can reproduce the comparison live on whatever corpus is loaded.

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

# 3. Run the L1 pipeline                     -> outputs/classified_*, extracted_*, sensitive_*
python scripts/run_pipeline.py --evaluate --ids data/mandatory_demo_ids.csv

# --- L2 ------------------------------------------------------------------

# 4. Build the semantic projection            -> models/semantic.json
python scripts/build_index.py

# 5. Run the L2 ledger                        -> outputs/priority_*, message_groups_*, privacy_*
python scripts/run_l2.py \
    --input data/messages.csv data/l2_messages.csv \
    --demo  data/l2_demo_messages.csv \
    --queries data/l2_demo_queries.csv

# 6. Benchmark baseline against optimised     -> outputs/benchmark_report.json
python scripts/benchmark.py \
    --input data/messages.csv data/l2_messages.csv \
    --demo data/l2_demo_messages.csv --queries data/l2_demo_queries.csv

# 7. Tests
python -m pytest

# 8. Web app  —  /  is the L1 view,  /l2  is the L2 view
uvicorn api.index:app --reload --port 8000
```

`--input` files are processed **in the order given** and `--demo` is appended
last. That ordering is the point: the brief requires the L2 messages to be
processed after the L1 messages.

Generated outputs (all containing masked text only):

| File | Contents |
|---|---|
| `outputs/classified_messages.json` | Part 1 — category, confidence, reason, full evidence trail |
| `outputs/extracted_items.json` | Part 2 — tasks and events with unresolved fields marked |
| `outputs/sensitive_report.json` | Part 3 — type, risk, masked text, recommended action |
| `outputs/summary.json` | Aggregate counts |
| `outputs/classified_messages.csv` | Flat view of the above |
| `outputs/priority_decisions.json` | **L2 Part 1** — priority, reason, signals, confidence, history, per item |
| `outputs/message_groups.json` | **L2 Part 2** — related message IDs, item IDs, status, latest deadline, summary, timeline, conflicts |
| `outputs/privacy_routing.json` | **L2 privacy** — the policy table and a route per message |
| `outputs/priority_snapshots.json` | Priority as it stood at each batch boundary, with what changed and why |
| `outputs/assistant_answers.json` | The supplied demonstration queries, answered with evidence |
| `outputs/benchmark_report.json` | **Benchmark comparison** — latency, size and quality for all four index configurations |
| `outputs/message_groups.csv`, `outputs/priority_decisions.csv` | Flat views, easier to read on screen |

### HTTP API

| Route | Purpose |
|---|---|
| `GET /` · `GET /l2` | the L1 and L2 views |
| `GET /v1/analyze/sample` · `POST /v1/analyze` · `POST /v1/classify` | L1 |
| `GET /v2/analyze/sample` | run the bundled synthetic L2 sample |
| `POST /v2/analyze` | analyse one or more CSV batches, **in the order supplied** |
| `POST /v2/ask` | answer a question against an analysed corpus |
| `POST /v2/benchmark` | measure baseline against optimised, live, at two corpus sizes |
| `GET /v2/policy` | the privacy-routing policy table |
| `DELETE /v2/session/{id}` | drop an analysed corpus from memory immediately |

L2 answers questions *about a corpus*, so a query needs the analysed corpus to
still be there when it arrives, and a stateless function has to hold it
somewhere. What is held is deliberately narrow:

- **The raw CSV text is never retained.** Masking runs during ingest and the
  ledger that survives contains masked text only. There is nothing in the cache
  a request could not already see.
- **The cache is memory only, bounded to three corpora, and content-addressed**
  by a hash of the uploaded bytes, so re-uploading the same file reuses one
  session instead of accumulating copies. Nothing is written to disk.
- **Adding a batch means re-analysing from scratch.** There is no incremental
  append path that would require keeping the earlier raw text around. Re-doing
  1,100 messages costs ~205 ms, which is a good price for not storing them.
- **A cache miss is not the user's problem.** A cold instance returns 409 and the
  browser re-uploads the file it still holds.

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

The same applies to L2: `mint/embed.py` reproduces the SVD projection in
standard-library Python, so `models/semantic.json` — 412 KB of integers — is all
the runtime needs, and scikit-learn stays a training-time dependency.

`.vercelignore` excludes `data/`, `outputs/` and `eval/`, so the datasets and
everything derived from them never reach the hosting platform. The deployed demo
ships only the self-authored `sample_data/` files.

---

## Project layout

```
mint/                     the system (no network calls anywhere)
  ── L1 ──
  sensitive.py            detection + masking — the only module that sees raw text
  rules.py                linguistic frames; weak-supervision label source
  model.py                pure-Python TF-IDF + logistic-regression inference
  classifier.py           arbitration between detector, rules and model
  extract.py              task/event extraction with strict null handling
  pipeline.py             L1 orchestration; mask-first ordering
  normalize.py            opener stripping, tokenisation
  taxonomy.py             the six categories and the reasoning behind them
  ── L2 ──
  subject.py              subject identity: IDF-weighted containment matching
  acts.py                 speech acts: what a message does to the subject it names
  groups.py               related-message grouping and the lifecycle state machine
  priority.py             the 26-signal weighted priority model
  routing.py              privacy routes: what each message may be used for
  embed.py                local LSA embeddings, sparse + int8, pure-Python forward pass
  retrieve.py             both retrieval indexes — baseline and optimised
  assistant.py            intent routing over the ledger, with refusal
  ledger.py               L2 orchestration; batch order, snapshots, output files
api/index.py              FastAPI service (thin — all logic lives in mint/)
web/index.html            the L1 view          ·  web/l2.html   the L2 view
web/app.css               one stylesheet, shared by both
scripts/                  make_gold.py, train.py, run_pipeline.py,
                          build_index.py, run_l2.py, benchmark.py
tests/                    260 tests
eval/gold_labels.csv      hand labels — IDs and categories only, no message text
models/classifier.json    exported L1 classifier
models/semantic.json      exported LSA projection — n-gram keys and integers only
sample_data/              synthetic samples written for the public demo
data/, outputs/           gitignored — the datasets and their derivatives
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

**L2 adds five more.**

7. **Batch order is authoritative, timestamps are not.** Batches are
   concatenated in the order supplied and sorted only within themselves. One
   out-of-order timestamp must not be able to rewrite history.
8. **"Now" is the newest message in the corpus.** Every deadline-proximity
   judgement is relative to it, never to the wall clock, so the same input
   always produces the same output.
9. **Items extracted from one subject share that subject's priority.** They are
   the same work seen from different messages. An item whose subject could not
   be grouped is scored alone and flagged for review.
10. **IDF over subjects, not messages.** A word's identifying power is how many
    distinct subjects use it, not how many messages contain it.
11. **A relative deadline is a stated proximity, not a date.** "Due tomorrow"
    may raise a priority; it may not become a stored date, and the confidence
    attached to the decision is reduced to say so.

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
another produces two unlinked items — *which is what L2 was built to fix.*

### L2 limitations

**Out-of-vocabulary paraphrases retrieve nothing.** *"Settle the utility
payment"* does not find *"pay the electricity bill"* under any configuration,
because none of *settle*, *utility* or *payment* occurs in the corpus, so the
query has no lexical overlap and its latent vector is close to zero. This is the
measured limit of LSA on a small domain corpus (recall@10 = 0.500 on the
out-of-vocabulary set against 0.850 in-vocabulary), and it is the clearest case
for a subword or pretrained encoder.

**The semantic layer earns nothing as a scorer here.** Measured, reported, and
the reason the design changed — but the corpus is why. Every subject is phrased
consistently, so there is little synonymy to bridge. On genuinely free-form
messages I would expect the opposite result, and the flag to turn it back on is
already there.

**A one-word reference is genuinely ambiguous.** *"The report might already be
done"* attaches to *"Submit the weekly report"* on the single shared token
*report*. That may be right, and the system cannot know: it records the link as
weak, lowers the group's confidence, and says so in the answer. The alternative —
refusing to link it — loses a real follow-up.

**"In progress" is reachable but never reached.** No message in the supplied
corpus asserts progress; the corpus only ever *asks* whether something is in
progress, which is a status chase and not a status. The frames for a genuine
assertion exist so the value is not one the schema can express and the code can
never produce, but on this data the count is zero.

**Priority weights are hand-set.** The 26 signals and their weights are
principled and each one is justified, but they are not fitted — there is no
labelled priority ground truth to fit them to. The corroboration gate is a guard
against the worst failure mode, not a substitute for calibration.

**Group relevance labels are derived, not annotated.** The retrieval quality
numbers use "belongs to the same subject group" as the relevance criterion,
which measures whether retrieval finds what the grouper considers related. It
does not independently verify the grouping. A hand-annotated relevance set would.

**The demonstration corpus is generated.** The L2 messages come from a handful
of templates per lifecycle event. The frames match them well, which is partly a
measure of the frames and partly a measure of the generator; free-form
follow-ups would fall through to the model and the residual acts more often.

### Possible improvements

- **A local subword encoder** (MiniLM or a distilled equivalent, run offline) in
  place of LSA. This is the single change that would move the out-of-vocabulary
  number, and it is a deployment-size decision rather than a research one.
- **Character n-gram features** as a cheaper half-step: they would bridge
  morphology the stemmer misses without shipping a neural model.
- **Fitted priority weights**, once a labelled triage set exists. The signals are
  already the right features; only the coefficients are guessed.
- **A hand-annotated retrieval relevance set**, so quality is measured against
  human judgement rather than against the grouper's own output.
- **Coreference across subjects** — "it", "the same thing", "that meeting" — which
  currently resolve only when a content word repeats.
- Active learning: the `needs_review` queue is already the ideal labelling
  batch — every current error is in it.
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
  both web views, the test suite and this README — in L1 and again in L2.
- Designing the evaluation methodology, including the grouped-by-template
  cross-validation, the review-flag coverage metric, and L2's separation of
  literal, in-vocabulary and out-of-vocabulary query sets.

**What it was *not* used for:**
- **No message text was ever sent to any AI service for classification.** The
  system contains no API calls. Classification, extraction and detection are
  performed entirely by the local regex frames and the local logistic-regression
  model in this repository.
- No labels were produced by an LLM. The 115 template judgements in
  `scripts/make_gold.py` are my own, as are the 14 hand-written paraphrase
  queries in `scripts/benchmark.py`.
- **No LLM is anywhere in the inference path, at either level.** The trained
  classifier is a scikit-learn logistic regression fitted on this corpus; the
  semantic model is a truncated SVD over its TF-IDF matrix. Grouping, priority,
  routing and the assistant's answers are deterministic code. Group summaries
  and answer text are assembled from recorded facts by template — there is no
  generative step, and a test asserts that no summary names a message outside
  its own group.

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
kept separate from `date`, why fixed-width masking matters; and in L2, why
matching needs a containment test as well as a cosine, why restatements are
absorbed rather than re-asserted, why "now" is the newest message rather than
the wall clock, why critical requires two families of evidence, why blocked
messages are excluded from the index rather than filtered from results, and why
the semantic layer ended up as a fallback instead of a scorer. The reasoning is
documented inline throughout the source, not only in this file.

**Where the AI tool was most and least useful, honestly.** It was fastest at the
parts with a clear specification — frame grammars, serialisation, the web views,
test scaffolding. It was least reliable exactly where this project needed
judgement: the first hybrid retrieval design it produced was slower *and* worse
than the baseline, and only measuring it revealed that. Several of the decisions
I am most confident in — the containment test, the corroboration gate, the
assertion test behind refusal — came from looking at outputs that were wrong and
working out why. The benchmark exists because I did not trust the design, and it
turned out I was right not to.

**Other libraries:** scikit-learn and NumPy (training only — the classifier fit
and the SVD), FastAPI and python-multipart (serving), pytest (tests). No
pre-trained language models, no vector database, no external inference APIs.
