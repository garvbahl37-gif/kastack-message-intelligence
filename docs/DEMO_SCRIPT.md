# Video demonstration script (7–10 minutes)

Every item the brief lists as required is mapped to an exact action below.
Tick each box as you record — anything not shown may be treated as incomplete.

**Before you start**

```bash
cd /Users/garvbahl/Documents/Projects/KaStack
source .venv/bin/activate
python scripts/run_pipeline.py --evaluate --ids data/mandatory_demo_ids.csv > /tmp/run.txt
uvicorn api.index:app --port 8000        # leave running in a second terminal
```

Open two windows: **terminal** and **browser at http://localhost:8000**.

> **Which host to demo on.** Use **localhost** for anything involving
> `data/messages.csv`. It is the same code and the same UI as the cloud demo,
> and it keeps the supplied dataset off the network entirely. Show the cloud URL
> separately (§1 and §11) to prove the hosted demo is live.

> **Masking is automatic.** Every view renders `******` — there is no screen in
> this system that can display a raw secret. You do not need to blur anything.
> Just don't open `data/messages.csv` itself in an editor on camera.

---

## 0 · Opening (0:00–0:45)

- [ ] "This is a local message-intelligence system. It classifies 900 messages
      into six categories, extracts tasks and events without inventing missing
      information, and detects and masks sensitive values. Nothing is sent to
      any external AI service — there are no network calls in the package."
- [ ] Show the repo tree briefly (`ls mint/`).

**Say the architecture in one breath:**
> Three mechanisms with deliberately different authority. A deterministic
> detector that *vetoes* — because deciding whether to redact something should
> never be a probability. High-precision rule frames that abstain when unsure.
> And a logistic-regression model trained on masked text that always has an
> opinion. Where the last two agree, confidence goes up; where they disagree,
> it goes down and the message gets flagged.

---

## 1 · System flow + cloud demo is live (0:45–1:30)

- [ ] Browser → **https://kastack-message-intelligence.vercel.app**
- [ ] Point out the header: "runs locally · nothing sent out", and the bundled
      synthetic sample. Say: *"The assessment dataset is not in the repo and not
      on the host. This sample is written from scratch for the public demo."*
- [ ] Switch to **http://localhost:8000** for the rest of the demo.

---

## 2 · Dataset structure (1:30–2:00)

- [ ] Terminal: `head -1 data/messages.csv` — show only the header row
      (`message_id, timestamp, sender, message`).
- [ ] `wc -l data/messages.csv` → 901 lines = 900 messages.
- [ ] **Do not** cat the file. Instead click **Upload CSV** in the browser and
      load `data/messages.csv`.
- [ ] Header now reads `messages.csv · 900 messages · in memory only`.

Say: *"Rows are sorted by timestamp before anything else runs, so processing is
chronological regardless of file order."*

---

## 3 · All six categories (2:00–2:40)

- [ ] **Overview** tab. Read the distribution aloud:

| Category | Count |
|---|---|
| action_required | 250 |
| meeting_or_event | 170 |
| general_information | 160 |
| personal_information | 110 |
| promotional | 110 |
| sensitive_information | 100 |

- [ ] **Messages** tab → use the category dropdown to show one example of each
      of the six. That visibly demonstrates all six categories.

---

## 4 · The 15 mandatory message IDs (2:40–3:40) ⚠️ required

- [ ] **Required IDs** tab. Paste all fifteen (this is exactly the set in
      `mandatory_demo_ids.csv`, sorted for legibility on screen):

```
MSG_0001, MSG_0002, MSG_0003, MSG_0004, MSG_0005, MSG_0006, MSG_0007, MSG_0009,
MSG_0012, MSG_0013, MSG_0014, MSG_0015, MSG_0016, MSG_0024, MSG_0037
```

- [ ] Click **Show these**. Scroll slowly through all 15 cards so each ID is
      legible on screen.
- [ ] Call out that MSG_0005 and MSG_0013 render as redaction bars.

Terminal alternative (also fine, shows the same 15):
```bash
sed -n '/MANDATORY/,$p' /tmp/run.txt | head -80
```

---

## 5 · Three correctly extracted tasks (3:40–4:10) ⚠️ required

- [ ] **Tasks & events** tab → filter **Tasks**.

| Source | Title | Deadline | Priority |
|---|---|---|---|
| MSG_0002 | Review the privacy checklist | 2026-09-09 | low — 8 days away |
| MSG_0007 | Reply to the client email | 2026-09-04 | high — 3 days after the message |
| MSG_0010 | Pay the electricity bill | 2026-09-09 | low — 8 days away |

- [ ] Say: *"Priority is derived from deadline proximity and always carries the
      reason it was assigned."*

---

## 6 · Three correctly extracted meetings/events (4:10–4:40) ⚠️ required

- [ ] Filter **Events**.

| Source | Title | Date | Time | Location |
|---|---|---|---|---|
| MSG_0001 | Family dinner | 2026-09-19 | 10:00 | Library |
| MSG_0003 | Mentor catch-up | 2026-09-16 | 11:00 | City clinic |
| MSG_0011 | Internship orientation | 2026-09-18 | 13:00 | Conference Room 2 |

---

## 7 · Missing / unclear information (4:40–5:10) ⚠️ required

- [ ] Toggle **Unresolved fields only**.
- [ ] Open **MSG_0037** — *"The review could be Friday afternoon."*

```json
{ "title": "Review", "date": null, "date_raw": "Friday afternoon",
  "date_status": "unresolved_relative", "time": null, "person": null,
  "unresolved_fields": ["date", "time", "person"] }
```

**Say this — it is the heart of Part 2:**
> The date field is null. We kept the phrase "Friday afternoon" verbatim and
> marked it unresolved. We could guess a date from the message timestamp — and
> we'd often be right — but "often right" is exactly the silent inference this
> system is built to avoid. Where a relative phrase *is* resolvable, like
> "tomorrow", the result goes in a separate `date_suggestion` field and is never
> promoted into `date`. A suggestion is for a human; `date` is an assertion.

---

## 8 · Sensitive detection, masking, risk, action (5:10–6:10) ⚠️ required

- [ ] **Sensitive** tab. 110 findings across 11 types.
- [ ] Show the columns: type, risk, masked message, recommended action, why.
- [ ] Filter by **High** risk (80 messages).
- [ ] **Hover a redaction bar** — a tooltip names the type but never the value.
      Say: *"The text node is removed entirely. There is nothing under that bar
      to select, copy, or read out of the DOM."*
- [ ] Point at a `do_not_store` row vs a `ask_for_confirmation` row and explain
      the difference: live credentials have no upside to retaining; a postal
      address may legitimately need local retention.
- [ ] Show **MSG_0012** — *"I will send the login details separately."*
      Detected as `credential_mention`, **low** risk, `safe_to_process_locally`.
      Say: *"It mentions credentials but contains none, so there's nothing to
      redact and nothing to protect. Distinguishing 'mentions a secret' from
      'contains a secret' is the whole point of that type — and it's why this
      one is low risk while everything above it is high."*

      Note: this message is also one of the system's known errors — see §10.
      Its **category** is `action_required` at 0.43 and flagged for review,
      whereas I labelled it `general_information`. Mentioning that here is
      optional; §10 covers it.

---

## 9 · Three classification decisions with explanations (6:10–7:00) ⚠️ required

Go to **Messages** and click each row to open the evidence drawer.

- [ ] **MSG_0002** — "Can you review the privacy checklist before 2026-09-09?"
      → `action_required` **0.99**. Rule frames `request_question: can you
      review` + `date_bounded: before 2026-09-09` (0.96); model agreed at 0.95.
      *Two independent methods agreeing is real evidence, so confidence gets a
      bounded bonus — but it's capped at 0.99, never 1.0.*

- [ ] **MSG_0001** — "Calendar update: family dinner, 2026-09-19 at 10:00…"
      → `meeting_or_event` **0.98**, frame `calendar_marker: calendar update:`
      (0.90); model 0.94.

- [ ] **MSG_0014** — "Special festival discount on clothing. Use code SAVE17."
      → `promotional` **0.99**, three corroborating signals: `coupon_code`,
      `discount_offer`, and `promotional_sender` (sender = Promotions).
      *Promotional is checked first, because marketing copy borrows the grammar
      of every other category — "Join our premium plan" looks like an
      invitation until you see the coupon code.*

- [ ] Show the **model's token contributions** in the drawer. Say: *"The model
      is linear, so each token's contribution to the winning class is exactly
      weight × coefficient — that's a real decomposition, not an approximation."*

---

## 10 · An incorrect / uncertain result and why (7:00–7:45) ⚠️ required

- [ ] Toggle **Needs review only** → 50 messages.
- [ ] Open **MSG_0152** — *"The report may be needed tomorrow."*

> The rule layer reads "may be needed" as a hedged action at 0.50. The model
> says general information at 0.77. They disagree, so confidence is damped to
> 0.58 and it's flagged. I labelled this `action_required` in my gold set, so
> the system is wrong here by my own judgement — but it's wrong *and it knows
> it*. The sentence hedges twice; I don't think there is a clean right answer.

- [ ] Then show the payoff:

```bash
grep -A6 "EVALUATION" /tmp/run.txt
```

> 97.8% overall. 100% on the templates I didn't flag as ambiguous. And — the
> number I care about most — **all 20 errors are inside the 50 messages the
> system flagged for review. On the 850 it didn't flag, it's 100%.**

---

## 11 · One code section explained in your own words (7:45–8:45) ⚠️ required

Open **`mint/pipeline.py`**, the `run()` function.

> This is the security design of the whole system, and it's really just an
> ordering decision. Read the row, scan and mask it, and *then* classify. The
> raw message never gets attached to the record we pass downstream —
> `ProcessedMessage` has no `raw` field at all. It can't leak a secret it is
> structurally incapable of holding.
>
> That's why nothing downstream has its own redaction logic. The classifier,
> the extractor, the JSON writer, the HTTP response, the UI — none of them need
> to remember to redact, because by the time they run there is nothing left to
> redact. Getting this right once, at the top, replaces getting it right in
> eight places.

Then show the test that enforces it:

```bash
python -m pytest tests/test_no_leakage.py -v
```

> This plants invented secrets in a synthetic corpus, runs the real pipeline,
> and asserts none of them appears in any artifact — including the sensitive
> report itself, which is the file most likely to leak the thing it's reporting.

**Optional, strong:** show that this test caught a real bug —
`mint/sensitive.py`, the phone detector comment. The original pattern assumed
5+5 digit grouping and missed a 6+6 number; it's now liberal about grouping
because the keyword anchor already established what the value is.

---

## 12 · Limitations and improvements (8:45–9:45) ⚠️ required

- [ ] `python -m pytest` → 95 passing.
- [ ] State plainly:

**Limitations**
1. 900 messages from 115 templates — far less varied than a real inbox. I'd
   expect accuracy to drop on free-form text.
2. I wrote both the rules and the gold labels, so their agreement partly
   measures my own consistency. A second annotator is the fix.
3. The honest ML number is **95.5%**, from cross-validation grouped by template
   so the model is tested on sentence forms it never saw. A random split scores
   100% — that number is memorisation and I report it only to dismiss it.
4. English only; frames are language-specific.
5. Detection is recall-limited by its keyword anchors — a bare 6-digit number
   with no surrounding keyword is not detected.
6. Confidence is calibrated by construction, not fitted.
7. No cross-message reasoning: a meeting scheduled in one message and moved in
   another produces two unlinked items.

**Improvements**
- Local sentence embeddings as a third voice (still fully offline).
- Active learning — the `needs_review` queue is already the perfect labelling
  batch, since every current error is in it.
- Checksum validation (Luhn for cards) to raise precision on unanchored numbers.
- Proper probability calibration once a second annotator exists.

- [ ] **Disclose AI-tool usage on camera:**
> I used Claude Code to build this — dataset analysis, drafting the modules,
> the tests and the README. It's declared in the README. But no message is ever
> sent to an AI service at runtime: classification is the regex frames and the
> scikit-learn model in this repo, and the 115 template labels in the gold set
> are my own judgement.

---

## 13 · Close (9:45–10:00)

- [ ] Show the three links on screen:
  - Repo — https://github.com/garvbahl37-gif/kastack-message-intelligence
  - Live demo — https://kastack-message-intelligence.vercel.app
  - Loom — this recording

---

## Final checklist

| Required item | Section |
|---|---|
| Approach & system flow | 0, 1 |
| Dataset structure, no sensitive values exposed | 2 |
| Results from all six categories | 3 |
| **All 15 mandatory message IDs** | 4 |
| ≥3 correctly extracted tasks | 5 |
| ≥3 correctly extracted meetings/events | 6 |
| One example with missing/unclear information | 7 |
| Sensitive detection + masking + risk + action | 8 |
| ≥3 classification decisions with explanations | 9 |
| One incorrect/uncertain result and why | 10 |
| One code section explained in your own words | 11 |
| Limitations and improvements | 12 |
| System actually running (not slides) | throughout |
