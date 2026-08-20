# L2 video demonstration script — 5 minutes maximum

Every item the L2 brief lists is mapped to an exact action below. Tick each box
as you record; anything not shown may be treated as incomplete. Five minutes is
the **maximum**, so the timings are tight and the running order is chosen so
nothing needs to be repeated.

## Before you start

```bash
cd /Users/garvbahl/Documents/Projects/KaStack/l1_repo
source .venv/bin/activate

# Have these ready in a terminal, already run, output on screen:
python scripts/run_l2.py \
    --input data/messages.csv data/l2_messages.csv \
    --demo  data/l2_demo_messages.csv \
    --queries data/l2_demo_queries.csv > /tmp/l2.txt

python scripts/benchmark.py \
    --input data/messages.csv data/l2_messages.csv \
    --demo data/l2_demo_messages.csv --queries data/l2_demo_queries.csv > /tmp/bench.txt

uvicorn api.index:app --port 8000     # leave running in a second terminal
```

Open **http://localhost:8000/l2** in the browser and a terminal beside it.

> **Which host to demo on.** Use **localhost** for anything touching the
> supplied datasets — same code, same UI as the cloud demo, and the datasets
> stay off the network. Show the cloud URL once, at the start, to prove the
> hosted demo is live.
>
> **Masking is automatic.** Every view renders a redaction bar with the text
> node removed — there is no screen in this system that can display a raw
> secret. Nothing needs blurring. Just don't open the CSVs in an editor on
> camera.

**The load order matters and is worth saying out loud:** upload
`messages.csv`, then `l2_messages.csv`, then `l2_demo_messages.csv` — the demo
batch is loaded **as a new unseen batch**, which is what makes the priority
updates visible.

---

## 0 · What L1 was, and what L2 added (0:00–0:45)

- [ ] Open **https://kastack-message-intelligence.vercel.app/l2** — cloud demo is live.
- [ ] Click **← L1 view**: "L1 read each message once — six categories, extracted
      tasks and events, sensitive values masked before anything else ran. 97.8%
      against my hand-labelled gold set."
- [ ] Back to **/l2**: "L2 stops reading messages and starts tracking subjects.
      Same pipeline runs first; then every message is assigned to a subject and
      advances that subject's state machine, in chronological order."
- [ ] Point at the **How L2 extends L1** card: reference time is the newest
      message, not the wall clock — same input, same output, every time.

## 1 · Load the three batches, in order (0:45–1:15)

- [ ] Switch to localhost. **Upload batches** → select `messages.csv` and
      `l2_messages.csv` together.
- [ ] Then **Add a batch** → `l2_demo_messages.csv`.
- [ ] Point at the batch chips: 900 · 180 · 24, each labelled, processed in that
      order. "1,104 messages in about 205 milliseconds."

## 2 · Two groups, correctly grouped (1:15–2:00) — *required*

- [ ] **Groups** tab → open **Internship orientation**.
      - Timeline: raised in L1, ten announcements, two reschedules in L2, then
        `DEMO_007` moves it to 2026-10-07 15:00 and `DEMO_009` changes **only
        the time** to 17:30 — the date is kept. "The latest schedule is the
        merge of every change in order, not the contents of the last message."
      - `DEMO_017` hedges — "we may move it, I'll confirm later" — and changes
        nothing. Recorded as a contradiction so the uncertainty is visible.
- [ ] Open **Update the project tracker**.
      - 18 messages, chased 15 times, cancelled in `MSG_0942`, then `DEMO_006`
        changes its deadline **after** it was cancelled → contested.
      - Say the rule: "grouping is not word overlap. *Review the privacy
        checklist* and *review the model results* share the word *review* and
        stay in separate groups; *the assignment* reaches *upload the
        assignment* because the whole of the shorter phrase is covered."

## 3 · Priority, and how it updated (2:00–2:45) — *required*

- [ ] **Priority** tab: 7 critical, sorted by score.
- [ ] Open **Confirm the interview slot** → the signal table. Read two or three
      signals with their weights. "Every signal is a named fact with a weight
      and a sentence. The reason is assembled from the ones that fired."
- [ ] **Priority history**: score 4.7 → 9.3 on `DEMO_001`. "That is the demo
      batch escalating it — the deadline moved to *tomorrow* and the sender said
      urgent. The word *tomorrow* was never turned into a date; it is kept as a
      stated proximity and the confidence is reduced to say so."
- [ ] Terminal, `/tmp/l2.txt`: the **PRIORITY UPDATES BY BATCH** block. Point at
      one `via message` row and one `via elapsed_time` row. "Priority can change
      with no new message at all, because a deadline passed. The system records
      which of the two happened."

## 4 · Privacy-aware routing — all three routes (2:45–3:30) — *required*

- [ ] **Privacy** tab, policy table on screen.
- [ ] **Processed locally** — filter route = *local only*, point at any ordinary
      message. Full local analysis, masked text, never exportable off-device.
- [ ] **Requires confirmation** — filter *confirm required* → `DEMO_014`
      (a delivery address). "Indexed in masked form, because this is often
      exactly what someone is searching for. Quoting or exporting it needs an
      explicit confirmation."
- [ ] **Blocked** — filter *blocked* → `DEMO_012`, `DEMO_013`, `DEMO_024`. The
      content column says *withheld by P1-CREDENTIAL*. "These were never added
      to the search index. Not filtered out of results — never candidates."
- [ ] **Assistant** tab → ask *"Which demo messages must be blocked from external
      processing?"* Three IDs, types named, content withheld, route badge reads
      **blocked**.
- [ ] Then *"Which message requires confirmation before processing?"* — route
      badge reads **confirmation required**; tick the confirm box and re-ask to
      show the gate opening.

## 5 · The assistant, including a refusal (3:30–4:15)

- [ ] *"Which meeting was rescheduled and what is its latest schedule?"* —
      internship orientation, 2026-10-07 at 17:30.
- [ ] *"Which existing task became critical in the demo data?"* — the interview
      slot, with the score movement and the message that caused it.
- [ ] *"Was the compliance form approved by the finance director?"* — **refused.**
      "This is the one I care about most. That question has a near-perfect
      lexical match in the corpus, because `DEMO_022` asks exactly the same
      thing. Returning it would be the most confident possible way to be wrong.
      A yes/no question can only be answered by a message that asserts an
      outcome, so the system checks for an assertion, finds none, and says so."
- [ ] Point at the evidence panel on any answer: relevance scores, why each
      piece was selected, and the retrieval scores shown alongside for
      comparison.

## 6 · Baseline against optimised (4:15–4:45) — *required*

- [ ] **Benchmark** tab → **Run the comparison**. It runs live on the corpus
      just loaded.
- [ ] Read three numbers: **median latency 0.386 → 0.053 ms (7.4×)**, **19.5× at
      8,832 documents**, **99.8% of the exact index's top 10 reproduced**.
- [ ] Memory: **1.46× the baseline** — "it uses more memory. That is the trade,
      and hiding it would be picking the favourable metric."
- [ ] Terminal, `/tmp/bench.txt`: the quality table. "Same recall as exact
      scoring. int8 embeddings are 8× smaller than float64."
- [ ] Testing device: "Apple silicon, macOS, CPython 3.12, single process,
      warm-up discarded, median and p95 over 42 queries × 30 repeats. The
      absolute numbers are this machine's; the ratios are the comparable part."

## 7 · The interesting result, and what I'd do next (4:45–5:00) — *required*

- [ ] `/tmp/bench.txt`, the `v2-semantic-always` row. "My first design blended
      lexical and semantic scores on every query. Measuring it showed it was
      slower, reproduced less of the exact ranking, and was slightly *worse* on
      paraphrases. Every subject in this corpus is phrased consistently, so
      there is almost no synonymy for a latent space to bridge — it just blurs a
      sharp signal. So the semantic layer became a recall fallback: the query
      embedding isn't computed at all unless lexical scoring can't fill the
      result set."
- [ ] "The limitation that leaves: out-of-vocabulary paraphrases retrieve
      nothing — *settle the utility payment* doesn't find *pay the electricity
      bill*, because none of those words is in the corpus. Recall drops from
      0.85 to 0.50 on that set. A local subword encoder is the fix, and it's a
      deployment-size decision rather than a research one."

---

## Coverage checklist

| The brief asks for | Section |
|---|---|
| Your L1 system and how you extended it in L2 | 0 |
| Two examples of correct related-message grouping | 2 |
| One request processed locally | 4 |
| One request requiring confirmation | 4 |
| One blocked request | 4 |
| Sensitive information shown only in masked form | throughout — automatic |
| Comparison: response time | 6 |
| Comparison: model / index / application size | 6 |
| Comparison: change in result quality | 6 |
| Testing device and how performance was measured | 6 |
| An interesting challenge or incorrect result, and the fix | 7 |
| The supplied demo messages and queries used live | 1, 4, 5 |

## If you run over five minutes

Cut in this order: §2's second group, then §5's first two queries, then §0's
L1 detour (keep the sentence, drop the click). Never cut §4 or §6 — those are
the two the brief calls out most explicitly.
