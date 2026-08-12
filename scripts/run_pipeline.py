"""CLI: run the full pipeline over a message CSV and write structured outputs.

    python scripts/run_pipeline.py --input data/messages.csv --out outputs/

Add --evaluate to score the run against eval/gold_labels.csv, and
--ids data/mandatory_demo_ids.csv to print the required demonstration rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mint import pipeline  # noqa: E402
from mint import taxonomy as T  # noqa: E402

BAR = "=" * 74


def _bar(n: int, total: int, width: int = 34) -> str:
    filled = int(round(width * n / total)) if total else 0
    return "#" * filled + "." * (width - filled)


def print_summary(result: pipeline.PipelineResult) -> None:
    s = result.summary()
    print(BAR)
    print("SUMMARY")
    print(BAR)
    print(f"  messages processed : {s['total_messages']}")
    print(f"  chronological span : {s['date_range']['from']}  ->  {s['date_range']['to']}")
    print(f"  mean confidence    : {s['mean_confidence']}")
    print(f"  flagged for review : {s['flagged_for_review']}")
    print()
    print("  Categories")
    total = s["total_messages"]
    for cat in T.CATEGORIES:
        n = s["categories"].get(cat, 0)
        print(f"    {cat:<24s} {n:>4d}  {_bar(n, total)}  {n/total:6.1%}")
    print()
    print(f"  Tasks extracted    : {s['tasks_extracted']}")
    print(f"  Events extracted   : {s['events_extracted']}")
    print(f"  Items w/ unresolved fields : {s['items_with_unresolved_fields']}")
    print()
    print(f"  Sensitive messages : {s['sensitive_messages']}")
    for t, n in sorted(s["sensitive_by_type"].items(), key=lambda kv: -kv[1]):
        print(f"    {t:<24s} {n:>4d}")
    print(f"  By risk            : {s['sensitive_by_risk']}")


def evaluate(result: pipeline.PipelineResult, gold_path: Path) -> None:
    with gold_path.open(encoding="utf-8") as fh:
        gold = {r["message_id"]: r for r in csv.DictReader(fh)}

    pred = {m.message_id: m.classification for m in result.messages}
    common = [mid for mid in pred if mid in gold]
    if not common:
        print("no overlap between predictions and gold labels", file=sys.stderr)
        return

    correct = sum(1 for mid in common if pred[mid].category == gold[mid]["category"])
    hard = [mid for mid in common if gold[mid].get("ambiguous") == "yes"]
    easy = [mid for mid in common if gold[mid].get("ambiguous") != "yes"]

    print()
    print(BAR)
    print("EVALUATION vs hand-labelled gold set")
    print(BAR)
    print(f"  overall accuracy            : {correct}/{len(common)} = "
          f"{correct/len(common):.4f}")
    if easy:
        ce = sum(1 for m in easy if pred[m].category == gold[m]["category"])
        print(f"  excluding ambiguous templates: {ce}/{len(easy)} = {ce/len(easy):.4f}")
    if hard:
        ch = sum(1 for m in hard if pred[m].category == gold[m]["category"])
        print(f"  on ambiguous templates only  : {ch}/{len(hard)} = {ch/len(hard):.4f}")

    # Per-category precision / recall.
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for mid in common:
        p, g = pred[mid].category, gold[mid]["category"]
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    print()
    print(f"  {'category':<24s}{'prec':>8s}{'rec':>8s}{'F1':>8s}{'n':>7s}")
    for cat in T.CATEGORIES:
        support = tp[cat] + fn[cat]
        prec = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) else 0.0
        rec = tp[cat] / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {cat:<24s}{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}{support:>7d}")

    errors = [mid for mid in common if pred[mid].category != gold[mid]["category"]]
    if errors:
        print()
        print(f"  {len(errors)} disagreement(s); grouped by (gold -> predicted):")
        grouped = defaultdict(list)
        for mid in errors:
            grouped[(gold[mid]["category"], pred[mid].category)].append(mid)
        for (g, p), ids in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
            flag = " [ambiguous]" if gold[ids[0]].get("ambiguous") == "yes" else ""
            print(f"    {g} -> {p}: {len(ids)}{flag}  e.g. {', '.join(ids[:3])}")


def print_mandatory(result: pipeline.PipelineResult, ids_path: Path) -> None:
    with ids_path.open(encoding="utf-8-sig") as fh:
        wanted = [r["message_id"].strip() for r in csv.DictReader(fh)
                  if r.get("message_id", "").strip()]

    by_id = {m.message_id: m for m in result.messages}
    items_by_msg = {i.source_message_id: i for i in result.items}

    print()
    print(BAR)
    print(f"MANDATORY DEMONSTRATION IDS ({len(wanted)})")
    print(BAR)
    for mid in wanted:
        m = by_id.get(mid)
        if m is None:
            print(f"  {mid}: NOT FOUND in the input file")
            continue
        c = m.classification
        flag = "  [REVIEW]" if c.needs_review else ""
        print(f"\n  {mid}  {m.timestamp}  from {m.sender}{flag}")
        print(f"    text     : {m.masked_text}")
        print(f"    category : {c.category}   confidence: {c.confidence}")
        print(f"    reason   : {c.reason}")
        if c.secondary_category:
            print(f"    secondary: {c.secondary_category}")
        if m.scan_result.is_sensitive:
            sr = m.scan_result
            print(f"    SENSITIVE: {sr.primary_type} | risk={sr.overall_risk} "
                  f"| action={sr.overall_action}")
        it = items_by_msg.get(mid)
        if it:
            print(f"    extracted: [{it.item_id}] {it.type} \"{it.title}\"")
            print(f"               date={it.date} ({it.date_status})  "
                  f"time={it.time}  person={it.person}  priority={it.priority}")
            print(f"               priority reason: {it.priority_reason}")
            if it.unresolved_fields:
                print(f"               unresolved: {', '.join(it.unresolved_fields)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", default="data/messages.csv")
    ap.add_argument("--out", "-o", default="outputs")
    ap.add_argument("--evaluate", action="store_true",
                    help="score against eval/gold_labels.csv")
    ap.add_argument("--ids", help="CSV of message IDs to print in full")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    src = (ROOT / args.input) if not Path(args.input).is_absolute() else Path(args.input)
    if not src.exists():
        print(f"error: input file not found: {src}", file=sys.stderr)
        return 1

    result = pipeline.run(src)
    written = pipeline.write_outputs(result, ROOT / args.out
                                     if not Path(args.out).is_absolute()
                                     else Path(args.out))

    if not args.quiet:
        print_summary(result)

    if args.evaluate:
        gold = ROOT / "eval" / "gold_labels.csv"
        if gold.exists():
            evaluate(result, gold)
        else:
            print(f"\ngold labels not found at {gold}; run scripts/make_gold.py first",
                  file=sys.stderr)

    if args.ids:
        p = Path(args.ids)
        p = p if p.is_absolute() else ROOT / p
        if p.exists():
            print_mandatory(result, p)
        else:
            print(f"\nids file not found: {p}", file=sys.stderr)

    print()
    print(BAR)
    print("OUTPUT FILES")
    print(BAR)
    for path in written:
        print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
