"""CLI: run the L2 ledger over the L1 and L2 message files and write outputs.

    python scripts/run_l2.py \
        --input data/messages.csv data/l2_messages.csv \
        --demo data/l2_demo_messages.csv \
        --queries data/l2_demo_queries.csv \
        --out outputs/

`--input` files are processed in the order given, and `--demo` is appended
last. That ordering is the point: the brief requires the L2 messages to be
processed after the L1 messages, and the demo batch after both.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mint import groups as G  # noqa: E402
from mint import ledger as L  # noqa: E402
from mint import priority as P  # noqa: E402
from mint import routing as R  # noqa: E402
from mint.assistant import Assistant  # noqa: E402

BAR = "=" * 78


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def print_summary(ledger: L.Ledger) -> None:
    s = ledger.summary()
    print(BAR)
    print("L2 LEDGER")
    print(BAR)
    print(f"  reference time (newest message) : {s['as_of']}")
    print(f"  messages processed              : {s['total_messages']}")
    for batch in s["batches"]:
        print(f"    {batch['name']:<28s} {batch['messages']:>5d}  "
              f"{batch['from']} -> {batch['to']}")
    print()
    print(f"  subject groups                  : {s['groups']}")
    print(f"  messages in a group             : {s['grouped_messages']}")
    print(f"  messages in no group            : {s['ungrouped_messages']}")
    print(f"  mean group confidence           : {s['mean_group_confidence']}")
    print(f"  groups with a recorded conflict : {s['groups_with_conflicts']}")
    print(f"  contested groups                : {s['contested_groups']}")
    print()
    print("  Group status")
    for status, n in s["group_status_counts"].items():
        print(f"    {status:<16s}{n:>5d}")
    print()
    print("  Group priority (as of the newest message)")
    for band in P.BANDS:
        print(f"    {band:<16s}{s['group_priority_counts'][band]:>5d}")
    print()
    print("  Privacy routing")
    for route, n in s["route_counts"].items():
        print(f"    {route:<28s}{n:>5d}")
    print(f"    excluded from search index  {s['excluded_from_index']:>5d}")


def print_priority_changes(ledger: L.Ledger) -> None:
    print()
    print(BAR)
    print("PRIORITY UPDATES BY BATCH")
    print(BAR)
    for snapshot in ledger.snapshots:
        real = [(gid, c) for gid, c in snapshot.changes if c.kind != "initial"]
        print(f"\n  {snapshot.batch}  (as of {snapshot.as_of})  "
              f"{len(real)} change(s)")
        for gid, change in real:
            group = ledger.group(gid)
            arrow = (f"{change.previous} -> {change.new}"
                     if change.previous != change.new
                     else f"{change.new} ({change.delta:+.1f})")
            trigger = (", ".join(change.trigger_message_ids[:3])
                       if change.trigger_message_ids else "time passed")
            print(f"    {gid:<11s} {group.title[:32]:<34s} {arrow:<24s} "
                  f"{trigger}")


def print_groups(ledger: L.Ledger, limit: int) -> None:
    print()
    print(BAR)
    print(f"LARGEST SUBJECT GROUPS (top {limit})")
    print(BAR)
    ordered = sorted(ledger.groups, key=lambda g: -len(g.members))[:limit]
    for group in ordered:
        decision = ledger.priorities.get(group.group_id)
        print(f"\n  {group.group_id}  {group.title}  [{group.kind}]")
        print(f"    status   : {G.STATUS_LABEL[group.status]}"
              + ("  (contested)" if group.contested else ""))
        print(f"    priority : {decision.priority if decision else '-'}"
              f"   confidence {group.confidence}")
        print(f"    messages : {len(group.members)}  "
              f"({', '.join(group.message_ids[:6])}"
              f"{', ...' if len(group.members) > 6 else ''})")
        print(f"    summary  : {group.summary}")


def print_answers(rows: list) -> None:
    print()
    print(BAR)
    print("ASSISTANT")
    print(BAR)
    for row in rows:
        print(f"\n  Q  {row['query']}")
        print(f"     intent {row['intent']}  |  evidence "
              f"{'sufficient' if row['sufficient_evidence'] else 'INSUFFICIENT'}"
              f"  |  confidence {row['confidence']}  |  route "
              f"{row['privacy_route']['route']}")
        print(f"  A  {row['answer']}")
        print(f"  Why evidence: {row['reason']}")
        ids = ", ".join(row["supporting_message_ids"][:10]) or "none"
        print(f"  Support: {ids}")
        if row["retrieved"]:
            top = row["retrieved"][0]
            print(f"  Top retrieval: {top['message_id']} at {top['score']} "
                  f"(lexical {top['lexical_score']}, semantic "
                  f"{top['semantic_score']})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", nargs="+",
                    default=["data/messages.csv", "data/l2_messages.csv"])
    ap.add_argument("--demo", help="an extra batch processed last")
    ap.add_argument("--queries", help="CSV of demo queries to run")
    ap.add_argument("--out", "-o", default="outputs")
    ap.add_argument("--groups", type=int, default=8,
                    help="how many groups to print")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sources = []
    for path in args.input:
        resolved = _resolve(path)
        if not resolved.exists():
            print(f"error: input not found: {resolved}", file=sys.stderr)
            return 1
        sources.append((resolved.name, resolved))
    if args.demo:
        resolved = _resolve(args.demo)
        if not resolved.exists():
            print(f"error: demo batch not found: {resolved}", file=sys.stderr)
            return 1
        sources.append((resolved.name, resolved))

    ledger = L.run(sources)
    assistant = Assistant(ledger)

    answers = []
    if args.queries:
        qpath = _resolve(args.queries)
        if qpath.exists():
            text = qpath.read_text(encoding="utf-8-sig").lstrip("﻿")
            for row in csv.DictReader(io.StringIO(text)):
                query = (row.get("query") or "").strip()
                if query:
                    answers.append(assistant.answer(query).to_dict())
        else:
            print(f"warning: queries file not found: {qpath}", file=sys.stderr)

    written = L.write_outputs(ledger, _resolve(args.out),
                              answers=answers or None)

    if not args.quiet:
        print_summary(ledger)
        print_priority_changes(ledger)
        print_groups(ledger, args.groups)
        if answers:
            print_answers(answers)

    print()
    print(BAR)
    print("OUTPUT FILES")
    print(BAR)
    for path in written:
        try:
            shown = path.relative_to(ROOT)
        except ValueError:                     # --out pointed outside the repo
            shown = path
        print(f"  {shown}  ({path.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
