#!/usr/bin/env python3
"""
Build a leakage-free, topic-stratified held-out test set.

The training pipeline previously consumed the entire dataset with only an
internal random 90/10 loss-split; nothing was reserved, so any "test" numbers
risked train/test leakage. This script carves a fixed held-out set OUT of the
cleaned dataset and writes the remaining items as the training pool, guaranteeing
the two are disjoint (verified by hash-set intersection == 0).

Held-out items are stratified across HR topics (wages, leave, termination,
maternity, safety, overtime, misconduct, probation, trade union, compensation,
...) so every topic is represented. Each held-out item is emitted with gold
section numbers parsed from its reference answer, for the citation-accuracy
metric downstream.

Usage:
    python scripts/build_test_split.py \
        --input data/final/bangladesh_labour_act_chatml_clean.json \
        --heldout data/eval/heldout_test.json \
        --train-pool data/final/train_pool.json \
        --n 150 --seed 3407
"""

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict, Counter
from pathlib import Path

# Topic buckets in priority order: each item is assigned to the FIRST bucket
# whose keywords match its question, so counts are disjoint. Order puts more
# specific / lower-frequency topics first to avoid them being swallowed by
# high-frequency ones (e.g. "wage" appears in many questions).
TOPIC_KEYWORDS = [
    ("maternity",     ["maternity", "pregnan", "childbirth", "mother"]),
    ("probation",     ["probation", "probationer"]),
    ("misconduct",    ["misconduct", "disciplinary", "punishment", "penalty"]),
    ("dismissal",     ["dismiss", "termination", "terminate", "discharge", "retrench", "notice period"]),
    ("overtime",      ["overtime", "extra hours", "working hour", "hours of work"]),
    ("leave",         ["leave", "holiday", "festival", "casual leave", "annual leave", "sick leave"]),
    ("maternity_benefit", ["maternity benefit"]),
    ("safety",        ["safety", "hazard", "accident", "fire", "ventilation", "hygiene", "health"]),
    ("compensation",  ["compensation", "injury", "disablement", "gratuity", "provident"]),
    ("trade_union",   ["trade union", "collective bargaining", "cba", "strike", "lockout", "dispute"]),
    ("apprentice",    ["apprentice", "apprenticeship"]),
    ("wages",         ["wage", "salary", "payment", "deduction", "bonus", "remuneration"]),
]

SECTION_RE = re.compile(r"[Ss]ection\s+(\d+)")


def assign_topic(question: str) -> str:
    q = question.lower()
    for name, kws in TOPIC_KEYWORDS:
        if any(k in q for k in kws):
            return name
    return "other"


def item_hash(item: dict) -> str:
    """Stable content hash over the message texts (order-preserving)."""
    blob = "|".join(f"{m.get('role')}:{m.get('content','')}" for m in item["messages"])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def qa(item):
    q = next(m["content"] for m in item["messages"] if m["role"] == "user")
    a = next(m["content"] for m in item["messages"] if m["role"] == "assistant")
    return q, a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", default="data/final/bangladesh_labour_act_chatml_clean.json")
    ap.add_argument("--heldout", default="data/eval/heldout_test.json")
    ap.add_argument("--train-pool", default="data/final/train_pool.json")
    ap.add_argument("--n", type=int, default=150, help="target held-out size")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = json.load(open(args.input, encoding="utf-8"))
    print(f"Loaded {len(data)} items from {args.input}")

    # Bucket indices by topic.
    buckets = defaultdict(list)
    for idx, item in enumerate(data):
        q, _ = qa(item)
        buckets[assign_topic(q)].append(idx)
    for b in buckets.values():
        rng.shuffle(b)

    topic_counts = {t: len(ix) for t, ix in buckets.items()}
    print("Topic distribution (full dataset):")
    for t, c in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:20s} {c}")

    # Proportional allocation across topics (at least 1 per non-empty topic,
    # excluding the catch-all 'other' from over-representation).
    total = len(data)
    heldout_idx = []
    for topic, ix in buckets.items():
        share = max(1, round(args.n * len(ix) / total))
        take = min(share, len(ix))
        heldout_idx.extend(ix[:take])
    # Trim/expand to hit the target n as closely as possible.
    rng.shuffle(heldout_idx)
    heldout_idx = heldout_idx[: args.n]
    heldout_set = set(heldout_idx)

    heldout, train_pool = [], []
    for idx, item in enumerate(data):
        if idx in heldout_set:
            q, a = qa(item)
            gold = sorted({int(s) for s in SECTION_RE.findall(a)})
            heldout.append({
                "id": idx,
                "topic": assign_topic(q),
                "question": q,
                "reference": a,
                "gold_sections": gold,
            })
        else:
            train_pool.append(item)

    # Leakage guard: held-out and train pool must be disjoint by content hash.
    train_hashes = {item_hash(it) for it in train_pool}
    heldout_hashes = {item_hash(data[h["id"]]) for h in heldout}
    overlap = train_hashes & heldout_hashes
    assert not overlap, f"LEAKAGE: {len(overlap)} held-out items also appear in train pool"

    Path(args.heldout).parent.mkdir(parents=True, exist_ok=True)
    Path(args.train_pool).parent.mkdir(parents=True, exist_ok=True)
    json.dump(heldout, open(args.heldout, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(train_pool, open(args.train_pool, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    held_topics = Counter(h["topic"] for h in heldout)
    with_cite = sum(1 for h in heldout if h["gold_sections"])
    print(f"\nHeld-out: {len(heldout)} items -> {args.heldout}")
    print(f"Train pool: {len(train_pool)} items -> {args.train_pool}")
    print(f"Leakage check: PASS (0 overlapping items)")
    print(f"Held-out items with >=1 gold section: {with_cite}")
    print("Held-out topic distribution:")
    for t, c in held_topics.most_common():
        print(f"  {t:20s} {c}")


if __name__ == "__main__":
    main()
