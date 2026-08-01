#!/usr/bin/env python3
"""
Validate the refusal detector against hand-labelled ground truth.

Out-of-scope refusal is the paper's safety-relevant metric, and it is computed by
a phrase-matching heuristic. A heuristic reported without a validation figure is
an assumption, so this measures the detector against
`data/eval/out_of_scope_refusal_labels.json` - 20 answers labelled by hand from
the base model's out-of-scope generations - and prints precision, recall and the
individual disagreements.

The labelling rule is that hedging is not declining: an answer that opens "I'm
not an expert on taxation, but..." and then states the tax rates has answered an
out-of-scope question, however heavily qualified.

The agreement figure this prints is what the paper cites when it reports refusal
rates. Run it after changing anything in eval_metrics.is_refusal.

Usage:
    python scripts/check_refusal_detector.py
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_metrics as M

LABELS = ROOT / "data/eval/out_of_scope_refusal_labels.json"
GENERATIONS = ROOT / "results/generations/base__oos.json"
REPORT = ROOT / "results/refusal_detector_validation.json"


def main():
    if not LABELS.exists():
        raise SystemExit(f"missing {LABELS}")
    if not GENERATIONS.exists():
        raise SystemExit(f"missing {GENERATIONS}; run evaluate.py first")

    labels = json.load(open(LABELS, encoding="utf-8"))["labels"]
    gens = json.load(open(GENERATIONS, encoding="utf-8"))

    tp = fp = tn = fn = 0
    disagreements = []
    for i, rec in enumerate(gens):
        key = str(i)
        if key not in labels:
            continue
        gold = labels[key]["label"] == "refused"
        pred = M.is_refusal(rec.get("prediction") or "")
        if gold and pred:
            tp += 1
        elif gold and not pred:
            fn += 1
            disagreements.append((i, "MISSED refusal", labels[key]["note"], rec))
        elif not gold and pred:
            fp += 1
            disagreements.append((i, "FALSE refusal", labels[key]["note"], rec))
        else:
            tn += 1

    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec_ = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0

    print(f"Refusal detector vs {n} hand-labelled answers\n")
    print(f"  gold refusals      {tp + fn}")
    print(f"  detected refusals  {tp + fp}")
    print(f"  accuracy           {acc:.2f}")
    print(f"  precision          {prec:.2f}")
    print(f"  recall             {rec_:.2f}")
    print(f"  F1                 {f1:.2f}")

    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s):")
        for i, kind, note, rec in disagreements:
            print(f"\n  [{i}] {kind} - gold note: {note}")
            print(f"      {' '.join((rec.get('prediction') or '').split())[:180]}")
    else:
        print("\n  no disagreements")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": n, "true_positive": tp, "false_positive": fp,
               "true_negative": tn, "false_negative": fn,
               "accuracy": acc, "precision": prec, "recall": rec_, "f1": f1},
              open(REPORT, "w"), indent=1)
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
