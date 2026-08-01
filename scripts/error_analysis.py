#!/usr/bin/env python3
"""
Error analysis over the scored evaluation outputs.

The comparison table says *how much* each system scores; this says *how it
fails*, which is what a reader needs to judge whether the system is safe to put
in front of an HR professional. Every scored record is assigned zero or more
error categories:

  hallucinated_section  cited a section number that does not exist in the Act
  wrong_section         gold sections known, but none of them were cited
  missing_citation      answered an in-scope question with no section citation
  weak_grounding        little of the answer is lexically supported by the Act
  unfaithful            judge faithfulness <= 2
  incomplete            judge completeness <= 2
  not_useful            judge usefulness <= 2
  harmful               judge flagged misleading / potentially harmful advice
  over_refusal          refused a question that is in scope
  failure_to_refuse     answered an out-of-scope question instead of declining
  generation_error      the backend call itself failed

Categories are deliberately non-exclusive: one answer can both cite a
non-existent section and be incomplete, and collapsing that into a single label
would understate the failure modes.

Outputs:
  results/error_analysis.json   per-record labels + per-system/per-topic counts
  results/error_analysis.csv    flat per-(system,set,category) counts
  results/error_examples.md     verbatim examples for the paper

Usage:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --examples-per-category 3
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import eval_metrics as M
from publication_content import canon_topic as M_TOPIC

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SCORE_DIR = RESULTS / "scored"
ACT_PDF = ROOT / "data/input/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf"

# Thresholds. Judge scores are 1-5, so <=2 is the bottom two bands ("misses the
# point" / "contradicts or fabricates"). GROUNDING_MIN is deliberately low: the
# metric counts verbatim 5-gram overlap with the statute, and a correct answer
# paraphrases heavily, so only near-zero overlap is treated as a signal.
JUDGE_LOW = 2.0
GROUNDING_MIN = 0.02
HARM_MIN = 0.5

IN_SCOPE_SETS = ("heldout", "scenario")

CATEGORIES = [
    "generation_error", "hallucinated_section", "wrong_section",
    "missing_citation", "weak_grounding", "unfaithful", "incomplete",
    "not_useful", "harmful", "over_refusal", "failure_to_refuse",
]


def load_valid_sections():
    """Section numbers that actually exist in the Act (for hallucination checks)."""
    return M.valid_sections()


def classify(rec, setname, valid_sections):
    """Return the list of error categories that apply to one scored record."""
    errs = []
    pred = rec.get("prediction") or ""
    if pred.startswith("[GEN_ERROR]"):
        return ["generation_error"]

    refused = bool(rec.get("refusal"))

    if setname == "oos":
        # The only thing that matters out of scope is whether it declined.
        if not refused:
            errs.append("failure_to_refuse")
        return errs

    if refused:
        errs.append("over_refusal")

    cited = set(M.extract_sections(pred))
    gold = set(rec.get("gold_sections") or [])

    if cited - valid_sections:
        errs.append("hallucinated_section")
    if not cited and not refused:
        errs.append("missing_citation")
    if gold and cited and not (cited & gold):
        errs.append("wrong_section")

    g = rec.get("grounding")
    if isinstance(g, (int, float)) and g < GROUNDING_MIN and not refused:
        errs.append("weak_grounding")

    j = rec.get("judge") or {}
    if isinstance(j, dict) and not j.get("parse_failed"):
        if isinstance(j.get("faithfulness"), (int, float)) and j["faithfulness"] <= JUDGE_LOW:
            errs.append("unfaithful")
        if isinstance(j.get("completeness"), (int, float)) and j["completeness"] <= JUDGE_LOW:
            errs.append("incomplete")
        if isinstance(j.get("usefulness"), (int, float)) and j["usefulness"] <= JUDGE_LOW:
            errs.append("not_useful")
        if isinstance(j.get("harm"), (int, float)) and j["harm"] >= HARM_MIN:
            errs.append("harmful")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples-per-category", type=int, default=2)
    args = ap.parse_args()

    if not SCORE_DIR.exists():
        raise SystemExit(f"no scored results in {SCORE_DIR}; run evaluate.py --phase score")

    valid_sections = load_valid_sections()
    print(f"{len(valid_sections)} valid section numbers parsed from the Act")

    per_record = []
    counts = defaultdict(Counter)          # (system,set) -> category counts
    totals = defaultdict(int)              # (system,set) -> n
    topic_counts = defaultdict(Counter)    # (system,topic) -> category counts
    examples = defaultdict(list)           # category -> records

    for sp in sorted(SCORE_DIR.glob("*.json")):
        system, setname = sp.stem.split("__", 1)
        recs = json.load(open(sp, encoding="utf-8"))
        for r in recs:
            errs = classify(r, setname, valid_sections)
            totals[(system, setname)] += 1
            counts[(system, setname)].update(errs)
            if setname in IN_SCOPE_SETS:
                topic_counts[(system, M_TOPIC(r.get("topic")))].update(errs)
            per_record.append({
                "system": system, "set": setname, "id": r.get("id"),
                "topic": r.get("topic"), "errors": errs,
            })
            for e in errs:
                if len(examples[e]) < args.examples_per_category * 4:
                    examples[e].append({
                        "system": system, "set": setname, "id": r.get("id"),
                        "question": r.get("question"),
                        "reference": r.get("reference"),
                        "prediction": r.get("prediction"),
                        "gold_sections": r.get("gold_sections"),
                        "cited": M.extract_sections(r.get("prediction") or ""),
                    })

    # ---- JSON ----
    out = {
        "thresholds": {"judge_low": JUDGE_LOW, "grounding_min": GROUNDING_MIN,
                       "harm_min": HARM_MIN},
        "n_valid_sections": len(valid_sections),
        "by_system_set": {
            f"{s}__{ss}": {"n": totals[(s, ss)], "counts": dict(counts[(s, ss)]),
                           "rates": {k: v / totals[(s, ss)]
                                     for k, v in counts[(s, ss)].items()}}
            for (s, ss) in sorted(totals)
        },
        "by_system_topic": {
            f"{s}__{t}": dict(c) for (s, t), c in sorted(topic_counts.items())
        },
        "per_record": per_record,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RESULTS / "error_analysis.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    # ---- CSV ----
    with open(RESULTS / "error_analysis.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "set", "n"] + CATEGORIES)
        for (s, ss) in sorted(totals):
            w.writerow([s, ss, totals[(s, ss)]]
                       + [counts[(s, ss)].get(c, 0) for c in CATEGORIES])

    # ---- examples for the paper ----
    lines = ["# Error examples\n",
             "Verbatim failures drawn from the scored evaluation outputs.\n"]
    for cat in CATEGORIES:
        if not examples[cat]:
            continue
        lines.append(f"\n## {cat}\n")
        for ex in examples[cat][:args.examples_per_category]:
            lines.append(f"**{ex['system']} / {ex['set']} / id={ex['id']}**\n")
            lines.append(f"- Q: {ex['question']}\n")
            if ex.get("gold_sections"):
                lines.append(f"- Gold sections: {ex['gold_sections']} | "
                             f"Cited: {ex['cited']}\n")
            if ex.get("reference"):
                lines.append(f"- Reference: {ex['reference']}\n")
            pred = (ex.get("prediction") or "").strip().replace("\n", " ")
            lines.append(f"- Model: {pred[:600]}\n")
    (RESULTS / "error_examples.md").write_text("".join(lines), encoding="utf-8")

    # ---- console summary ----
    print(f"\n{'system':<16}{'set':<10}{'n':>4}  top failure modes")
    for (s, ss) in sorted(totals):
        top = ", ".join(f"{k}={v}" for k, v in counts[(s, ss)].most_common(4))
        print(f"{s:<16}{ss:<10}{totals[(s,ss)]:>4}  {top or '-'}")
    print(f"\nWrote {RESULTS/'error_analysis.json'}, "
          f"{RESULTS/'error_analysis.csv'}, {RESULTS/'error_examples.md'}")


if __name__ == "__main__":
    main()
