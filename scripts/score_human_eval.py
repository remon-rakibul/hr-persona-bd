#!/usr/bin/env python3
"""
Score the blind human evaluation: deblind, aggregate, and measure agreement.

build_human_eval.py produces the rating app; this consumes the CSVs it exports.
The two are separate on purpose - the rater CSVs carry only neutral labels
(A, B, C), and the mapping back to systems lives in answer_key.json, so nothing
in the rater's possession reveals which system produced which answer.

What this reports:

  per-system means per criterion, with 95% bootstrap CIs over items
  Krippendorff's alpha for inter-annotator agreement, ordinal for the 1-5
    criteria and nominal for the binary harm flag
  paired Wilcoxon tests between systems on each criterion

Agreement is reported first and prominently. Means computed from raters who do
not agree with each other are not interpretable, and an alpha below about 0.67
is conventionally treated as too low to draw conclusions from - so the number
that qualifies every other number belongs at the top, not in a footnote.

Usage:
    python scripts/score_human_eval.py human_eval/rater1.csv human_eval/rater2.csv
    python scripts/score_human_eval.py --key human_eval/answer_key.json human_eval/*.csv
"""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KEY = ROOT / "human_eval" / "answer_key.json"
OUT = ROOT / "results" / "human_eval_results.json"

ORDINAL_CRITERIA = ["accuracy", "citation", "completeness", "usefulness"]
NOMINAL_CRITERIA = ["harmful"]
SEED = 3407
N_BOOT = 10000


def load_ratings(paths):
    """[(rater, item, label, criterion, score_str)] from the exported CSVs."""
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append((r["rater"], int(r["item"]), r["answer_label"],
                             r["criterion"], r["score"]))
    return rows


def deblind(rows, key):
    """Attach the system identity to each rating."""
    mapping = {it["item"]: it["mapping"] for it in key["items"]}
    out = []
    for rater, item, label, crit, score in rows:
        sysname = mapping.get(item, {}).get(label)
        if sysname:
            out.append({"rater": rater, "item": item, "system": sysname,
                        "criterion": crit, "score": score})
    return out


def krippendorff_alpha(units, level="ordinal"):
    """Krippendorff's alpha.

    `units` maps a unit id to the list of values assigned by different coders.
    Units rated by fewer than two coders carry no agreement information and are
    dropped, per the standard definition.

    Computed from the observed and expected disagreement over all codeable
    pairs, which handles missing ratings without imputation - raters who skip
    different items still contribute.
    """
    pairable = {u: [v for v in vs if v is not None] for u, vs in units.items()}
    pairable = {u: vs for u, vs in pairable.items() if len(vs) >= 2}
    if not pairable:
        return None

    values = sorted({v for vs in pairable.values() for v in vs})
    if len(values) < 2:
        return 1.0                      # no variation: perfect agreement
    idx = {v: i for i, v in enumerate(values)}

    # Marginal counts over pairable values; the ordinal metric is defined in
    # terms of them, not of raw rank positions.
    counts = [0] * len(values)
    for vs in pairable.values():
        for v in vs:
            counts[idx[v]] += 1

    def delta(a, b):
        if level == "nominal":
            return 0.0 if a == b else 1.0
        # Krippendorff's ordinal metric:
        #   d(c,k) = ( sum_{g=c..k} n_g  -  (n_c + n_k)/2 ) ** 2
        # Distance depends on how many observations lie between the two ranks,
        # so a step across a densely used part of the scale counts for more than
        # a step across a sparsely used part. Plain rank distance ignores this
        # and gives materially different alphas.
        i, j = sorted((idx[a], idx[b]))
        between = sum(counts[i:j + 1])
        return float((between - (counts[i] + counts[j]) / 2.0) ** 2)

    # Observed disagreement, weighted by unit size.
    num, n_total = 0.0, 0
    for vs in pairable.values():
        m = len(vs)
        n_total += m
        for a, b in itertools.permutations(vs, 2):
            num += delta(a, b) / (m - 1)
    Do = num / n_total if n_total else 0.0

    # Expected disagreement from the pooled marginal distribution.
    allv = [v for vs in pairable.values() for v in vs]
    den = 0.0
    for a, b in itertools.permutations(allv, 2):
        den += delta(a, b)
    De = den / (len(allv) * (len(allv) - 1)) if len(allv) > 1 else 0.0

    return 1.0 - (Do / De) if De else 1.0


def bootstrap_ci(vals, rng, n_boot=N_BOOT):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return None, None
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    args = ap.parse_args()

    if not args.key.exists():
        raise SystemExit(f"missing answer key: {args.key}")
    key = json.load(open(args.key, encoding="utf-8"))
    ratings = deblind(load_ratings(args.csvs), key)
    if not ratings:
        raise SystemExit("no ratings could be matched to the answer key")

    raters = sorted({r["rater"] for r in ratings})
    systems = [s for s in key["systems"]
               if any(r["system"] == s for r in ratings)]
    print(f"{len(ratings)} ratings from {len(raters)} raters "
          f"({', '.join(raters)}) over {len(systems)} systems\n")

    # ---- agreement (reported first: it qualifies everything below) ----
    agreement = {}
    print("Inter-annotator agreement (Krippendorff's alpha)")
    for crit in ORDINAL_CRITERIA + NOMINAL_CRITERIA:
        level = "nominal" if crit in NOMINAL_CRITERIA else "ordinal"
        units = defaultdict(list)
        for r in ratings:
            if r["criterion"] != crit:
                continue
            val = r["score"] if level == "nominal" else float(r["score"])
            units[(r["item"], r["system"])].append(val)
        a = krippendorff_alpha(units, level)
        agreement[crit] = a
        if a is None:
            verdict = "insufficient overlap"
        elif a >= 0.8:
            verdict = "strong"
        elif a >= 0.67:
            verdict = "acceptable"
        else:
            verdict = "TOO LOW to draw conclusions"
        print(f"  {crit:<14} alpha = {a:>6.3f}  ({verdict})" if a is not None
              else f"  {crit:<14} alpha =     --  ({verdict})")
    if len(raters) < 2:
        print("\n  NOTE: agreement needs at least two raters; with one rater the "
              "means below carry no reliability estimate.")

    # ---- per-system means ----
    rng = np.random.default_rng(SEED)
    summary = defaultdict(dict)
    print("\nPer-system means (95% bootstrap CI over items)")
    for crit in ORDINAL_CRITERIA:
        print(f"\n  {crit}")
        for s in systems:
            # average the raters for each item first, so a rater who rated more
            # items does not get more weight
            per_item = defaultdict(list)
            for r in ratings:
                if r["system"] == s and r["criterion"] == crit:
                    per_item[r["item"]].append(float(r["score"]))
            vals = [np.mean(v) for v in per_item.values()]
            if not vals:
                continue
            lo, hi = bootstrap_ci(vals, rng)
            summary[crit][s] = {"mean": float(np.mean(vals)), "n_items": len(vals),
                                "ci_low": lo, "ci_high": hi}
            ci = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "--"
            print(f"    {s:<18}{np.mean(vals):>5.2f}  {ci}  (n={len(vals)})")

    # harm flag as a rate
    print("\n  harmful (rate of 'yes')")
    for s in systems:
        per_item = defaultdict(list)
        for r in ratings:
            if r["system"] == s and r["criterion"] == "harmful":
                per_item[r["item"]].append(1.0 if r["score"] == "yes" else 0.0)
        vals = [np.mean(v) for v in per_item.values()]
        if vals:
            summary["harmful"][s] = {"mean": float(np.mean(vals)),
                                     "n_items": len(vals)}
            print(f"    {s:<18}{np.mean(vals):>5.2f}  (n={len(vals)})")

    # ---- paired comparisons ----
    print("\nPaired comparisons (Wilcoxon signed-rank, per criterion)")
    comparisons = []
    for crit in ORDINAL_CRITERIA:
        for a, b in itertools.combinations(systems, 2):
            pa, pb = defaultdict(list), defaultdict(list)
            for r in ratings:
                if r["criterion"] != crit:
                    continue
                if r["system"] == a:
                    pa[r["item"]].append(float(r["score"]))
                elif r["system"] == b:
                    pb[r["item"]].append(float(r["score"]))
            common = sorted(set(pa) & set(pb))
            if len(common) < 5:
                continue
            x = np.array([np.mean(pa[i]) for i in common])
            y = np.array([np.mean(pb[i]) for i in common])
            if np.allclose(x, y):
                continue
            try:
                p = float(stats.wilcoxon(x, y).pvalue)
            except ValueError:
                continue
            comparisons.append({"criterion": crit, "system_a": a, "system_b": b,
                                "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                                "n": len(common), "p": p})
            print(f"  {crit:<14}{a} ({x.mean():.2f}) vs {b} ({y.mean():.2f})"
                  f"  n={len(common)}  p={p:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_ratings": len(ratings), "raters": raters, "systems": systems,
               "agreement": agreement, "summary": {k: v for k, v in summary.items()},
               "comparisons": comparisons},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
