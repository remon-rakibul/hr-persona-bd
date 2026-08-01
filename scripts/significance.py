#!/usr/bin/env python3
"""
Paired significance testing between systems on the Labour Act benchmark.

A table of means says system A scored higher than system B; it does not say
whether that gap survives the noise of a 150-item test set. With 150 items and
five systems, differences of a point or two of ROUGE are entirely consistent with
chance, and reporting them as findings is how a comparison becomes an overclaim.

Every system answers exactly the same questions, so the comparison is *paired*:
the per-item difference is the unit of analysis, which removes item difficulty
from the variance and is far more sensitive than comparing two independent means.

For each metric and each system versus the reference system, this reports:

  mean difference          the effect, in the metric's own units
  95% bootstrap CI         paired, resampling items (not answers) 10 000 times
  Wilcoxon signed-rank p   distribution-free; judge scores are ordinal, not
                           interval, so a t-test's assumptions do not hold
  Holm-adjusted p          controls family-wise error across the systems
                           compared against the reference within each metric
  rank-biserial r          effect size, so a significant-but-tiny difference is
                           visible as such

A difference is called significant only when the Holm-adjusted p is below 0.05
*and* the bootstrap CI excludes zero - the conservative conjunction of the two.

Note that the reported CI is unadjusted and descriptive: it is possible for a CI
to exclude zero while the adjusted p exceeds 0.05, because the adjustment pays
for testing several systems at once. Such a case is reported as not significant.
Read the CI as the plausible range of the effect, and the adjusted p as whether
the comparison survives multiplicity.

Statistical significance is not practical importance. A small difference that is
perfectly consistent across items will be significant; the mean difference and
its CI, not the significance flag, say whether the gap matters.

Usage:
    python scripts/significance.py
    python scripts/significance.py --set scenario --reference base
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SCORE_DIR = ROOT / "results" / "scored"
RESULTS = ROOT / "results"

# (key in the scored record, human label, higher-is-better)
METRICS = [
    ("bleu", "BLEU", True),
    ("rougeL", "ROUGE-L", True),
    ("grounding", "Grounding", True),
    ("cite_f1", "Citation F1", True),
    ("cited_valid", "Cited sections valid", True),
    ("faithfulness", "Faithfulness (judge)", True),
    ("completeness", "Completeness (judge)", True),
    ("usefulness", "Usefulness (judge)", True),
    ("harm", "Harmfulness (judge)", False),
    ("refusal", "Refusal rate", None),   # direction depends on the set
]

N_BOOT = 10000
SEED = 3407


def extract(rec, metric):
    """Pull one metric out of a scored record, or None if absent."""
    if metric in ("faithfulness", "completeness", "usefulness", "harm"):
        j = rec.get("judge")
        if not isinstance(j, dict) or j.get("parse_failed"):
            return None
        v = j.get(metric)
    elif metric in ("cite_f1", "cited_valid"):
        c = rec.get("citation")
        if not isinstance(c, dict):
            return None
        v = c.get("cite_f1" if metric == "cite_f1" else "cited_valid_frac")
    elif metric == "refusal":
        v = 1.0 if rec.get("refusal") else 0.0
    else:
        v = rec.get(metric)
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) else None


def load_system(system, setname):
    p = SCORE_DIR / f"{system}__{setname}.json"
    if not p.exists():
        return None
    return {r["id"]: r for r in json.load(open(p, encoding="utf-8"))}


def paired_bootstrap_ci(diffs, rng, n_boot=N_BOOT, alpha=0.05):
    """Percentile CI of the mean paired difference, resampling items."""
    if len(diffs) < 2:
        return None, None
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def compare(setname, reference, systems):
    rng = np.random.default_rng(SEED)
    loaded = {s: load_system(s, setname) for s in systems}
    loaded = {s: d for s, d in loaded.items() if d}
    if reference not in loaded:
        raise SystemExit(f"reference system '{reference}' has no scored results "
                         f"for set '{setname}'")
    others = [s for s in loaded if s != reference]
    if not others:
        raise SystemExit("need at least two systems with scored results")

    rows = []
    for metric, label, higher_better in METRICS:
        raw = []
        for s in others:
            ids = [i for i in loaded[reference] if i in loaded[s]]
            pairs = []
            for i in ids:
                a = extract(loaded[reference][i], metric)
                b = extract(loaded[s][i], metric)
                if a is not None and b is not None:
                    pairs.append((a, b))
            if len(pairs) < 5:
                continue
            ref_v = np.array([p[0] for p in pairs], dtype=float)
            sys_v = np.array([p[1] for p in pairs], dtype=float)
            diffs = sys_v - ref_v
            if np.allclose(diffs, 0):
                raw.append({"system": s, "metric": metric, "label": label,
                            "n": len(diffs), "ref_mean": float(ref_v.mean()),
                            "sys_mean": float(sys_v.mean()), "mean_diff": 0.0,
                            "ci_low": 0.0, "ci_high": 0.0, "p": 1.0,
                            "effect_r": 0.0, "identical": True})
                continue
            lo, hi = paired_bootstrap_ci(diffs, rng)
            try:
                w = stats.wilcoxon(sys_v, ref_v, zero_method="wilcox",
                                   alternative="two-sided")
                p = float(w.pvalue)
                nz = int(np.count_nonzero(diffs))
                # rank-biserial correlation for the signed-rank test
                r = float(1 - (2 * w.statistic) / (nz * (nz + 1) / 2)) if nz else 0.0
            except ValueError:
                p, r = 1.0, 0.0
            raw.append({"system": s, "metric": metric, "label": label,
                        "n": len(diffs), "ref_mean": float(ref_v.mean()),
                        "sys_mean": float(sys_v.mean()),
                        "mean_diff": float(diffs.mean()),
                        "ci_low": lo, "ci_high": hi, "p": p, "effect_r": r,
                        "identical": False})
        # Holm across the systems compared for this metric.
        if raw:
            adj = holm([x["p"] for x in raw])
            for x, a in zip(raw, adj):
                x["p_holm"] = a
                x["significant"] = bool(a < 0.05 and x["ci_low"] is not None
                                        and (x["ci_low"] > 0) == (x["ci_high"] > 0))
                x["higher_better"] = higher_better
            rows.extend(raw)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="setname", default="heldout")
    ap.add_argument("--reference", default="base",
                    help="system every other system is compared against")
    ap.add_argument("--systems", nargs="*",
                    default=["base", "finetuned", "rag_base", "rag_finetuned",
                             "qwen_general"])
    args = ap.parse_args()

    rows = compare(args.setname, args.reference, args.systems)
    if not rows:
        raise SystemExit("no comparable metrics found; run evaluate.py --phase score")

    out = {"set": args.setname, "reference": args.reference,
           "n_bootstrap": N_BOOT, "seed": SEED,
           "correction": "Holm-Bonferroni within each metric",
           "comparisons": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RESULTS / f"significance_{args.setname}.json", "w"),
              indent=1)

    cols = ["metric", "system", "n", "ref_mean", "sys_mean", "mean_diff",
            "ci_low", "ci_high", "p", "p_holm", "effect_r", "significant"]
    with open(RESULTS / f"significance_{args.setname}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Paired comparison on '{args.setname}' vs '{args.reference}' "
          f"({N_BOOT} bootstrap resamples, Holm-corrected)\n")
    print(f"{'metric':<22}{'system':<16}{'n':>4}{'diff':>9}"
          f"{'95% CI':>20}{'p_holm':>9}  sig")
    last = None
    for r in rows:
        if r["metric"] != last:
            print()
            last = r["metric"]
        ci = (f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
              if r["ci_low"] is not None else "--")
        print(f"{r['label']:<22}{r['system']:<16}{r['n']:>4}"
              f"{r['mean_diff']:>+9.3f}{ci:>20}{r['p_holm']:>9.4f}"
              f"  {'yes' if r['significant'] else ''}")
    print(f"\nWrote {RESULTS/f'significance_{args.setname}.json'} and .csv")


if __name__ == "__main__":
    main()
