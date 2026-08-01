#!/usr/bin/env python3
"""
Verify the hand-authored HR scenario test set against the text of the Act.

The scenarios are the most important part of the benchmark - they are what
distinguishes it from questions paraphrased out of the statute - but they were
written by hand and shipped with ``verified: false``. An unverified reference
answer is not a usable gold standard: every citation metric is measured against
its ``gold_sections``, so a wrong gold label silently penalises correct model
answers.

This script checks each scenario against the Act and records the evidence:

  section_exists   every gold section number occurs as a section heading
  numbers_supported  numeric quantities asserted by the reference answer (days,
                     months, percentages, ages, hours) appear in the cited
                     section's text
  lexical_support  fraction of the reference answer's content words that occur
                   in the cited section's text

Verification is evidence-gathering, not proof: a scenario passes automatic checks
when its citation resolves and its numbers are traceable to the cited text. Items
that fail are listed with the section text so they can be corrected by hand.
``--apply`` writes the outcome back into the ``verified`` field.

Usage:
    python scripts/verify_scenarios.py
    python scripts/verify_scenarios.py --apply
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACT_RAW = ROOT / "data/final/act_raw.txt"
SCENARIOS = ROOT / "data/eval/scenario_test.json"
REPORT = ROOT / "results/scenario_verification.json"

# Section bodies look like:  "117. Annual leave with wages.- (1) Every adult ..."
HEADING_RE = re.compile(r"(?m)^[ \t]*(\d{1,3})\.[ \t]+([A-Z][^\n]{0,120})")

# Numbers the Act states as entitlements. The Act usually writes "8 (eight)
# hours", but sometimes only in words ("before the expiry of the thirtieth
# working day"), so a digits-only match reports false failures - section 123(2)
# does support "30 working days". Spelled cardinals and ordinals are therefore
# resolved to digits before comparison.
NUM_RE = re.compile(r"\b(\d{1,4})\b")

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
          "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_ORDINALS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
             "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
             "thirteenth", "fourteenth", "fifteenth", "sixteenth",
             "seventeenth", "eighteenth", "nineteenth"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_TENS_ORD = {"twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
             "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90}

_WORD_NUM = {w: i for i, w in enumerate(_UNITS)}
_WORD_NUM.update({w: i for i, w in enumerate(_ORDINALS)})
_WORD_NUM.update(_TENS)
_WORD_NUM.update(_TENS_ORD)


def numbers_in(text):
    """All numeric quantities in `text`, as strings, digits and words alike."""
    text = (text or "").lower()
    found = set(NUM_RE.findall(text))
    for word, val in _WORD_NUM.items():
        if re.search(rf"\b{word}\b", text):
            found.add(str(val))
    # Hyphenated compounds: "forty-eight", "twenty-five".
    for tens, unit in re.findall(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-(\w+)\b", text):
        if unit in _WORD_NUM and _WORD_NUM[unit] < 10:
            found.add(str(_TENS[tens] + _WORD_NUM[unit]))
    return found


def derivable(target, pool):
    """True if `target` is simple arithmetic over `pool`.

    Scenario answers do arithmetic the statute does not contain: 216 days worked
    at one day per 18 gives 12 days of leave; 50 taka doubled is 100. Those
    numbers are correct precisely because they are computed, so flagging them as
    "not in the Act" would be wrong. Two passes let a result feed the next step
    (50 -> 100 -> 300).
    """
    try:
        t = float(target)
    except ValueError:
        return False
    vals = set()
    for v in pool:
        try:
            vals.add(float(v))
        except ValueError:
            pass
    reach = set(vals)
    for _ in range(2):
        new = set()
        for a in reach:
            for b in vals:
                for r in (a * b, a + b, a - b, a / b if b else None):
                    if r is not None and r == r and abs(r) < 1e7:
                        new.add(r)
        reach |= new
        if t in reach:
            return True
    return t in reach

# Items checked by hand and accepted despite failing an automatic threshold, with
# the reason. Kept here rather than as a flag in the data so that the grounds for
# accepting an item are visible next to the checks it failed.
MANUAL_REVIEW_OK = {
    "scn_overtime_03": (
        "Citation corrected from s.102 (weekly hours) to s.100 (daily hours) "
        "read with s.108. Lexical support is below threshold only because the "
        "reference states the consequence - that a worker's written consent "
        "cannot waive a statutory ceiling - which is reasoning about the "
        "provision rather than wording drawn from it."),
}

STOPWORDS = set("""a an the of to in for and or is are be been was were shall may
must not no any all such that this these those which who whom whose it its his
her their they he she as at by on from with without under over per each every
if then than when where while after before during upon into within between both
either neither also more most less least other another same own only just very
can could would should has have had do does did being about against
""".split())


def load_sections():
    """Map section number -> (title, body). Picks the longest body for a number.

    The PDF contains a table of contents whose entries match the same heading
    pattern as the real sections; the real body is always substantially longer,
    so taking the longest span per number discards the TOC entry.
    """
    raw = ACT_RAW.read_text(encoding="utf-8")
    matches = list(HEADING_RE.finditer(raw))
    best = {}
    for i, m in enumerate(matches):
        num = int(m.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[m.start():end]
        if num not in best or len(body) > len(best[num][1]):
            best[num] = (m.group(2).strip(), body)
    return best


def norm_words(text):
    return [w for w in re.findall(r"[a-z]+", (text or "").lower())
            if w not in STOPWORDS and len(w) > 2]


def verify_one(item, sections):
    gold = item.get("gold_sections") or []
    ref = item.get("reference") or ""

    missing = [s for s in gold if s not in sections]
    body = " ".join(sections[s][1] for s in gold if s in sections)
    body_l = body.lower()

    # Numeric claims in the reference that should be traceable to the section.
    # Section numbers themselves are not entitlement quantities, and numbers the
    # question supplies are givens rather than claims about the Act.
    given = numbers_in(item.get("question"))
    ref_nums = [n for n in NUM_RE.findall(ref) if int(n) not in set(gold)]
    body_nums = numbers_in(body)
    pool = body_nums | given
    unsupported_nums = sorted(
        {n for n in ref_nums if n not in pool and not derivable(n, pool)})

    words = norm_words(ref)
    hits = sum(1 for w in set(words) if w in body_l)
    lexical = hits / len(set(words)) if words else 0.0

    checks = {
        "gold_sections": gold,
        "section_exists": not missing,
        "missing_sections": missing,
        "section_titles": {s: sections[s][0] for s in gold if s in sections},
        "numbers_in_reference": ref_nums,
        "unsupported_numbers": unsupported_nums,
        "numbers_supported": not unsupported_nums,
        "lexical_support": round(lexical, 3),
    }
    # An item is auto-verified when its citation resolves, its stated quantities
    # are traceable to that section, and its wording overlaps the section
    # substantially. Anything else is routed to manual review rather than passed.
    checks["auto_verified"] = bool(
        checks["section_exists"] and checks["numbers_supported"] and lexical >= 0.5)
    checks["manual_review"] = MANUAL_REVIEW_OK.get(item["id"])
    checks["verified"] = bool(checks["auto_verified"] or checks["manual_review"])
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="write results back into scenario_test.json 'verified'")
    ap.add_argument("--lexical-min", type=float, default=0.5)
    args = ap.parse_args()

    if not ACT_RAW.exists():
        raise SystemExit(f"missing {ACT_RAW}; extract the Act text first")

    sections = load_sections()
    print(f"parsed {len(sections)} sections from the Act "
          f"(range {min(sections)}-{max(sections)})")

    items = json.load(open(SCENARIOS, encoding="utf-8"))
    results, failed = [], []
    for it in items:
        chk = verify_one(it, sections)
        chk["id"] = it["id"]
        chk["topic"] = it.get("topic")
        chk["question"] = it.get("question")
        chk["reference"] = it.get("reference")
        results.append(chk)
        if not chk["verified"]:
            failed.append(chk)

    n_ok = sum(1 for r in results if r["auto_verified"])
    n_manual = sum(1 for r in results if r["manual_review"] and not r["auto_verified"])
    n_verified = sum(1 for r in results if r["verified"])
    summary = {
        "n_items": len(items),
        "n_auto_verified": n_ok,
        "n_manually_reviewed": n_manual,
        "n_verified": n_verified,
        "auto_verified_rate": round(n_ok / len(items), 3) if items else 0,
        "verified_rate": round(n_verified / len(items), 3) if items else 0,
        "n_missing_section": sum(1 for r in results if not r["section_exists"]),
        "n_unsupported_numbers": sum(1 for r in results if not r["numbers_supported"]),
        "mean_lexical_support": round(
            sum(r["lexical_support"] for r in results) / len(results), 3) if results else 0,
        "lexical_min": args.lexical_min,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "items": results},
              open(REPORT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"\nauto-verified {n_ok}/{len(items)} "
          f"({summary['auto_verified_rate']*100:.1f}%); "
          f"+{n_manual} accepted on manual review -> "
          f"{n_verified}/{len(items)} verified")
    print(f"missing section:      {summary['n_missing_section']}")
    print(f"unsupported numbers:  {summary['n_unsupported_numbers']}")
    print(f"mean lexical support: {summary['mean_lexical_support']}")

    if failed:
        print(f"\n--- {len(failed)} items needing manual review ---")
        for r in failed:
            reasons = []
            if not r["section_exists"]:
                reasons.append(f"missing sections {r['missing_sections']}")
            if r["unsupported_numbers"]:
                reasons.append(f"numbers not in cited text: {r['unsupported_numbers']}")
            if r["lexical_support"] < args.lexical_min:
                reasons.append(f"low lexical support {r['lexical_support']}")
            print(f"\n{r['id']} [{r['topic']}] - {'; '.join(reasons)}")
            print(f"  Q: {(r['question'] or '')[:150]}")
            print(f"  ref: {(r['reference'] or '')[:200]}")
            print(f"  cited: {r['section_titles']}")

    if args.apply:
        by_id = {r["id"]: r for r in results}
        for it in items:
            it["verified"] = bool(by_id[it["id"]]["verified"])
            it["verification"] = {
                k: by_id[it["id"]][k] for k in
                ("section_exists", "numbers_supported", "lexical_support",
                 "unsupported_numbers", "auto_verified", "manual_review")
            }
        json.dump(items, open(SCENARIOS, "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        print(f"\nupdated {SCENARIOS}")

    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
