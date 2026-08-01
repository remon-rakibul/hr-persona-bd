#!/usr/bin/env python3
"""
Clean contaminated answers in the validated ChatML dataset.

The upstream generation/validation pipeline appended raw PDF boilerplate (the
Act's title page and preamble) and dangling statute fragments to a subset of
answers. These pollute reference-based metrics (BLEU/ROUGE) and make training
references noisy. This script:

  1. Truncates each assistant answer at the first occurrence of a known
     boilerplate marker (Act preamble, gazette date line, "WHEREAS ...").
  2. Removes obvious trailing statute dumps ("Notwithstanding anything in
     sub-section ..." style fragments) appended after a complete short answer.
  3. Trims dangling punctuation/whitespace and flags answers that become empty
     or suspiciously short (for manual review) and answers that remain very long.

Outputs a cleaned dataset plus a JSON report of what changed.

Usage:
    python scripts/clean_dataset.py \
        --input data/final/bangladesh_labour_act_chatml_validated.json \
        --output data/final/bangladesh_labour_act_chatml_clean.json \
        --report data/final/clean_report.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Markers that indicate the start of appended PDF boilerplate (title page /
# preamble). We cut the answer at the earliest marker found. Ordered longest /
# most-specific first for readability; matching is by position, not order.
BOILERPLATE_MARKERS = [
    r"XLII\s+OF\s+2006\]",
    r"XLII\s+OF\s+2006",
    r"\[ACT\s+NO",
    r"An\s+Act\s+to\s+amend\s+and\s+consolidate",
    r"WHEREAS\s+it\s+is\s+expedient",
    r"\[\s*11\s+October,?\s*2006\s*\]",
    r"BE\s+it\s+enacted",
    # Title-page repetition: "... Bangladesh Labour Act, 2006 THE BANGLADESH ..."
    r"Bangladesh\s+Labour\s+Act,?\s+2006\s+THE\s+BANGLADESH",
    # Title line followed immediately by a bracket ("...ACT, 2006 [ACT NO...").
    r"THE\s+BANGLADESH\s+LABOUR\s+ACT,?\s+2006\s*\[",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_MARKERS), re.IGNORECASE)

# Signatures of verbatim statute body text dumped after a real answer. Unlike a
# legitimately long answer, these are copied straight from the PDF and carry
# tell-tale artifacts: page running-headers, orphaned consecutive subsection
# labels "(3) (4) (5)", and verbatim statutory clauses. Cut at the earliest.
STATUTE_DUMP_MARKERS = [
    r"\s\d+\s+Bangladesh\s+Labour\s+Act,?\s+2006\b",          # page running header
    r"\([0-9a-z]+\)\s+\([0-9a-z]+\)\s+\([0-9a-z]+\)",         # orphaned (n) (n) (n)
    r"Notwithstanding\s+anything\s+contained\s+in\s+sub-?section",
    r"In\s+this\s+Act,\s+unless\s+there\s+is\s+anything\s+repugnant",
    r"this\s+Act\s+shall\s+not\s+apply\s+to\s+the\s+following",
    r"\b[IVXLC]+\s+of\s+1[89]\d\d\)",            # colonial-act cross-ref "V of 1908)"
    r"\(\d+\)\s*[“”\"']",              # numbered statutory definition (21) "shop"
]
STATUTE_DUMP_RE = re.compile("|".join(STATUTE_DUMP_MARKERS), re.IGNORECASE)

# Trailing statute fragments that get appended after an otherwise complete
# answer. Only cut when they appear after enough real answer text (guarded in
# code) so we do not truncate answers that legitimately use these phrases.
TRAILING_FRAGMENT_RE = re.compile(
    r"\s+(?:Notwithstanding\s+anything\s+in\s+sub-?section|"
    r"Provided\s+that\s+nothing\s+in\s+this\s+section)\b",
    re.IGNORECASE,
)

SHORT_FLAG = 15   # answers shorter than this after cleaning are flagged
LONG_FLAG = 800   # answers longer than this after cleaning are flagged


def clean_answer(text: str):
    """Return (cleaned_text, action) where action describes what was done."""
    original = text
    action = "none"

    m = BOILERPLATE_RE.search(text)
    if m and m.start() > 0:
        text = text[: m.start()].rstrip()
        action = "boilerplate_truncated"

    # Cut appended statute-body dumps (a real answer precedes the artifact).
    sd = STATUTE_DUMP_RE.search(text)
    if sd and sd.start() > 0:
        text = text[: sd.start()].rstrip()
        action = "statute_dump_cut" if action == "none" else action + "+dump"

    # Only strip a trailing statute fragment if there is a substantial answer
    # before it (avoids nuking short answers that are themselves such a clause).
    tm = TRAILING_FRAGMENT_RE.search(text)
    if tm and tm.start() >= 60:
        text = text[: tm.start()].rstrip()
        action = "trailing_fragment_stripped" if action == "none" else action + "+fragment"

    # Tidy dangling punctuation and whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[\s,;:]+$", "", text).strip()
    # Re-add a period if the answer looks like a complete sentence without one.
    if text and text[-1] not in ".!?)]\"'":
        text += "."

    if text == original:
        action = "none"
    return text, action


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i",
                    default="data/final/bangladesh_labour_act_chatml_validated.json")
    ap.add_argument("--output", "-o",
                    default="data/final/bangladesh_labour_act_chatml_clean.json")
    ap.add_argument("--report", "-r",
                    default="data/final/clean_report.json")
    args = ap.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input not found: {args.input}")
        sys.exit(1)

    data = json.load(open(args.input, encoding="utf-8"))
    print(f"Loaded {len(data)} items from {args.input}")

    report = {"total": len(data), "changed": 0, "boilerplate": 0,
              "statute_dump": 0, "fragment": 0,
              "flag_short": [], "flag_long": [], "examples": []}
    out = []

    for idx, item in enumerate(data):
        msgs = item.get("messages", [])
        new_msgs = []
        changed_here = False
        for msg in msgs:
            if msg.get("role") == "assistant":
                cleaned, action = clean_answer(msg.get("content", ""))
                if action != "none":
                    changed_here = True
                    if "boilerplate" in action:
                        report["boilerplate"] += 1
                    if "dump" in action:
                        report["statute_dump"] += 1
                    if "fragment" in action:
                        report["fragment"] += 1
                    if len(report["examples"]) < 10:
                        report["examples"].append({
                            "idx": idx, "action": action,
                            "before": msg["content"][:200],
                            "after": cleaned[:200],
                        })
                if len(cleaned) < SHORT_FLAG:
                    report["flag_short"].append(idx)
                if len(cleaned) > LONG_FLAG:
                    report["flag_long"].append(idx)
                new_msgs.append({"role": "assistant", "content": cleaned})
            else:
                new_msgs.append(msg)
        if changed_here:
            report["changed"] += 1
        out.append({"messages": new_msgs})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    json.dump(report, open(args.report, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"\nCleaned dataset -> {args.output}")
    print(f"Report          -> {args.report}")
    print(f"  Items changed:            {report['changed']}")
    print(f"  Boilerplate truncations:  {report['boilerplate']}")
    print(f"  Statute-dump cuts:        {report['statute_dump']}")
    print(f"  Trailing fragments:       {report['fragment']}")
    print(f"  Flagged short (<{SHORT_FLAG} ch): {len(report['flag_short'])}")
    print(f"  Flagged long (>{LONG_FLAG} ch):  {len(report['flag_long'])}")


if __name__ == "__main__":
    main()
