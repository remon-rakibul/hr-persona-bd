#!/usr/bin/env python3
"""
Build a blind human-evaluation pack for the Labour Act QA benchmark.

Automatic metrics and an LLM judge are proxies; the reviewer asked for expert
ratings from HR professionals. This script produces everything needed to collect
them defensibly:

  human_eval/rate.html       self-contained offline rating app (no network, no
                             build step - open it in a browser)
  human_eval/protocol.md     written protocol: rubric, sampling, instructions
  human_eval/answer_key.json system identities, kept OUT of the HTML

Blinding: each item shows the systems' answers under neutral labels (A, B, C...)
whose order is re-randomised per item with a fixed seed, so a rater cannot learn
"the first one is always the fine-tuned model". The mapping lives only in
answer_key.json, which raters never open. The HTML contains no system names, no
model names, and no metric scores.

Raters produce a CSV per person; score_human_eval.py (or any spreadsheet) joins
it back to the answer key.

Usage:
    python scripts/build_human_eval.py --n 40 --systems base finetuned rag_finetuned
"""

import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "results" / "generations"
OUT_DIR = ROOT / "human_eval"

# Criteria the reviewer named, phrased so a non-ML HR professional can apply them.
CRITERIA = [
    ("accuracy", "Accuracy", "Is the answer correct under the Bangladesh Labour Act 2006?"),
    ("citation", "Citation correctness", "Are the section numbers cited correct and relevant? (Score 1 if no section is cited but one was needed.)"),
    ("completeness", "Completeness", "Does it fully address what was asked, without leaving out an essential condition or exception?"),
    ("usefulness", "Usefulness", "Could you act on this answer in day-to-day HR work?"),
]
SCALE_HINT = "1 = unacceptable, 3 = adequate, 5 = excellent"

HARM_LABEL = ("harmful", "Potentially harmful / misleading",
              "Would following this answer risk a wrong or unlawful HR decision?")


def load_generations(systems, setname):
    """Return {item_id: {system: prediction}} plus the item metadata."""
    by_id = defaultdict(dict)
    meta = {}
    for s in systems:
        p = GEN_DIR / f"{s}__{setname}.json"
        if not p.exists():
            raise SystemExit(f"missing generations: {p} (run evaluate.py first)")
        for r in json.load(open(p, encoding="utf-8")):
            pred = r.get("prediction") or ""
            if pred.startswith("[GEN_ERROR]"):
                continue
            by_id[r["id"]][s] = pred
            meta.setdefault(r["id"], {"question": r.get("question"),
                                      "reference": r.get("reference"),
                                      "topic": r.get("topic"),
                                      "gold_sections": r.get("gold_sections")})
    # Only items where every system produced an answer, so each is rated on the
    # same footing.
    complete = {i: d for i, d in by_id.items() if len(d) == len(systems)}
    return complete, meta


def build_items(systems, setname, n, seed):
    by_id, meta = load_generations(systems, setname)
    rng = random.Random(seed)
    ids = sorted(by_id)
    rng.shuffle(ids)
    ids = ids[:n]

    items, key = [], []
    for idx, iid in enumerate(ids):
        order = list(systems)
        rng.shuffle(order)          # re-randomised per item
        labels = [chr(ord("A") + k) for k in range(len(order))]
        items.append({
            "item": idx,
            "topic": meta[iid].get("topic"),
            "question": meta[iid]["question"],
            "reference": meta[iid].get("reference"),
            "answers": [{"label": lab, "text": by_id[iid][sysname]}
                        for lab, sysname in zip(labels, order)],
        })
        key.append({"item": idx, "source_id": iid, "set": setname,
                    "mapping": dict(zip(labels, order))})
    return items, key


CSS = """
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
 background:#f6f7f9;color:#14171a;line-height:1.55}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #dde1e6;
 padding:14px 24px;z-index:10;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:650}
.wrap{max-width:920px;margin:0 auto;padding:24px}
.card{background:#fff;border:1px solid #dde1e6;border-radius:10px;padding:20px;
 margin-bottom:22px}
.q{font-weight:600;font-size:16px;margin:0 0 6px}
.topic{display:inline-block;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
 color:#5b6570;background:#eef1f4;border-radius:4px;padding:2px 7px;margin-bottom:10px}
.ans{border:1px solid #e3e7eb;border-radius:8px;padding:14px;margin:14px 0;background:#fcfcfd}
.ans h3{margin:0 0 8px;font-size:13px;letter-spacing:.08em;color:#38414a}
.text{white-space:pre-wrap;font-size:14px;margin-bottom:12px}
table{width:100%;border-collapse:collapse;font-size:13px}
td{padding:5px 4px;vertical-align:middle}
td.crit{width:38%;color:#38414a}
input[type=radio]{margin:0 3px 0 9px}
.harm{color:#a3341f}
.ref{font-size:13px;color:#4a545e;background:#f3f6f9;border-left:3px solid #c6d0da;
 padding:9px 12px;border-radius:0 6px 6px 0;margin-top:6px}
details summary{cursor:pointer;font-size:13px;color:#41729f;font-weight:500}
button{background:#1f6feb;color:#fff;border:0;border-radius:7px;padding:9px 16px;
 font-size:14px;cursor:pointer;font-weight:550}
button:hover{background:#1a5fd0}
input[type=text]{padding:7px 10px;border:1px solid #ccd2d9;border-radius:6px;font-size:13px}
#progress{font-size:13px;color:#5b6570;margin-left:auto}
@media (prefers-color-scheme:dark){
 body{background:#14171a;color:#e6e9ec}
 header,.card{background:#1c2024;border-color:#2c3238}
 .ans{background:#20252a;border-color:#2c3238}
 .topic{background:#252b31;color:#9aa5b1}
 .ref{background:#20252a;border-left-color:#3a434c;color:#aeb8c2}
 .ans h3{color:#c3cbd3} td.crit{color:#c3cbd3}
 input[type=text]{background:#20252a;border-color:#39414a;color:#e6e9ec}
}
"""

JS = """
const ITEMS = __ITEMS__;
const CRITERIA = __CRITERIA__;
const HARM = __HARM__;

function esc(s){const d=document.createElement('div');d.textContent=s==null?'':s;return d.innerHTML;}

function render(){
  const root = document.getElementById('items');
  ITEMS.forEach(it => {
    const card = document.createElement('div');
    card.className = 'card';
    let h = '';
    if (it.topic) h += `<span class="topic">${esc(it.topic)}</span>`;
    h += `<p class="q">Q${it.item+1}. ${esc(it.question)}</p>`;
    if (it.reference) {
      h += `<details><summary>Show reference answer (open only after forming your own view)</summary>
            <div class="ref">${esc(it.reference)}</div></details>`;
    }
    it.answers.forEach(a => {
      h += `<div class="ans"><h3>ANSWER ${a.label}</h3>
            <div class="text">${esc(a.text)}</div><table>`;
      CRITERIA.forEach(c => {
        h += `<tr><td class="crit">${esc(c[1])}<br><small>${esc(c[2])}</small></td><td>`;
        for (let v=1; v<=5; v++)
          h += `<label><input type="radio" name="${it.item}|${a.label}|${c[0]}" value="${v}">${v}</label>`;
        h += `</td></tr>`;
      });
      h += `<tr><td class="crit harm">${esc(HARM[1])}<br><small>${esc(HARM[2])}</small></td><td>`;
      ['no','yes'].forEach(v =>
        h += `<label><input type="radio" name="${it.item}|${a.label}|harmful" value="${v}">${v}</label>`);
      h += `</td></tr></table></div>`;
    });
    card.innerHTML = h;
    root.appendChild(card);
  });
  root.addEventListener('change', save);
  restore(); updateProgress();
}

function key(){ return 'hreval_' + (document.getElementById('rater').value || 'anon'); }

function save(){
  const data = {};
  document.querySelectorAll('input[type=radio]:checked').forEach(i => data[i.name] = i.value);
  localStorage.setItem(key(), JSON.stringify(data));
  updateProgress();
}

function restore(){
  const raw = localStorage.getItem(key());
  if (!raw) return;
  const data = JSON.parse(raw);
  Object.entries(data).forEach(([n,v]) => {
    const el = document.querySelector(`input[name="${CSS.escape(n)}"][value="${v}"]`);
    if (el) el.checked = true;
  });
}

function updateProgress(){
  const total = ITEMS.reduce((s,it) => s + it.answers.length * (CRITERIA.length+1), 0);
  const done = document.querySelectorAll('input[type=radio]:checked').length;
  document.getElementById('progress').textContent = `${done} / ${total} ratings complete`;
}

function download(){
  const rater = document.getElementById('rater').value || 'anon';
  const rows = [['rater','item','answer_label','criterion','score']];
  document.querySelectorAll('input[type=radio]:checked').forEach(i => {
    const [item,label,crit] = i.name.split('|');
    rows.push([rater,item,label,crit,i.value]);
  });
  const csv = rows.map(r => r.map(c => `"${String(c).replace(/"/g,'""')}"`).join(',')).join('\\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download = `human_eval_${rater}.csv`;
  a.click();
}

document.addEventListener('DOMContentLoaded', () => {
  render();
  document.getElementById('rater').addEventListener('input', () => { restore(); updateProgress(); });
  document.getElementById('dl').addEventListener('click', download);
});
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Labour Act QA - blind rating</title>
<style>__CSS__</style></head>
<body>
<header>
  <h1>Bangladesh Labour Act QA &mdash; blind rating</h1>
  <label style="font-size:13px">Your name/ID:
    <input type="text" id="rater" placeholder="rater1"></label>
  <button id="dl">Download CSV</button>
  <span id="progress"></span>
</header>
<div class="wrap">
  <div class="card">
    <strong>Instructions.</strong> Each question below is followed by several
    answers labelled A, B, C&hellip; produced by different systems. You are not
    told which system produced which answer, and the order changes on every
    question. Rate each answer independently on the scale
    <em>__SCALE__</em>. Work top to bottom and do not go back to
    &ldquo;even out&rdquo; your scores. Your progress is saved in this browser
    automatically; click <em>Download CSV</em> when you are finished and send the
    file back.
  </div>
  <div id="items"></div>
  <div class="card" style="text-align:center">
    <button id="dl2" onclick="download()">Download CSV</button>
  </div>
</div>
<script>__JS__</script>
</body></html>
"""


def build_html(items):
    js = (JS.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
            .replace("__CRITERIA__", json.dumps(CRITERIA, ensure_ascii=False))
            .replace("__HARM__", json.dumps(HARM_LABEL, ensure_ascii=False)))
    return (PAGE.replace("__CSS__", CSS).replace("__JS__", js)
                .replace("__SCALE__", html.escape(SCALE_HINT)))


PROTOCOL = """# Human evaluation protocol

## Purpose
Automatic metrics (BLEU, ROUGE-L, lexical grounding) and the LLM judge are
proxies. This protocol collects expert judgements from HR professionals on the
dimensions that matter for a statute-grounded assistant: accuracy, citation
correctness, completeness, usefulness, and potential for harm.

## Raters
Target 3 raters with HR or labour-law experience in Bangladesh. Raters must not
have been involved in building the dataset or the model.

## Materials
- `rate.html` - the blind rating application. It is fully offline: open it in any
  browser, no installation or internet connection is required.
- Raters receive **only** `rate.html`. They must not receive `answer_key.json`.

## Blinding
Each question is shown with one answer per system under neutral labels
(A, B, C, ...). The label-to-system assignment is randomised independently for
every question, so position carries no information. The HTML contains no system
names, model names, or automatic scores. The mapping is stored separately in
`answer_key.json` and is joined to the ratings only after collection.

## Sample
{n} questions sampled with a fixed seed ({seed}) from the {setname} test set,
covering the HR topics represented in that set. Each rater rates every question,
so all raters see the same items (required for the agreement statistic).

## Rating scale
{scale}

| Criterion | Question put to the rater |
|---|---|
{criteria_rows}
| Potentially harmful / misleading | Would following this answer risk a wrong or unlawful HR decision? (yes/no) |

The reference answer is available behind a collapsed control on each question.
Raters are instructed to form their own view first and open it only to check a
specific point, so that ratings are not anchored to the reference wording.

## Procedure
1. Rater enters an identifier, then works through the questions top to bottom.
2. Progress is saved to browser local storage, so the task can be done in
   several sittings on the same machine and browser.
3. On completion the rater clicks *Download CSV* and returns the file.

## Analysis
- Join each rater CSV to `answer_key.json` on `item` + `answer_label` to recover
  the system identity.
- Report the mean per criterion per system, with 95% confidence intervals over
  items (bootstrap, 10 000 resamples).
- Report inter-annotator agreement with Krippendorff's alpha (ordinal level) for
  the 1-5 criteria and nominal level for the harm flag. Report alpha alongside
  the means: means from raters who do not agree are not interpretable.
- Compare systems with a paired test over items (Wilcoxon signed-rank), since
  every system answers the same questions.

## Reporting
State the number of raters, their background, the agreement statistic, and the
sample size. Human ratings are reported separately from automatic metrics and
are never mixed into the same column.
"""


def write_protocol(n, seed, setname):
    rows = "\n".join(f"| {label} | {desc} |" for _, label, desc in CRITERIA)
    return PROTOCOL.format(n=n, seed=seed, setname=setname, scale=SCALE_HINT,
                           criteria_rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="*",
                    default=["base", "finetuned", "rag_finetuned"])
    ap.add_argument("--set", dest="setname", default="scenario",
                    help="test set to sample from (default: scenario)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    items, key = build_items(args.systems, args.setname, args.n, args.seed)
    if not items:
        raise SystemExit("no items with answers from every system")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rate.html").write_text(build_html(items), encoding="utf-8")
    json.dump({"systems": args.systems, "set": args.setname, "seed": args.seed,
               "items": key},
              open(args.out / "answer_key.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    (args.out / "protocol.md").write_text(
        write_protocol(len(items), args.seed, args.setname), encoding="utf-8")

    n_ratings = len(items) * len(args.systems) * (len(CRITERIA) + 1)
    print(f"Wrote {args.out}/rate.html ({len(items)} questions x "
          f"{len(args.systems)} systems = {n_ratings} ratings per rater)")
    print(f"Wrote {args.out}/answer_key.json  <-- do NOT send this to raters")
    print(f"Wrote {args.out}/protocol.md")


if __name__ == "__main__":
    main()
