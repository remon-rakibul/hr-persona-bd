#!/usr/bin/env python3
"""
Evaluation runner: compare systems on the Labour Act QA benchmark.

Systems compared (all local, via Ollama):
  base           - base Llama 3.2 3B Instruct (no fine-tuning)
  finetuned      - the fine-tuned hr-persona-bd model
  rag_base       - retrieval-augmented base Llama 3.2 3B
  rag_finetuned  - retrieval-augmented fine-tuned model
  qwen_general   - Qwen2.5 7B, a larger general open model (reference point;
                   commercial APIs such as GPT/Claude/Gemini are future work)

Test sets:
  heldout   - 150 leakage-free QA pairs held out of training (has references)
  scenario  - 38 hand-authored HR scenarios (has references; verify pending)
  oos       - 20 out-of-scope probes (refusal is the desired behaviour)

Phases (each resumable; generations and judgements are cached to disk):
  generate  - produce model answers, cache to results/generations/
  score     - compute metrics (+ optional LLM judge), cache to results/scored/
  aggregate - write results/comparison.csv and results/comparison.md

Run everything on a small sample first, e.g.:
    python scripts/evaluate.py --phase all --limit 5
Then the full run (slow on small GPUs; use a background run):
    python scripts/evaluate.py --phase all
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import eval_metrics as M

# Anchored to the project root so the script works from any cwd (running it from
# scripts/ previously created a stray scripts/results/ tree).
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
GEN_DIR = RESULTS / "generations"
SCORE_DIR = RESULTS / "scored"
ACT_PDF = ROOT / "data/input/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf"

# Consistent deployment-style system prompt for all direct (non-RAG) systems,
# so refusal behaviour reflects the model, not prompt differences.
SYSTEM_PROMPT = (
    "You are an expert HR consultant providing statute-grounded informational "
    "support (not legal advice) on the Bangladesh Labour Act 2006, amended to "
    "2018. Answer questions about employment matters under the Act — wages, "
    "leave, working hours, overtime, termination, maternity benefit, misconduct, "
    "probation, safety, compensation and trade unions — and cite the relevant "
    "section when possible. If a question is outside the Bangladesh Labour Act, "
    "politely decline and say it is outside your scope."
)

# Decoding is greedy and seeded so a re-run reproduces the reported numbers
# exactly. Sampling (the previous temperature=0.7) made results unreproducible.
TEMPERATURE = 0.0
SEED = 3407
NUM_PREDICT = 384

# `heavy` systems do not fit small GPUs and must be requested explicitly: on a
# 4 GB card qwen2.5:7b (4.7 GB) offloads to CPU at ~40-120 s/item, which is what
# made a naive all-systems run appear to hang the machine.
SYSTEMS = {
    "base": {"kind": "direct", "model": "llama3.2:3b-instruct-q4_K_M"},
    "finetuned": {"kind": "direct", "model": "hr-persona-bd:latest"},
    "rag_base": {"kind": "rag", "model": "llama3.2:3b-instruct-q4_K_M"},
    "rag_finetuned": {"kind": "rag", "model": "hr-persona-bd:latest"},
    "qwen_general": {"kind": "direct", "model": "qwen2.5:7b", "heavy": True},
}
DEFAULT_SYSTEMS = [s for s, c in SYSTEMS.items() if not c.get("heavy")]

# Rough per-item wall-clock, measured on a GTX 1050 4 GB, for --dry-run.
SECS_PER_ITEM = {"direct": 11.0, "rag": 20.0}
HEAVY_SECS_PER_ITEM = 120.0

SETS = {
    "heldout": "data/eval/heldout_test.json",
    "scenario": "data/eval/scenario_test.json",
    "oos": "data/eval/out_of_scope.json",
}


def load_set(name):
    return json.load(open(ROOT / SETS[name], encoding="utf-8"))


def direct_answer(model, question, temperature=TEMPERATURE, num_predict=NUM_PREDICT):
    import ollama
    r = ollama.chat(model=model, messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ], options={"temperature": temperature, "num_predict": num_predict,
                "seed": SEED})
    return r["message"]["content"]


def run_provenance(systems):
    """Capture what produced these generations, for the reproducibility section."""
    import platform
    prov = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "temperature": TEMPERATURE, "seed": SEED, "num_predict": NUM_PREDICT,
        "system_prompt": SYSTEM_PROMPT,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "models": {},
    }
    try:
        import ollama
        listing = ollama.list()
        digests = {m.get("model") or m.get("name"): m.get("digest")
                   for m in listing.get("models", [])}
        for s in systems:
            name = SYSTEMS[s]["model"]
            prov["models"][s] = {"model": name, "digest": digests.get(name)}
    except Exception as e:
        prov["models_error"] = str(e)
    try:
        import subprocess
        prov["ollama_version"] = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True,
            timeout=10).stdout.strip()
    except Exception:
        pass
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(prov, open(RESULTS / "run_provenance.json", "w"), indent=1)
    return prov


def estimate(systems, sets, limit=None):
    """Print projected wall-clock so a long run is a deliberate choice."""
    total = 0.0
    print(f"{'system':<16}{'set':<10}{'items':>7}{'est':>12}")
    for system in systems:
        cfg = SYSTEMS[system]
        rate = (HEAVY_SECS_PER_ITEM if cfg.get("heavy")
                else SECS_PER_ITEM[cfg["kind"]])
        for setname in sets:
            n = len(load_set(setname))
            if limit:
                n = min(n, limit)
            done = 0
            gp = gen_path(system, setname)
            if gp.exists():
                done = sum(1 for r in json.load(open(gp, encoding="utf-8"))
                           if r.get("prediction"))
            todo = max(0, n - done)
            secs = todo * rate
            total += secs
            print(f"{system:<16}{setname:<10}{todo:>7}{secs/60:>10.0f}m")
    print(f"\nTOTAL remaining: {total/3600:.1f}h ({total/60:.0f}m)")
    return total


# ---------------- generate ----------------

def gen_path(system, setname):
    return GEN_DIR / f"{system}__{setname}.json"


def unload(model):
    """Evict a model from VRAM (keep_alive=0).

    On a 4 GB GPU two resident 3B models plus the judge exceed VRAM and push the
    machine into swap, so each system is evicted before the next one loads.
    """
    try:
        import ollama
        ollama.chat(model=model, messages=[{"role": "user", "content": "hi"}],
                    options={"num_predict": 1}, keep_alive=0)
    except Exception:
        pass


def phase_generate(systems, sets, limit=None):
    from rag_baseline import RagRetriever, rag_answer
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    retriever = None
    for system in systems:
        cfg = SYSTEMS[system]
        if cfg["kind"] == "rag" and retriever is None:
            retriever = RagRetriever()
        for setname in sets:
            items = load_set(setname)
            if limit:
                items = items[:limit]
            out_path = gen_path(system, setname)
            done = {}
            if out_path.exists():
                done = {r["id"]: r for r in json.load(open(out_path, encoding="utf-8"))}
            records = []
            t0 = time.time()
            for i, it in enumerate(items):
                iid = it["id"]
                q = it["question"]
                if iid in done and done[iid].get("prediction"):
                    records.append(done[iid])
                    continue
                try:
                    if cfg["kind"] == "rag":
                        pred, ctx = rag_answer(q, cfg["model"], retriever)
                    else:
                        pred, ctx = direct_answer(cfg["model"], q), None
                except Exception as e:
                    pred, ctx = f"[GEN_ERROR] {e}", None
                rec = {"id": iid, "topic": it.get("topic"), "question": q,
                       "reference": it.get("reference"),
                       "gold_sections": it.get("gold_sections", []),
                       "expected_behavior": it.get("expected_behavior"),
                       "prediction": pred}
                if ctx is not None:
                    rec["retrieved"] = ctx
                records.append(rec)
                # Persist incrementally so the run is resumable.
                json.dump(records, open(out_path, "w", encoding="utf-8"),
                          indent=1, ensure_ascii=False)
                if (i + 1) % 10 == 0:
                    rate = (time.time() - t0) / (i + 1)
                    print(f"  [{system}/{setname}] {i+1}/{len(items)} "
                          f"({rate:.1f}s/item)", flush=True)
            json.dump(records, open(out_path, "w", encoding="utf-8"),
                      indent=1, ensure_ascii=False)
            print(f"generated {system}/{setname}: {len(records)} -> {out_path}",
                  flush=True)
        unload(cfg["model"])


# ---------------- score ----------------

# Order in which items are offered to the judge. Fixed so a run is
# reproducible, shuffled so that stopping early yields an unbiased subsample
# rather than whichever topics happen to sort first.
JUDGE_ORDER_SEED = 3407


def phase_score(systems, sets, judge_model=None, act_norm=None):
    """Score every (system, set): fast metrics first, then judging.

    The two passes are separate because they have wildly different costs. The
    overlap and citation metrics are pure Python over text and finish in
    seconds; judging is a local LLM call that runs ~10-45s on a 4 GB card and
    dominates the wall clock.

    Judging walks *items* in the outer loop and systems in the inner loop, so
    every judged item is judged for all systems before moving on. This matters
    because the significance tests are paired: interrupting a system-major run
    leaves some systems fully judged and others empty, which yields no valid
    comparison at all, whereas interrupting this one leaves a smaller but fully
    paired sample that every downstream script can still use.
    """
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    if act_norm is None:
        act_norm = M.act_normalized()
    valid_sections = M.valid_sections()

    # ---- pass 1: fast metrics, and load prior judgements -------------------
    store = {}          # (system, setname) -> list of records
    for system in systems:
        for setname in sets:
            gp = gen_path(system, setname)
            if not gp.exists():
                print(f"skip (no generations): {system}/{setname}")
                continue
            recs = json.load(open(gp, encoding="utf-8"))
            sp = SCORE_DIR / f"{system}__{setname}.json"

            # Reuse judgements already on disk. Judging is the slow part and
            # this machine suspends, so an interrupted run must not redo them.
            prior = {}
            if sp.exists():
                try:
                    prior = {r["id"]: r
                             for r in json.load(open(sp, encoding="utf-8"))}
                except Exception:
                    prior = {}

            for r in recs:
                pred = r.get("prediction", "")
                r["refusal"] = M.is_refusal(pred)
                old = prior.get(r["id"], {}).get("judge")
                if isinstance(old, dict) and not old.get("parse_failed"):
                    r["judge"] = old
                if setname == "oos":
                    continue  # only refusal matters
                ref = r.get("reference") or ""
                r["bleu"] = M.bleu(pred, ref)
                r["rougeL"] = M.rouge_l(pred, ref)
                r["grounding"] = M.grounding_score(pred, act_norm)
                r["citation"] = M.citation_metrics(pred, r.get("gold_sections"),
                                                   valid_sections)
            store[(system, setname)] = recs
            _write_scored(system, setname, recs)
            print(f"scored {system}/{setname}: {len(recs)} items", flush=True)

    if not judge_model:
        return

    # ---- pass 2: judging, item-major so the sample stays paired ------------
    # Interleave across sets as well as items. Walking the sets in sequence
    # would judge all 150 hold-out items before touching the 38 scenarios, so
    # an interrupted run would have judge scores for the set the paper leans on
    # least and none for the set it leans on most. Shuffling (set, item)
    # together makes both accumulate in proportion to their size.
    units = []
    for setname in sets:
        if setname == "oos":
            continue
        present = [s for s in systems if (s, setname) in store]
        if not present:
            continue
        n = min(len(store[(s, setname)]) for s in present)
        units += [(setname, i, present) for i in range(n)]
    random.Random(JUDGE_ORDER_SEED).shuffle(units)

    todo = []
    for setname, i, present in units:
        for s in present:
            todo.append((s, setname, i))

    pending = [(s, st, i) for s, st, i in todo
               if not store[(s, st)][i].get("judge")]
    total = len(pending)
    if not total:
        print("all judgements already on disk", flush=True)
        return
    print(f"judging {total} items with {judge_model} "
          f"(item-major; safe to interrupt)", flush=True)

    t0 = time.time()
    dirty = set()
    for k, (system, setname, i) in enumerate(pending, 1):
        r = store[(system, setname)][i]
        r["judge"] = M.judge(r.get("prediction", ""), r["question"],
                             r.get("reference") or "", judge_model)
        dirty.add((system, setname))
        # Checkpoint often: losing one judgement is fine, losing an hour is not.
        if k % 5 == 0 or k == total:
            for key in dirty:
                _write_scored(key[0], key[1], store[key])
            dirty.clear()
            rate = (time.time() - t0) / k
            eta = (total - k) * rate / 3600
            print(f"  judged {k}/{total} ({rate:.1f}s/judgement, "
                  f"~{eta:.1f}h left)", flush=True)
    for key in dirty:
        _write_scored(key[0], key[1], store[key])


def _write_scored(system, setname, recs):
    sp = SCORE_DIR / f"{system}__{setname}.json"
    json.dump(recs, open(sp, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)


# ---------------- aggregate ----------------

def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def phase_aggregate(systems, sets):
    rows = []
    for system in systems:
        for setname in sets:
            sp = SCORE_DIR / f"{system}__{setname}.json"
            if not sp.exists():
                continue
            recs = json.load(open(sp, encoding="utf-8"))
            row = {"system": system, "set": setname, "n": len(recs)}
            row["refusal_rate"] = _mean([1.0 if r.get("refusal") else 0.0 for r in recs])
            if setname != "oos":
                row["bleu"] = _mean([r.get("bleu") for r in recs])
                row["rougeL"] = _mean([r.get("rougeL") for r in recs])
                row["grounding"] = _mean([r.get("grounding") for r in recs])
                row["has_citation"] = _mean(
                    [1.0 if r.get("citation", {}).get("has_citation") else 0.0 for r in recs])
                row["cited_valid"] = _mean(
                    [r.get("citation", {}).get("cited_valid_frac") for r in recs])
                row["cite_f1"] = _mean([r.get("citation", {}).get("cite_f1") for r in recs])
                judged = [r.get("judge", {}) for r in recs if isinstance(r.get("judge"), dict)]
                ok = [j for j in judged if not j.get("parse_failed")]
                row["faithfulness"] = _mean([j.get("faithfulness") for j in ok])
                row["completeness"] = _mean([j.get("completeness") for j in ok])
                row["usefulness"] = _mean([j.get("usefulness") for j in ok])
                row["harm_rate"] = _mean([j.get("harm") for j in ok])
                # Surfaced so a silently-failing judge can never masquerade as a
                # low score: judge_ok < 1.0 means some judgements were discarded.
                row["judge_ok"] = (len(ok) / len(judged)) if judged else None
                # judge_ok is a rate over *attempted* judgements, so it reads
                # 1.00 whether every item was judged or only a handful were.
                # n_judged is what tells the reader the sample size the judge
                # columns are actually means over, which matters because judging
                # is slow enough that a run may legitimately be stopped early.
                row["n_judged"] = len(ok)
            rows.append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    cols = ["system", "set", "n", "bleu", "rougeL", "grounding", "has_citation",
            "cited_valid", "cite_f1", "faithfulness", "completeness",
            "usefulness", "harm_rate", "refusal_rate", "judge_ok", "n_judged"]
    # CSV
    import csv
    with open(RESULTS / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    # Markdown
    def fmt(v):
        return "" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    (RESULTS / "comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {RESULTS/'comparison.csv'} and {RESULTS/'comparison.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["generate", "score", "aggregate", "all"],
                    default="all")
    ap.add_argument("--systems", nargs="*", default=DEFAULT_SYSTEMS,
                    help=f"default: {' '.join(DEFAULT_SYSTEMS)}; heavy systems "
                         "(qwen_general) must be named explicitly")
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge-model", default="qwen3.5:4b",
                    help="LLM judge, independent of all candidates; '' to disable")
    ap.add_argument("--dry-run", action="store_true",
                    help="print projected wall-clock and exit")
    args = ap.parse_args()

    for s in args.systems:
        assert s in SYSTEMS, f"unknown system {s}"
    for s in args.sets:
        assert s in SETS, f"unknown set {s}"

    if args.dry_run:
        estimate(args.systems, args.sets, args.limit)
        return

    if args.phase in ("generate", "all"):
        run_provenance(args.systems)
        estimate(args.systems, args.sets, args.limit)
        phase_generate(args.systems, args.sets, args.limit)
    if args.phase in ("score", "all"):
        judge = args.judge_model or None
        phase_score(args.systems, args.sets, judge_model=judge)
    if args.phase in ("aggregate", "all"):
        phase_aggregate(args.systems, args.sets)


if __name__ == "__main__":
    main()
