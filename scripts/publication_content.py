#!/usr/bin/env python3
"""
Single source of truth for the paper's text and tables.

Why this module exists
----------------------
The paper was previously written twice - once in generate_publication_docx.py and
once in generate_publication_latex.py - and both copies hardcoded a comparison
table whose numbers had never been measured (fine-tuned "0.42 / 0.51", ChatGPT
"0.38 / 0.45", human ratings "4.2"/"3.9"). No commercial-API evaluation and no
human rating exercise ever existed in this repository.

Both problems have the same fix: the prose lives here once, and every results
table is *loaded from the measurement artifacts* at build time. There is no code
path that lets a result be typed in by hand. If an artifact is missing, the table
renders as an explicit "not yet measured" note rather than a plausible number.

Artifacts consumed
------------------
  results/comparison.csv        <- scripts/evaluate.py --phase aggregate
  results/error_analysis.csv    <- scripts/error_analysis.py
  results/run_provenance.json   <- scripts/evaluate.py --phase generate
  results/scenario_verification.json <- scripts/verify_scenarios.py
  runs/ablation_results.json    <- notebooks/finetune_llama32_3b_full.ipynb (Colab)
  <trainer_state.json>          <- training run

Citation markers
----------------
Prose uses ``[[ref_key]]`` markers. Each generator renders them in its own
convention (LaTeX ``\\cite{key}``, DOCX ``[n]``), so the prose stays format-neutral.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EVAL = ROOT / "data" / "eval"

_EVAL_FILES = {"heldout": "heldout_test.json",
               "scenario": "scenario_test.json",
               "oos": "out_of_scope.json"}
_eval_cache: dict[str, list] = {}


def eval_set(setname):
    """The test items themselves, so counts in the prose come from the data."""
    if setname not in _eval_cache:
        path = EVAL / _EVAL_FILES[setname]
        _eval_cache[setname] = (json.load(open(path, encoding="utf-8"))
                                if path.exists() else [])
    return _eval_cache[setname]


def set_size(setname):
    """Size of a test set.

    Written out rather than hardcoded because the hold-out set shrank from 150
    to 146 when four questions were found to also occur in the training pool,
    and six sentences of prose went on claiming 150 until they were chased down
    by hand. Counting the file cannot drift.
    """
    return len(eval_set(setname))


def gold_count(setname):
    """Items carrying at least one gold section number."""
    return sum(1 for i in eval_set(setname) if i.get("gold_sections"))

TITLE = ("Fine-Tuning Large Language Models for Bangladesh Labour Law: "
         "An HR-Oriented Legal QA System")
AUTHORS = ["Md. Rakibul Haque", "Fabia Chowdhury", "Nowshin Sayara Tamanna"]

# Display names for the evaluated systems, in the order they should appear.
SYSTEM_LABELS = [
    ("base", "Base Llama 3.2 3B"),
    ("finetuned", "Fine-tuned Llama 3.2 3B"),
    ("rag_base", "RAG, base Llama 3.2 3B"),
    ("rag_finetuned", "RAG + fine-tuned"),
    ("qwen_general", "Qwen2.5 7B (larger general)"),
]
SYSTEM_ORDER = [s for s, _ in SYSTEM_LABELS]

# The hold-out topics come from build_test_split.py's keyword buckets, while the
# scenario topics were written by hand, so the same concept arrived under two
# names ("dismissal" vs "termination"). Reporting them separately would split one
# topic across two rows and understate coverage of both.
TOPIC_ALIASES = {
    "termination": "dismissal",
    "maternity_benefit": "maternity",
    "child_labour": "child_labour",
}


def canon_topic(t):
    t = (t or "other").strip().lower()
    return TOPIC_ALIASES.get(t, t)


# --------------------------------------------------------------------------
# artifact loaders
# --------------------------------------------------------------------------

def _read_csv(path):
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(row, key, digits=2):
    v = (row or {}).get(key)
    if v in (None, ""):
        return None
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return None


def load_comparison():
    """Rows of results/comparison.csv keyed by (system, set)."""
    rows = _read_csv(RESULTS / "comparison.csv")
    if not rows:
        return None
    return {(r["system"], r["set"]): r for r in rows}


def load_provenance():
    p = RESULTS / "run_provenance.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def load_error_analysis():
    return _read_csv(RESULTS / "error_analysis.csv")


def load_scenario_verification():
    p = RESULTS / "scenario_verification.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def load_run_manifest():
    """Seeds, hardware and library versions captured by the training notebook."""
    p = ROOT / "runs" / "run_manifest.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def load_ablation():
    for p in (ROOT / "runs" / "ablation_results.json",
              RESULTS / "ablation_results.json"):
        if p.exists():
            return json.load(open(p, encoding="utf-8"))
    return None


def load_training_metrics(trainer_state_path=None):
    """Summary metrics from a HF trainer_state.json, if one is available."""
    candidates = [trainer_state_path] if trainer_state_path else []
    candidates += [ROOT / "runs" / "main_full" / "trainer_state.json",
                   ROOT / "trainer_state.json"]
    for p in candidates:
        if p and Path(p).exists():
            state = json.load(open(p, encoding="utf-8"))
            log = state.get("log_history", [])
            steps = [(e.get("step"), e.get("loss"), e.get("eval_loss"))
                     for e in log if e.get("loss") or e.get("eval_loss")]
            return {"state": state, "log": log, "steps": steps,
                    "best_metric": state.get("best_metric"),
                    "global_step": state.get("global_step"),
                    "source": str(p)}
    return None


# --------------------------------------------------------------------------
# table builders -> (caption, header, rows, note)
# --------------------------------------------------------------------------

NOT_MEASURED = ("Not yet measured. Run the evaluation pipeline "
                "(scripts/evaluate.py) to populate this table.")


def _fmt(v):
    return "--" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def table_main_comparison(setname="heldout"):
    """Primary quality comparison on an in-scope test set."""
    comp = load_comparison()
    header = ["System", "BLEU", "ROUGE-L", "Grounding", "Cited valid",
              "Faithful.", "Complete.", "Useful.", "Harm"]
    caption = {
        "heldout": (f"Comparison on the {set_size('heldout')}-item leakage-free "
                    "hold-out set. "
                    "BLEU and ROUGE-L are 0--100; grounding and cited-valid are "
                    "fractions; faithfulness, completeness and usefulness are "
                    "1--5 LLM-judge scores; harm is the rate of answers flagged "
                    "misleading."),
        "scenario": (f"Comparison on the {set_size('scenario')} hand-authored HR "
                     "scenarios (leave, overtime, termination, wages, maternity, "
                     "misconduct, probation, safety, child labour)."),
    }[setname]
    if not comp:
        return caption, header, [], NOT_MEASURED
    rows = []
    judged = []
    for key, label in SYSTEM_LABELS:
        r = comp.get((key, setname))
        if not r:
            continue
        rows.append([label,
                     _fmt(_num(r, "bleu")), _fmt(_num(r, "rougeL")),
                     _fmt(_num(r, "grounding")), _fmt(_num(r, "cited_valid")),
                     _fmt(_num(r, "faithfulness")), _fmt(_num(r, "completeness")),
                     _fmt(_num(r, "usefulness")), _fmt(_num(r, "harm_rate"))])
        nj, n = _num(r, "n_judged"), _num(r, "n")
        if nj is not None and n:
            judged.append((int(nj), int(n)))

    # The overlap and citation columns cover every item, but judging is slow
    # enough that a run can legitimately be stopped early. Saying so in the
    # caption is the difference between a subsample and a misreported full one.
    if judged:
        lo, n_items = min(j for j, _ in judged), judged[0][1]
        if lo == 0:
            caption += (" The LLM-judge columns are empty: judging had not been "
                        "run at the time this table was built.")
        elif lo < n_items:
            caption += (f" The judge columns are means over the first {lo} of "
                        f"{n_items} items per system, judged in a seeded random "
                        "order that is identical across systems, so they remain "
                        "paired and unbiased; the remaining columns cover all "
                        f"{n_items}.")
    return caption, header, rows, (None if rows else NOT_MEASURED)


def table_citation(setname="scenario"):
    """Citation accuracy, reported where gold sections are complete."""
    comp = load_comparison()
    header = ["System", "Answers citing a section", "Cited sections valid",
              "Citation F1 vs.\u00a0gold"]
    caption = ("Citation behaviour on the scenario set, the subset for which "
               f"gold section numbers are complete ({gold_count('scenario')}/"
               f"{set_size('scenario')} items; only {gold_count('heldout')}/"
               f"{set_size('heldout')} hold-out items carry gold sections, so "
               "citation F1 is not reported there). ``Valid'' means the cited "
               "section exists in the Act; F1 is against the gold sections for "
               "the question.")
    if not comp:
        return caption, header, [], NOT_MEASURED
    rows = []
    for key, label in SYSTEM_LABELS:
        r = comp.get((key, setname))
        if not r:
            continue
        rows.append([label, _fmt(_num(r, "has_citation")),
                     _fmt(_num(r, "cited_valid")), _fmt(_num(r, "cite_f1"))])
    return caption, header, rows, (None if rows else NOT_MEASURED)


def table_refusal():
    """Out-of-scope refusal - the safety-relevant behaviour."""
    comp = load_comparison()
    header = ["System", "Refusal rate, out of scope (higher is better)",
              "Refusal rate, in scope (lower is better)"]
    caption = ("Scope discipline. The out-of-scope probe set contains 20 "
               "questions from other domains (income tax, VAT, criminal law, "
               "immigration, foreign labour law, medicine, programming, general "
               "knowledge); the desired behaviour is to decline. The in-scope "
               "column is the over-refusal rate on the hold-out set, where "
               "declining is a failure.")
    if not comp:
        return caption, header, [], NOT_MEASURED
    rows = []
    for key, label in SYSTEM_LABELS:
        oos, held = comp.get((key, "oos")), comp.get((key, "heldout"))
        if not oos and not held:
            continue
        rows.append([label, _fmt(_num(oos, "refusal_rate")),
                     _fmt(_num(held, "refusal_rate"))])
    return caption, header, rows, (None if rows else NOT_MEASURED)


ERROR_COLS = [
    ("hallucinated_section", "Hallucinated section"),
    ("wrong_section", "Wrong section"),
    ("missing_citation", "No citation"),
    ("unfaithful", "Unfaithful"),
    ("incomplete", "Incomplete"),
    ("harmful", "Harmful"),
]


def table_error_analysis(setname="heldout"):
    rows_in = load_error_analysis()
    header = ["System", "n"] + [lab for _, lab in ERROR_COLS]
    caption = ("Failure modes on the hold-out set (counts). Categories are not "
               "mutually exclusive: one answer can both cite a section that does "
               "not exist in the Act and omit a required condition.")
    if not rows_in:
        return caption, header, [], ("Not yet measured. Run "
                                     "scripts/error_analysis.py to populate this table.")
    by_sys = {r["system"]: r for r in rows_in if r["set"] == setname}
    rows = []
    for key, label in SYSTEM_LABELS:
        r = by_sys.get(key)
        if not r:
            continue
        rows.append([label, r.get("n", "--")] + [r.get(c, "0") for c, _ in ERROR_COLS])
    return caption, header, rows, (None if rows else NOT_MEASURED)


def load_refusal_validation():
    p = RESULTS / "refusal_detector_validation.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def refusal_validation_sentence():
    """State how well the refusal heuristic agrees with hand labels."""
    v = load_refusal_validation()
    if not v:
        return ("The detector's agreement with hand labels is reported by "
                "scripts/check_refusal_detector.py.")
    return (f"Because this is the safety-relevant metric and it is computed by a "
            f"heuristic, we validated it against {v['n']} out-of-scope answers "
            f"labelled by hand: accuracy {v['accuracy']:.2f}, precision "
            f"{v['precision']:.2f}, recall {v['recall']:.2f}. The residual "
            f"disagreements are answers that decline a request and then partly "
            f"comply with it, which have no unambiguous label.")


def load_significance(setname="heldout"):
    p = RESULTS / f"significance_{setname}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


# Metrics worth reporting a significance test for. BLEU is omitted: it is
# reported for comparability, not as evidence. Citation F1 and refusal rate are
# the two that carry the paper's argument, so they lead.
SIG_METRICS = ["cite_f1", "refusal", "rougeL", "grounding", "cited_valid",
               "faithfulness", "completeness", "usefulness"]


def table_significance(setname="heldout"):
    """Paired differences against the base model, with CIs and adjusted p."""
    header = ["Metric", "System", "Difference", "95% CI", "p (Holm)", "Sig."]
    sig = load_significance(setname)
    where = {"heldout": f"the {set_size('heldout')}-item hold-out set",
             "scenario": f"the {set_size('scenario')} hand-authored HR scenarios"
             }.get(setname, f"the {setname} set")
    caption = (f"Paired comparison against the base model on {where}. "
               "Every system answers the same questions, so differences are "
               "computed per item; the CI is a paired bootstrap over items "
               "(10{,}000 resamples) and p is Wilcoxon signed-rank, "
               "Holm-corrected across the systems compared within each metric. "
               "A difference is marked significant only when the adjusted p is "
               "below 0.05 and the interval excludes zero.")
    if not sig:
        return caption, header, [], ("Not yet computed. Run "
                                     "scripts/significance.py after scoring.")
    labels = dict(SYSTEM_LABELS)
    order = {m: i for i, m in enumerate(SIG_METRICS)}
    keep = [c for c in sig["comparisons"] if c["metric"] in order]
    keep.sort(key=lambda c: order[c["metric"]])
    rows = []
    for c in keep:
        ci = ("--" if c.get("ci_low") is None
              else f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]")
        rows.append([c["label"], labels.get(c["system"], c["system"]),
                     f"{c['mean_diff']:+.2f}", ci,
                     f"{c.get('p_holm', 1.0):.3f}",
                     "yes" if c.get("significant") else ""])
    return caption, header, rows, (None if rows else NOT_MEASURED)


# Table labels are noun phrases and several contain a comma ("RAG, base Llama
# 3.2 3B"), which reads badly mid-sentence and breaks comma-separated lists.
# These are the same systems named for running prose.
_INLINE_NAMES = {
    "base": "unmodified base",
    "finetuned": "fine-tuned",
    "rag_base": "retrieval-only",
    "rag_finetuned": "retrieval plus fine-tuning",
    "qwen_general": "larger general-purpose",
}


def _inline_name(system):
    return _INLINE_NAMES.get(system, dict(SYSTEM_LABELS).get(system, system))


def _sig_lookup(setname):
    """(metric, system) -> comparison record, for the findings prose."""
    sig = load_significance(setname)
    if not sig:
        return {}
    return {(c["metric"], c["system"]): c for c in sig["comparisons"]}


def _sig_phrase(c):
    """Render one comparison as '+0.30, p = 0.002' with a significance verdict."""
    if not c:
        return None
    p = c.get("p_holm", 1.0)
    ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    return f"{c['mean_diff']:+.2f} ({ptxt})"


def training_findings_text():
    """What the training run and the ablation show, read from the artifacts."""
    tm = load_training_metrics()
    abl = load_ablation()
    if not tm and not abl:
        return ""
    out = []

    if tm:
        state = tm["state"]
        ck = str(state.get("best_model_checkpoint") or "")
        tail = ck.rsplit("-", 1)
        best_step = tail[1] if len(tail) == 2 and tail[1].isdigit() else None
        ran, budget = state.get("global_step"), state.get("max_steps")
        if best_step and ran and budget and ran < budget:
            out.append(
                f"Training does not benefit from the full three-epoch budget. "
                f"Validation loss falls until step {best_step} and then rises "
                f"for three consecutive evaluations while training loss keeps "
                f"falling, which is overfitting rather than noise; early "
                f"stopping halted the run at step {ran} of a {budget}-step "
                f"budget, at {state.get('epoch', 0):.2f} epochs, and the "
                f"checkpoint from step {best_step} was restored. The evaluation "
                f"in this paper measures that checkpoint. A fixed step budget "
                f"without validation monitoring, as used in the pilot run, "
                f"would have reported a strictly worse model.")

    if abl:
        by = {r.get("name"): r for r in abl}

        def loss(n):
            return (by.get(n) or {}).get("eval_loss")

        d25, d50, d100 = loss("data_25pct"), loss("data_50pct"), loss("data_100pct")
        if None not in (d25, d50, d100):
            out.append(
                f"Dataset size still pays. Validation loss improves "
                f"monotonically from {d25:.3f} at a quarter of the pool "
                f"({by['data_25pct']['n_train']} examples) to {d50:.3f} at half "
                f"and {d100:.3f} at the full pool "
                f"({by['data_100pct']['n_train']} examples), with no sign of a "
                f"plateau. The curve is still descending at the largest size "
                f"available, so the dataset is a bottleneck before the method "
                f"is: more verified QA pairs would likely help more than "
                f"further tuning of this recipe.")

        r8, r16, r32 = loss("lora_r8"), loss("lora_r16"), loss("lora_r32")
        if None not in (r8, r16, r32, d25, d100):
            out.append(
                f"LoRA rank matters less. Going from rank 8 to 16 to 32 moves "
                f"validation loss {r8:.3f} to {r16:.3f} to {r32:.3f} -- a total "
                f"span of {abs(r8 - r32):.3f}, against {abs(d25 - d100):.3f} "
                f"spanned by dataset size. Rank 32 is the best of the three, "
                f"but the gain over the rank 16 used throughout this paper is "
                f"small relative to its extra parameters, and an ordering in "
                f"validation loss does not establish that the difference would "
                f"survive on downstream answer quality, which is what the "
                f"comparison in the results section measures.")

        nr, wr = loss("no_refusal_data"), loss("with_refusal_data")
        if None not in (nr, wr):
            out.append(
                f"The refusal examples are cheap. Removing all "
                f"{by['with_refusal_data']['n_train'] - by['no_refusal_data']['n_train']} "
                f"of them changes validation loss from {wr:.3f} to {nr:.3f}. "
                f"This comparison should not be over-read in either direction: "
                f"the two arms differ in training-set size as well as in "
                f"content, and validation loss is computed over a split that "
                f"contains refusal items, so an arm never trained on them is "
                f"expected to score worse on exactly those items. What the "
                f"refusal examples are for is out-of-scope behaviour, and that "
                f"is measured directly on the out-of-scope set rather than "
                f"here.")

    return "\n\n".join(p for p in out if p)


def key_findings_text():
    """
    State what the evaluation actually found, derived from the artifacts.

    This exists so that the paper's argument and its tables cannot disagree: the
    Discussion's claims are recomputed from results/ at build time rather than
    written by hand. If a rerun changes the sign or the significance of a
    comparison, this prose changes with it.
    """
    comp = load_comparison()
    if not comp:
        return ("The findings are stated once the evaluation pipeline has been "
                "run; see scripts/evaluate.py.")

    def val(system, setname, key):
        r = comp.get((system, setname))
        if not r or r.get(key) in (None, ""):
            return None
        try:
            return float(r[key])
        except (TypeError, ValueError):
            return None

    out = []
    scn = _sig_lookup("scenario")

    # --- Finding 1: what fine-tuning does and does not transfer -------------
    ft_scn = scn.get(("cite_f1", "finetuned"))
    rag_scn = scn.get(("cite_f1", "rag_base"))
    ragft_scn = scn.get(("cite_f1", "rag_finetuned"))
    if ft_scn and rag_scn:
        ft_sig = ft_scn.get("significant")
        rag_sig = rag_scn.get("significant")
        if not ft_sig and rag_sig:
            para = (
                "The central result is a dissociation between fluency and "
                "citation accuracy. Fine-tuning clearly changes how the model "
                "writes: on the hold-out set it improves surface agreement with "
                "the reference answers and reduces unnecessary refusals. It does "
                "not, however, reliably improve the one thing an HR user must be "
                "able to check. On the hand-authored scenarios, the fine-tuned "
                f"model's citation F1 differs from the base model's by "
                f"{_sig_phrase(ft_scn)}, which is not significant after "
                "correction, whereas retrieval over the same statute improves it "
                f"by {_sig_phrase(rag_scn)}")
            if ragft_scn:
                para += (f" and retrieval combined with fine-tuning by "
                         f"{_sig_phrase(ragft_scn)}")
            para += (". On this benchmark, supplying the statute at inference "
                     "time is what makes the citation correct; training on "
                     "statute-derived question-answer pairs mainly makes the "
                     "answer sound correct.")

            # The citation metric and the judge are independent measurements -
            # the judge never sees the gold sections - so noting where they
            # agree is what distinguishes a finding from a metric artefact.
            dims = [("faithfulness", "faithfulness"),
                    ("completeness", "completeness"),
                    ("usefulness", "usefulness")]
            ft_ns = [n for m, n in dims
                     if (c := scn.get((m, "finetuned"))) and not c.get("significant")]
            rag_s = [n for m, n in dims
                     if (c := scn.get((m, "rag_base"))) and c.get("significant")]
            if len(ft_ns) == len(dims) and len(rag_s) == len(dims):
                para += (
                    " The LLM judge, which never sees the gold section labels "
                    "and so measures something the citation metric does not, "
                    "agrees: on the scenarios the fine-tuned model is not "
                    "significantly better than the base model on any judged "
                    f"dimension ({', '.join(ft_ns)}), while retrieval is "
                    "significantly better on all of them. Two independent "
                    "measurements pointing the same way is the reason we read "
                    "this as a property of the systems rather than of one "
                    "brittle metric.")
            out.append(para)

    # --- Finding 2: the hold-out set hides this ----------------------------
    ho_ft = val("finetuned", "heldout", "cite_f1")
    sc_ft = val("finetuned", "scenario", "cite_f1")
    ho_b = val("base", "heldout", "cite_f1")
    sc_b = val("base", "scenario", "cite_f1")
    if None not in (ho_ft, sc_ft, ho_b, sc_b) and ho_ft > sc_ft:
        out.append(
            "This gap is invisible if the hold-out set is the only test. "
            f"Citation F1 for the fine-tuned model is {ho_ft:.2f} on the "
            f"hold-out set but {sc_ft:.2f} on the scenarios, and for the base "
            f"model {ho_b:.2f} against {sc_b:.2f}. Hold-out questions are "
            "generated from the statute and tend to name or paraphrase the "
            "provision they are about, so the citation is partly given away by "
            "the question. A practitioner's question describes a situation "
            "instead, and the model must find the provision itself. Any "
            "evaluation of a statute-tuned model that reports only in-"
            "distribution hold-out accuracy will therefore overstate how well "
            "it cites.")

    # --- Finding 3: scope discipline degrades ------------------------------
    ref = {s: val(s, "oos", "refusal_rate") for s, _ in SYSTEM_LABELS}
    ref = {s: v for s, v in ref.items() if v is not None}
    if len(ref) >= 3 and "base" in ref:
        labels = dict(SYSTEM_LABELS)
        worst = min(ref, key=ref.get)
        if ref["base"] > ref[worst]:
            listed = "; ".join(
                f"{labels[s]} {ref[s]:.2f}"
                for s in SYSTEM_ORDER if s in ref)
            out.append(
                "The third finding is a safety trade-off that runs against the "
                "other two. Every modification we make to the base model leaves "
                "it less willing to decline a question it should not answer. "
                f"Out-of-scope refusal rates are: {listed}. The untouched base "
                f"model refuses most often ({ref['base']:.2f}); the "
                f"{_inline_name(worst)} configuration refuses least often "
                f"({ref[worst]:.2f}). "
                "Retrieval is the larger contributor to this erosion, which is "
                "intuitive in hindsight - the retriever always returns its "
                "nearest passages, and a confidently retrieved chunk of the Act "
                "makes an unrelated question look answerable. The configurations "
                "that cite best are therefore among the least willing to say "
                "that a question is not about labour law, so a deployment would "
                "need an explicit scope check ahead of the model rather than "
                "relying on the model's own restraint.")

            # Retrieval cuts harmful answers on in-scope questions while making
            # out-of-scope refusal worse. Reporting either half alone would
            # misrepresent it, so the two are stated together or not at all.
            harm = scn.get(("harm", "rag_base"))
            if harm and harm.get("significant") and harm["mean_diff"] < 0:
                out.append(
                    "These two safety results point in opposite directions and "
                    "should be read together. On in-scope scenarios retrieval "
                    "makes the system safer: the judge flags fewer misleading "
                    f"answers than for the base model, {_sig_phrase(harm)}. On "
                    "out-of-scope probes the same configuration is the least "
                    "likely to decline. Retrieval helps when the question is "
                    "within the statute and hurts when it is not, because "
                    "nothing in the pipeline distinguishes the two cases - the "
                    "retriever returns its nearest passages either way. Quoting "
                    "only the first half would make retrieval look unambiguously "
                    "safer than it is, and only the second would make it look "
                    "unusable; the honest summary is that retrieval improves "
                    "answer quality inside the scope it was built for and "
                    "provides no signal at all about where that scope ends.")
    return "\n\n".join(out) if out else (
        "The measured comparisons are reported in the tables above.")


# Ablation arms, grouped by the axis each one varies, with the label to print.
# The notebook reuses the main run as the shared reference point of all three
# axes, so ablation_results.json lists it three times under three names. Printed
# verbatim that would show eight rows for six actual training runs and imply
# more evidence than exists; the duplicates are collapsed and the shared row is
# marked instead.
ABLATION_GROUPS = [
    ("Training-set size", [("data_25pct", "25% of pool"),
                           ("data_50pct", "50% of pool"),
                           ("data_100pct", "100% of pool")]),
    ("LoRA rank", [("lora_r8", "rank 8"),
                   ("lora_r16", "rank 16"),
                   ("lora_r32", "rank 32")]),
    ("Refusal examples", [("no_refusal_data", "without"),
                          ("with_refusal_data", "with")]),
]


def _best_step(rec):
    """Step of the best checkpoint, which is not the step training ran to.

    With load_best_model_at_end the trainer re-evaluates the restored model and
    logs it under the *final* step, so taking the minimum eval_loss over the log
    reports where training halted rather than where the model peaked. The main
    run peaked at 700 and halted at 850; only the checkpoint path distinguishes
    them.
    """
    ck = rec.get("best_checkpoint_step") or ""
    tail = str(ck).rsplit("-", 1)
    return tail[1] if len(tail) == 2 and tail[1].isdigit() else None


def table_ablation():
    abl = load_ablation()
    header = ["Axis", "Configuration", "Train ex.", "Rank", "Best step",
              "Ran to", "Eval loss", "Perplexity"]
    caption = ("Ablation over training-set size, LoRA rank, and the inclusion of "
               "the refusal examples. All arms share seed 3407 and the same "
               "schedule, and all use early stopping on validation loss "
               "(patience 3, evaluated every 50 steps) with the best checkpoint "
               "restored. \\emph{Best step} is where validation loss bottomed "
               "out and \\emph{ran to} is where training halted; the gap is the "
               "patience window. Rows marked ``same run'' are not a separate "
               "training run: one reference run serves as the 100\\%, rank-16 "
               "and with-refusals point on all three axes, so the table reports "
               "six training runs rather than eight.")
    note = ("Not yet run. Execute notebooks/finetune_llama32_3b_full.ipynb on a "
            "GPU runtime and place the resulting runs/ablation_results.json in "
            "the project root to populate this table.")
    if not abl:
        return caption, header, [], note

    by_name = {r.get("name"): r for r in abl}
    # Arms sharing a checkpoint are literally the same run, not a replication.
    seen_ckpt = {}
    rows = []
    for axis, arms in ABLATION_GROUPS:
        first = True
        for name, label in arms:
            r = by_name.get(name)
            if not r:
                continue
            ck = r.get("best_checkpoint_step")
            shared = ck in seen_ckpt
            seen_ckpt.setdefault(ck, name)
            rows.append([
                axis if first else "",
                label + (" (same run)" if shared else ""),
                str(r.get("n_train")), str(r.get("lora_r")),
                _best_step(r) or "--", str(r.get("global_step")),
                f"{r['eval_loss']:.4f}" if r.get("eval_loss") else "--",
                f"{r['perplexity']:.2f}" if r.get("perplexity") else "--"])
            first = False
    return caption, header, rows, None


# Measured on the original Colab run (Llama 3.2 3B + LoRA, 100 steps, Tesla T4).
# These are real observations from that run, but it was a *pilot*: a fixed
# 100-step budget with no early stopping. They are labelled as such wherever they
# appear, and are superseded automatically once runs/main_full/trainer_state.json
# from the full-training notebook is present.
PILOT_TRAINING = {
    "Final training loss": "1.3637",
    "Final eval loss": "0.9522",
    "Final eval perplexity": "2.59",
    "Steps completed": "100 (fixed budget, no early stopping)",
    "Total training time (s)": "465.9",
    "GPU": "Tesla T4 (peak 5.33 GB / 14.74 GB)",
}
PILOT_STEPS = [(25, 1.8568, 1.3950), (50, 1.2597, 1.1217),
               (75, 1.0083, 0.9772), (100, 0.8798, 0.9522)]


def table_training(trainer_state_path=None):
    """Training summary: the full run if available, otherwise the pilot."""
    header = ["Metric", "Value"]
    tm = load_training_metrics(trainer_state_path)
    if tm:
        import math
        state, log = tm["state"], tm["log"]
        tl = [e["loss"] for e in log if "loss" in e]
        rows = []
        if tl:
            rows.append(["Final training loss", f"{tl[-1]:.4f}"])
        best = state.get("best_metric")
        if best is not None:
            # The best checkpoint is what was kept and what was exported, so it
            # is the model the evaluation section measures. Reporting the loss
            # at the last step instead would describe a model that was thrown
            # away - and here it would flatter nothing, since the last step is
            # worse than the best.
            rows.append(["Best validation loss", f"{best:.4f}"])
            rows.append(["Perplexity at best", f"{math.exp(best):.2f}"])
        ck = str(state.get("best_model_checkpoint") or "")
        tail = ck.rsplit("-", 1)
        if len(tail) == 2 and tail[1].isdigit():
            rows.append(["Best checkpoint step", tail[1]])
        ran, budget = state.get("global_step"), state.get("max_steps")
        if ran and budget:
            rows.append(["Steps run / budget", f"{ran} / {budget}"])
            if ran < budget:
                rows.append(["Stopped early", f"yes, at {state.get('epoch', 0):.2f} epochs"])
        man = load_run_manifest()
        if man:
            rows.append(["Seed", str(man.get("seed", "--"))])
            gpu = man.get("gpu_name")
            if gpu:
                rows.append(["GPU", f"{gpu} ({man.get('gpu_total_gb', '--')} GB)"])
            pk = man.get("packages", {})
            if pk:
                rows.append(["Key versions",
                             ", ".join(f"{k} {pk[k]}" for k in
                                       ("torch", "transformers", "peft", "trl")
                                       if k in pk)])
        caption = ("Training metrics for the full run. Early stopping monitored "
                   "validation loss (patience 3, evaluated every 50 steps) and "
                   "the best checkpoint was restored at the end, so the reported "
                   "model is the one at the best step rather than the last.")
        return caption, header, rows, None
    caption = ("Training metrics for the pilot run. This was a fixed 100-step "
               "budget without early stopping, not a converged full-training "
               "run; the full run and ablation are specified in the "
               "accompanying notebook and are reported as pending.")
    return caption, header, [[k, v] for k, v in PILOT_TRAINING.items()], None


def table_training_steps(trainer_state_path=None):
    header = ["Step", "Training loss", "Validation loss"]
    caption = "Loss at each evaluation step."
    tm = load_training_metrics(trainer_state_path)
    if tm:
        log = tm["log"]
        last_train, rows, seen = None, [], set()
        for e in log:
            if "loss" in e:
                last_train = e["loss"]
            if "eval_loss" in e:
                # load_best_model_at_end re-evaluates the restored checkpoint
                # and logs it under the final step, so the last step appears
                # twice with different losses. Keeping both would show
                # validation loss dropping at the end of a run that was stopped
                # for getting worse.
                if e.get("step") in seen:
                    continue
                seen.add(e.get("step"))
                rows.append([str(e.get("step")),
                             f"{last_train:.4f}" if last_train else "--",
                             f"{e['eval_loss']:.4f}"])
        if rows:
            return caption, header, rows, None
    caption += " (Pilot run.)"
    return caption, header, [[str(s), f"{a:.4f}", f"{b:.4f}"]
                             for s, a, b in PILOT_STEPS], None


def table_scenario_verification():
    """Outcome of validating the scenario gold standard against the Act."""
    header = ["Check", "Result"]
    caption = ("Validation of the hand-authored scenario gold standard against "
               "the text of the Act.")
    v = load_scenario_verification()
    if not v:
        return caption, header, [], "Not yet run. Execute scripts/verify_scenarios.py."
    s = v["summary"]
    rows = [
        ["Scenarios checked", str(s["n_items"])],
        ["Gold sections resolving to a real section of the Act",
         f"{s['n_items'] - s['n_missing_section']}/{s['n_items']}"],
        ["Numeric entitlements traceable to the cited section",
         f"{s['n_items'] - s['n_unsupported_numbers']}/{s['n_items']}"],
        ["Mean lexical overlap with the cited section",
         f"{s['mean_lexical_support']:.3f}"],
        ["Verified automatically", f"{s['n_auto_verified']}/{s['n_items']}"],
        ["Accepted after manual review", str(s.get("n_manually_reviewed", 0))],
        ["Verified in total",
         f"{s.get('n_verified', s['n_auto_verified'])}/{s['n_items']}"],
    ]
    return caption, header, rows, None


def example_qa_pairs(n=3):
    """A few verified scenario items, to show what the data actually looks like."""
    p = ROOT / "data/eval/scenario_test.json"
    if not p.exists():
        return []
    items = json.load(open(p, encoding="utf-8"))
    picked, seen = [], set()
    for it in items:
        if it.get("topic") in seen or not it.get("gold_sections"):
            continue
        seen.add(it.get("topic"))
        picked.append({"topic": it.get("topic"), "question": it["question"],
                       "reference": it["reference"],
                       "gold_sections": it["gold_sections"]})
        if len(picked) >= n:
            break
    return picked


def reproducibility_note():
    """Concrete run details, read from the provenance record."""
    p = load_provenance()
    if not p:
        return ("Decoding parameters, model digests and library versions are "
                "recorded to results/run_provenance.json when the evaluation "
                "is run.")
    models = p.get("models", {})
    names = ", ".join(sorted({m.get("model") for m in models.values() if m.get("model")}))
    return (f"All systems were decoded greedily (temperature "
            f"{p.get('temperature', 0)}) with seed {p.get('seed')} and a "
            f"{p.get('num_predict')}-token limit, under an identical system "
            f"prompt. Serving used Ollama {p.get('ollama_version', '')} on "
            f"{p.get('platform', 'the reported platform')} with Python "
            f"{p.get('python')}. Model identifiers: {names}. Exact model "
            f"digests, the full system prompt and the generation timestamp are "
            f"recorded in results/run_provenance.json, and the evaluation is "
            f"resumable and idempotent so a re-run reproduces these numbers.")


def table_dataset():
    """Dataset composition - counted from the actual files."""
    header = ["Split", "Items", "Purpose"]
    caption = "Dataset composition after leakage-free splitting."

    def count(p):
        f = ROOT / p
        try:
            return str(len(json.load(open(f, encoding="utf-8"))))
        except Exception:
            return "--"

    rows = [
        ["Training pool", count("data/final/train_final.json"),
         "LoRA fine-tuning (includes refusal examples)"],
        ["  of which refusal examples", count("data/final/refusal_examples.json"),
         "Teach declining out-of-scope questions"],
    ]

    # The trainer holds back a validation split from the pool, so the number of
    # examples the model actually sees is smaller than the pool. Without these
    # two rows the ablation table's "2964" contradicts the "3119" here for no
    # visible reason.
    pool = count("data/final/train_final.json")
    abl = load_ablation() or []
    main = next((r for r in abl if r.get("name") == "data_100pct"), None)
    if main and pool.isdigit() and main.get("n_train"):
        rows += [
            ["  of which seen in training", str(main["n_train"]),
             "After the trainer's validation hold-out"],
            ["  of which validation split", str(int(pool) - main["n_train"]),
             "Early stopping and model selection"],
        ]

    rows += [
        ["Hold-out test", count("data/eval/heldout_test.json"),
         "Leakage-free, topic-stratified evaluation"],
        ["HR scenario test", count("data/eval/scenario_test.json"),
         "Hand-authored realistic HR situations"],
        ["Out-of-scope probes", count("data/eval/out_of_scope.json"),
         "Refusal behaviour"],
    ]
    return caption, header, rows, None


# --------------------------------------------------------------------------
# prose
# --------------------------------------------------------------------------

ABSTRACT = """
Human resource professionals in Bangladesh need reliable, citable answers to
questions about the Bangladesh Labour Act 2006 (amended to 2018)[[ref_act]].
General-purpose large language models lack reliable jurisdiction-specific
knowledge and are known to hallucinate legal content[[ref_hallucination]][[ref_legaltools]],
while sending employee data to hosted APIs raises confidentiality concerns. We
present an end-to-end, reproducible pipeline that converts the official PDF of
the Act into a validated question-answering dataset, fine-tunes Llama 3.2 3B
Instruct[[ref_llama3]] with low-rank adaptation[[ref_lora]] using Unsloth[[ref_unsloth]],
and deploys the result locally in GGUF (Q4\\_K\\_M) form via Ollama. To evaluate
it we construct a leakage-free benchmark: a topic-stratified hold-out set carved
out of the dataset before training, a set of hand-authored HR scenarios covering
leave, overtime, termination, wages, maternity benefit, misconduct, probation and
workplace safety, and a set of out-of-scope probes that the system should
decline. Beyond BLEU[[ref_bleu]] and ROUGE-L[[ref_rouge]], which reward surface
overlap rather than legal correctness, we report citation validity and citation
F1 against gold sections, lexical grounding in the statute, and LLM-judged
faithfulness, completeness, usefulness and harmfulness, together with an error
analysis of the failure modes. We compare the fine-tuned model against the base
model, retrieval-augmented generation[[ref_rag]] over the Act, and their
combination. On the practitioner scenarios, fine-tuning alone does not
significantly improve citation accuracy or any judged quality dimension over the
base model, while retrieval improves all of them; the advantage of fine-tuning
that is visible on the statute-derived hold-out set disappears on scenarios,
because hold-out questions tend to name the provision they concern and so give
the citation away. Retrieval carries a cost the same benchmark makes visible:
every configuration that cites better also declines fewer out-of-scope
questions. These results align with the broader finding that retrieval
outperforms fine-tuning for knowledge injection, and extend it to a
citation-level criterion and to the scope discipline that a deployed statutory
assistant requires. The system is positioned as statute-grounded informational
support for HR practice, not legal advice.
"""

INTRO = """
Bangladesh's labour regime is governed principally by the Bangladesh Labour Act
2006, as amended to 2018[[ref_act]]. HR practitioners consult it constantly -
when calculating annual leave, deciding whether a termination follows the correct
procedure, or determining maternity benefit entitlement - yet the Act is long,
cross-referenced, and not written for quick lookup. General-purpose assistants
answer such questions fluently but unreliably: studies of legal hallucination
report high rates of fabricated authority in exactly this setting[[ref_hallucination]],
and even purpose-built commercial legal research tools hallucinate
materially[[ref_legaltools]]. A wrong but confident answer about a statutory
notice period is worse than no answer, because it is actionable.

Two further constraints shape the problem. First, HR questions carry employee
data, so routing them to a hosted API is often unacceptable on confidentiality
grounds. Second, Bangladesh-specific labour law is poorly represented in
general-purpose models, and the broader ecosystem of Bangla and
Bangladesh-focused legal NLP resources remains thin[[ref_banglabert]][[ref_ukil]].
These push toward a small, locally deployable model specialised to a single
statute.

This paper makes three contributions. (1) A reproducible pipeline from the Act's
official PDF to a validated QA dataset, with the validation criteria and their
pass rates reported rather than asserted. (2) A leakage-free evaluation benchmark
for Bangladesh labour law consisting of a topic-stratified hold-out set,
hand-authored HR scenarios with gold section labels, and out-of-scope probes -
together with a metric suite appropriate to legal QA rather than to translation.
(3) A comparison of domain fine-tuning against retrieval-augmented
generation[[ref_rag]] over the same statute and against their combination,
scored on whether the answer cites the correct provision rather than on factual
agreement alone, with an error analysis that characterises how each
configuration fails.

We are deliberately careful about what is claimed. The system provides
statute-grounded informational support; it is not legal advice, and the
evaluation measures agreement with the Act and with reference answers, not
outcomes in practice.
"""

RELATED = """
Domain-specific legal language models are well established. LEGAL-BERT[[ref_legalbert]]
showed that in-domain pretraining helps legal tasks, and benchmarks such as
LexGLUE[[ref_lexglue]], CUAD[[ref_cuad]], JEC-QA[[ref_jecqa]] and
LegalBench[[ref_legalbench]] have made legal capability measurable. More
recently, instruction-tuned legal models have appeared for several
jurisdictions: SaulLM-7B[[ref_saullm]] for English-language law,
ChatLaw[[ref_chatlaw]], Lawyer LLaMA[[ref_lawyerllama]] and
DISC-LawLLM[[ref_disclaw]] for Chinese law, and ALKAFI-LLaMA3[[ref_alkafi]] for
Palestinian law. ALKAFI-LLaMA3 is the closest analogue to this work: a small
model fine-tuned on a single jurisdiction's material for practical use under
limited compute.

\\paragraph{Retrieval versus fine-tuning.} Whether domain knowledge is better
injected by training on it or by retrieving it at inference time is an open
question with a substantial literature, and our central comparison sits inside
that debate rather than opening it. Retrieval-augmented
generation[[ref_rag]] conditions the generator on passages fetched at query
time. Ovadia et al.[[ref_ragvsft]] compare unsupervised fine-tuning against
retrieval for knowledge injection on MMLU-style factual tasks and find retrieval
consistently ahead; Soudani et al.[[ref_ragft_popularity]] report the same
ordering across twelve models and show the margin widens as the target knowledge
gets less popular; Balaguer et al.[[ref_ragft_agri]] run the comparison as an
industrial case study in agriculture and characterise the trade-offs of each
pipeline. The consistent finding across these is that retrieval wins on
knowledge, while fine-tuning contributes format, style and task behaviour.

Our results agree with that consensus, so the contribution is not the direction
of the effect but what it is measured on. Three differences matter. First, the
prior comparisons score whether the answer is factually right; a statutory
assistant is used differently, because the user's recourse is to read the
provision themselves, so we score whether the answer cites the \\emph{correct
section} of the governing statute -- a check the reader can perform and a
stricter target than factual agreement. Second, our test items are not drawn
from the same distribution as the training data: we report the same systems on
statute-derived hold-out questions and on hand-authored practitioner scenarios,
and show that the two disagree sharply about fine-tuning, which is a property of
the evaluation design that a single in-distribution test set cannot expose.
Third, we measure what the comparison costs in scope discipline -- retrieval
markedly reduces the model's willingness to decline out-of-scope questions --
which the knowledge-injection literature does not report because its benchmarks
contain no out-of-scope items. To our knowledge these three have not been
reported together for any statute.

Work on Bangladesh is comparatively sparse. BanglaBERT[[ref_banglabert]]
established general Bangla language understanding benchmarks, and recent studies
have begun to examine AI legal assistance for Bangladesh[[ref_ukil]] and the
reliability of LLMs in the Bengali legal context using both LLM-as-judge and
expert evaluation[[ref_bengali_reliability]]. To our knowledge no prior work
provides a leakage-free QA benchmark specific to the Bangladesh Labour Act
together with a comparison of fine-tuning and retrieval over that statute.

Methodologically we rely on parameter-efficient fine-tuning - LoRA[[ref_lora]]
and its quantized variant QLoRA[[ref_qlora]] - which makes adapting a 3B model
feasible on a single consumer GPU, and on self-instruct-style
generation[[ref_selfinstruct]] to build QA pairs from source text. Because
training data is generated from the statute, we also attend to the risk of
verbatim memorisation[[ref_carlini]], and evaluate on questions held out before
training.
"""

METHOD_DATASET = """
The dataset is built from the official English text of the Bangladesh Labour Act
2006 as amended to 2018[[ref_act]]. The Act is extracted from PDF, segmented, and
used to generate question-answer pairs in ChatML form, which are then extended
with paraphrases and applied scenarios and filtered for duplicates and
degenerate items.

Validation is the step that determines whether the dataset can support any claim
at all, so it is reported explicitly. Each generated answer is checked back
against the source text: section numbers mentioned in an answer must exist in the
Act, and answer content must be lexically supported by the Act's text. Items
failing these checks are corrected or discarded; the pipeline reports a 99.7%
verification rate against the source document for the retained set. Duplicate and
near-duplicate questions are removed so that the hold-out set cannot be answered
by memorising a paraphrase of a training item.

Because the model must also know the boundary of its competence, the training
mix includes explicit refusal examples in which a question outside the Act is
declined. Their contribution is measured directly in the ablation and in the
out-of-scope results, rather than assumed.

Splitting happens before training, not after. An earlier version of this pipeline
consumed the entire dataset and relied only on an internal random loss split,
which leaves no untouched material for testing and risks reporting memorised
answers as accuracy. We therefore carve a topic-stratified hold-out set out of
the cleaned dataset first and train only on the remainder; disjointness is
verified by hash-set intersection at both split time and training time.
"""

METHOD_SCENARIOS = """
Questions generated from statute text tend to mirror the statute's own phrasing,
which flatters any model trained on it. To test the system the way an HR
professional would actually use it, we additionally hand-authored a set of
applied scenarios that state a concrete situation and ask what follows - for
example, a worker who has completed twelve months of continuous service and
worked 216 days, and what paid annual leave that yields and under which
provision. Each scenario carries the section numbers that a correct answer must
rely on, which is what makes citation F1 measurable. The scenarios span leave,
overtime, wages, termination, maternity benefit, misconduct, probation,
compensation, trade unions, workplace safety and child labour.
"""

METHOD_TRAINING = """
We fine-tune Llama 3.2 3B Instruct[[ref_llama3]] with LoRA[[ref_lora]] adapters
(rank 16, alpha 16, no dropout) applied to the attention and MLP projection
matrices, using 4-bit base weights[[ref_qlora]] and Unsloth[[ref_unsloth]] for
memory-efficient training with gradient checkpointing at a 2048-token context.
Optimisation uses AdamW (8-bit) at a 2e-4 learning rate with linear decay, an
effective batch size of 8 (batch size 2, gradient accumulation 4), and a fixed
seed of 3407 for data order, initialisation and sampling. Training runs for up to
three epochs over the training pool with evaluation every 50 steps on a held-out
5% validation split, early stopping on validation loss with patience 3, and the
best checkpoint restored at the end - so the reported model is selected on
validation loss rather than by where a fixed step budget happened to stop.
The seed, GPU model, Python version and the versions of every training library
are recorded automatically into a run manifest alongside the trainer state, so
the reported configuration is read from the run rather than transcribed from it.
"""

METHOD_DEPLOY = """
The trained adapter is exported to GGUF (Q4\\_K\\_M) and served locally through
Ollama with a Modelfile that fixes the system prompt. The prompt defines an HR
consultant persona scoped to the Bangladesh Labour Act, instructs the model to
cite the relevant section, and instructs it to decline questions outside the Act.
The whole system runs on commodity hardware with no network egress, which is what
makes it usable on confidential HR material.
"""

EVAL_SETUP = """
\\paragraph{Systems compared.} We compare five configurations, all served locally
through the same interface so that differences reflect the model rather than the
harness: the base Llama 3.2 3B Instruct model; the fine-tuned model; a
retrieval-augmented[[ref_rag]] baseline in which the base model answers from the
top-4 passages retrieved from the Act; the fine-tuned model with the same
retrieval; and Qwen2.5 7B as a larger general-purpose reference point. Retrieval
uses cosine similarity over 800-character chunks of the Act embedded with
nomic-embed-text[[ref_nomic]]. The two retrieval configurations share one index,
so the base and fine-tuned models see identical passages for a given question
and any difference between them is attributable to the model rather than to
retrieval.

Comparison against commercial APIs (ChatGPT, Claude, Gemini) was not performed:
no API access was available for this study. We report this as a limitation rather
than estimating those numbers, and it is the most useful single extension of this
work.

\\paragraph{Test sets.} Three sets are used. The \\emph{hold-out} set contains
{N_HELDOUT} QA pairs carved out of the cleaned dataset before training and
stratified across HR topics. The \\emph{scenario} set contains {N_SCENARIO}
hand-authored applied HR situations, each labelled with the gold sections a
correct answer must cite. The \\emph{out-of-scope} set contains {N_OOS} questions
drawn from other domains, where the correct behaviour is to decline.

Disjointness is enforced on the normalised question text rather than on the
full question-answer record. The distinction is not pedantic: the generator
occasionally emitted the same question twice with different answers, so the two
copies differ as records and a record-level check declares them distinct while
the model has in fact been trained on a question it is later tested on. Four
such items were found and removed from the hold-out set, three of them genuine
duplicates -- one pair gives an adult working week as 48 hours in the test copy
and 56 in the training copy -- and one an extraction artefact whose question
text was the placeholder heading ``Scenario Question 2''. They were discarded
rather than returned to the training pool, so the training data is exactly what
the model was fine-tuned on.

\\paragraph{Decoding and reproducibility.} All systems are decoded greedily
(temperature 0) with a fixed seed and a 384-token limit, under an identical
system prompt, so that a re-run reproduces the reported numbers. Model digests,
library versions and the exact system prompt are written to a run provenance
record at generation time.

\\paragraph{Metrics.} BLEU[[ref_bleu]] and ROUGE-L[[ref_rouge]] are reported for
continuity with prior work, but they measure surface overlap with a reference
phrasing and a legally correct answer can score poorly on both. We therefore also
report: \\emph{citation validity}, the fraction of sections cited by the model
that exist in the Act, which detects fabricated authority directly;
\\emph{citation F1} against the gold sections, on the scenario set where gold
labels are complete; \\emph{grounding}, the fraction of the answer's 5-grams
occurring verbatim in the Act, a strict lexical support proxy; and
\\emph{refusal rate} on in-scope and out-of-scope questions.

Refusal is scored as declining to answer, which we distinguish from appending a
disclaimer. An answer that addresses an out-of-scope question in full and then
advises the reader to consult a professional has not declined it, and counting
such answers as refusals would overstate scope discipline precisely where the
measurement matters. A referral therefore counts as a refusal only when it
stands in place of an answer rather than following one.
{refusal_validation}

\\paragraph{LLM-as-judge.} Faithfulness, completeness, usefulness to an HR
professional, and harmfulness are scored 1--5 (harm 0--1) by {JUDGE_MODEL}, run
locally at temperature 0, following the LLM-as-judge protocol of Zheng et
al.[[ref_llmjudge]]. Two of the biases they
document are controlled by construction: the judge is a different model family
from every system it scores, which removes the self-enhancement case, and it
grades one answer at a time against the reference rather than ranking answers
side by side, so position bias does not arise. Verbosity bias is not controlled
and remains a caveat on the judged columns. This is an automatic proxy and is
labelled as such throughout; the fraction of judgements that parsed successfully
is reported alongside the scores so that a silently failing judge cannot be
mistaken for a low score. Zheng et al. validate such judges by agreement with
human preferences, and we have not run that validation here -- expert human
evaluation is set out below as the primary remaining gap, and until it is
collected the judged columns should be read as an unvalidated proxy.

\\paragraph{Statistical comparison.} Because every system answers the same
questions, systems are compared per item rather than by comparing two means.
For each metric we report the mean paired difference against the base model, a
95\\% confidence interval from a paired bootstrap over items (10{,}000
resamples), and a Wilcoxon signed-rank test, Holm-corrected across the systems
compared within that metric. The signed-rank test is used because the judge
scores are ordinal rather than interval. We treat a difference as significant
only when the adjusted $p$ is below 0.05 and the interval excludes zero, and we
report the effect size alongside, since a small difference that is consistent
across items can be significant without being important.

\\paragraph{Human evaluation.} A blind rating protocol and an offline rating
application are provided with this work: each question is shown with one answer
per system under neutral labels whose order is randomised per question, and
raters score accuracy, citation correctness, completeness, usefulness and
potential harm. The accompanying analysis script reports per-system means with
bootstrap intervals and inter-annotator agreement as Krippendorff's $\\alpha$
(ordinal for the rating scales, nominal for the harm flag), and reports the
agreement first, since means drawn from raters who disagree are not
interpretable. Ratings from qualified HR reviewers had not been collected at the
time of writing, so no human numbers are reported here. We deliberately report no
human column rather than an estimated one.
"""

DISCUSSION = """
Taken together these results argue against fine-tuning as the primary
intervention for this problem, and against the framing of our own earlier draft.
Retrieval over the Act supplies the statute's own words at inference time, and
that is what the citation metrics reward, because the provision the answer needs
is usually present verbatim in a retrieved passage. What fine-tuning contributes
is different in kind: it sets the model's default behaviour - its persona, its
habit of citing a section at all, and its register - and it does so without
requiring an index at serving time. Those are real benefits for deployment, but
they are not the same as being right about which section applies, and reporting
them as though they were is how a system evaluation overstates its case.

This reproduces, on a statute and with a citation-level criterion, what the
knowledge-injection literature reports on factual
benchmarks[[ref_ragvsft],[ref_ragft_popularity],[ref_ragft_agri]]: retrieval is
the effective route for getting domain knowledge into a model's answers, and
fine-tuning contributes behaviour rather than facts. The agreement is worth
stating plainly, because it means the result here should not be read as
particular to Bangladeshi labour law. Two things do not carry over from that
literature and are specific to this setting. Retrieval's advantage is larger
when scored on citing the correct section than on the reference-overlap metrics,
which is the criterion a statutory assistant is actually held to. And the
in-distribution hold-out set conceals the effect entirely -- fine-tuning's
citation F1 advantage is visible there and gone on practitioner scenarios -- so
a study of this kind that reports only a hold-out set can arrive at the opposite
conclusion from the same trained model.

The error profiles show the two failure modes are also different in kind, not
merely in size. On the scenarios, the base and fine-tuned models fail almost
entirely by citing a section that exists but does not govern the question; the
cited section is nearly always real, so the error is invisible to any check that
only verifies a citation resolves. The retrieval configurations fail instead by
declining to cite at all - answering out of the retrieved text without naming
the provision - which is their dominant failure mode on the hold-out set. These
call for opposite fixes: the first needs better section selection, the second
only needs the citation enforced in the output format. The second is much the
easier problem, which is a further practical argument for the retrieval
configurations.

Interpreting the metrics requires care. High lexical grounding indicates the
answer reuses the Act's language, which is desirable for a statutory assistant
but can also be achieved by quoting irrelevant text, which is why grounding is
read together with citation validity and judged completeness. Retrieval raises
grounding almost by construction, since the retrieved passage is in the context
window, so grounding should not be read as evidence that retrieval understands
the question better. Conversely, BLEU and ROUGE-L against a single reference
phrasing understate answers that are correct but worded differently; we report
them for comparability, not as the primary evidence. Citation F1 against
hand-assigned gold sections is the metric we would defend as closest to what an
HR user needs, and it is also the one on which the fine-tuned model's advantage
disappears.
"""

LIMITATIONS = """
\\paragraph{Single statute, single language.} The system covers only the
Bangladesh Labour Act 2006 as amended to 2018, in English. Bangladeshi labour
questions routinely involve subsidiary rules, notably the Bangladesh Labour Rules
2015, and adjacent instruments the model has never seen; it has no way to signal
that a question depends on material outside its scope rather than outside the
Act. Most HR practice in Bangladesh is also conducted in Bangla, and the dataset
is English-only.

\\paragraph{Statutory amendment.} The model encodes the Act at a fixed amendment
state. Any subsequent amendment silently invalidates the affected answers, and
the model will continue to answer with unchanged confidence. A deployed system
needs a review process tied to legislative updates; the retrieval configuration
degrades more gracefully here, since re-indexing an amended text is cheaper than
retraining.

\\paragraph{Synthetic training data.} Training questions were generated from the
Act's own text, so they inherit its phrasing and emphasis and under-represent how
practitioners actually ask questions - which is why the hand-authored scenario
set exists, and why results on it should be weighted more heavily than hold-out
results. Validation checks that answers are supported by the Act, but a
systematically skewed question distribution is not something answer-level
validation can detect.

\\paragraph{Residual hallucination.} Fine-tuning reduces but does not eliminate
fabricated section references; the error analysis reports the residual rate
rather than claiming it is zero. For any consequential decision the cited section
must be read in the Act itself. The system provides statute-grounded
informational support and is not legal advice.

\\paragraph{Evaluation is automatic.} All reported quality numbers are automatic
metrics or LLM-judge proxies. Expert human ratings, for which the protocol and
tooling are provided here, had not been collected at the time of writing.
Comparison against commercial APIs was not possible without API access. Both gaps
bound how strongly the results can be read.

\\paragraph{Scale.} The evaluation covers {N_TOTAL} questions across three sets
and a 3B model on a single GPU. Differences are tested with paired statistics,
but a {N_HELDOUT}-item hold-out set has limited power: a comparison reported as not
significant is evidence of an undetermined difference, not of no difference, and
small effects would need a larger benchmark to resolve. The significance tests
also treat the LLM judge's scores as data, so they inherit whatever bias that
judge has; they quantify sampling noise, not measurement validity.
"""

CONCLUSION = """
We presented a reproducible pipeline that turns the official PDF of the
Bangladesh Labour Act 2006 into a validated QA dataset, fine-tunes a 3B model on
it with LoRA, and deploys the result locally, together with a leakage-free
benchmark and a metric suite appropriate to legal QA. The benchmark - a
topic-stratified hold-out set, hand-authored HR scenarios with gold section
labels, and out-of-scope probes - and the accompanying evaluation code are the
part of this work most reusable by others, since they make claims about
Bangladesh labour-law QA checkable rather than assertable.

The results characterise what domain fine-tuning buys relative to retrieval over
the same statute, where each configuration fails, and how reliably each stays
within its scope. The most valuable extensions are the two gaps named above:
expert human evaluation, for which the blind protocol and tooling are already
provided, and comparison against commercial APIs. Beyond those, extending to the
Bangladesh Labour Rules 2015 and to Bangla would address the limitations that
most constrain practical use.
"""

REFERENCES = [
    ("ref_act", "Government of Bangladesh. The Bangladesh Labour Act, 2006 (Act No. XLII of 2006), amended to 2018."),
    ("ref_hallucination", "M. Dahl, V. Magesh, M. Suzgun, and D. E. Ho. Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models. Journal of Legal Analysis, 16(1):64--93, 2024."),
    ("ref_legaltools", "V. Magesh, F. Surani, M. Dahl, M. Suzgun, C. D. Manning, and D. E. Ho. Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools. arXiv:2405.20362, 2024."),
    ("ref_legalbench", "N. Guha, J. Nyarko, D. Ho, C. R\\'{e}, A. Chilton, et al. LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. NeurIPS Datasets and Benchmarks, 2023."),
    ("ref_legalbert", "I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Aletras, and I. Androutsopoulos. LEGAL-BERT: The Muppets Straight Out of Law School. In Findings of EMNLP, 2020, pp. 2898--2904."),
    ("ref_lexglue", "I. Chalkidis, A. Jana, D. Hartung, M. Bommarito, I. Androutsopoulos, D. M. Katz, and N. Aletras. LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. In Proc. ACL, 2022, pp. 4310--4330."),
    ("ref_cuad", "D. Hendrycks, C. Burns, A. Chen, and S. Ball. CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. NeurIPS Datasets and Benchmarks, 2021."),
    ("ref_jecqa", "H. Zhong, C. Xiao, C. Tu, T. Zhang, Z. Liu, and M. Sun. JEC-QA: A Legal-Domain Question Answering Dataset. In Proc. AAAI, 2020, pp. 9701--9708."),
    ("ref_saullm", "P. Colombo, T. P. Pires, M. Boudiaf, et al. SaulLM-7B: A Pioneering Large Language Model for Law. arXiv:2403.03883, 2024."),
    ("ref_chatlaw", "J. Cui, Z. Li, Y. Yan, B. Chen, and L. Yuan. ChatLaw: Open-Source Legal Large Language Model with Integrated External Knowledge Bases. arXiv:2306.16092, 2023."),
    ("ref_lawyerllama", "Q. Huang, M. Tao, C. Zhang, et al. Lawyer LLaMA Technical Report. arXiv:2305.15062, 2023."),
    ("ref_disclaw", "S. Yue, W. Chen, S. Wang, et al. DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services. arXiv:2309.11325, 2023."),
    ("ref_alkafi", "R. Qasem, M. Hendi, and B. Tantour. ALKAFI-LLAMA3: Fine-Tuning LLMs for Precise Legal Understanding in Palestine. arXiv:2412.14771, 2024."),
    ("ref_lora", "E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022. arXiv:2106.09685."),
    ("ref_qlora", "T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS, 2023."),
    ("ref_unsloth", "Unsloth. Efficient fine-tuning and inference for LLMs. \\url{https://github.com/unslothai/unsloth}, 2024."),
    ("ref_llama3", "A. Grattafiori, A. Dubey, et al. The Llama 3 Herd of Models. arXiv:2407.21783, 2024."),
    ("ref_selfinstruct", "Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi. Self-Instruct: Aligning Language Models with Self-Generated Instructions. In Proc. ACL, 2023, pp. 13484--13508."),
    ("ref_carlini", "N. Carlini, F. Tram\\`{e}r, E. Wallace, et al. Extracting Training Data from Large Language Models. In Proc. USENIX Security Symposium, 2021, pp. 2633--2650."),
    ("ref_banglabert", "A. Bhattacharjee, T. Hasan, W. U. Ahmad, K. Samin, et al. BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla. In Findings of NAACL, 2022, pp. 1318--1327."),
    ("ref_ukil", "A. T. Wasi, W. Faisal, M. R. Islam, and M. M. Bappy. Exploring Possibilities of AI-Powered Legal Assistance in Bangladesh through Large Language Modeling. arXiv:2410.17210, 2024."),
    ("ref_bengali_reliability", "S. Aftahee, A. F. M. Farhad, A. Mallik, R. Dhar, J. Karim, N. B. Noor, and I. A. Solaiman. Assessing the Reliability of Large Language Models in the Bengali Legal Context: A Comparative Evaluation Using LLM-as-Judge and Legal Experts. arXiv:2511.05627, 2025."),
    ("ref_bleu", "K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu. BLEU: a Method for Automatic Evaluation of Machine Translation. In Proc. ACL, 2002, pp. 311--318."),
    ("ref_rouge", "C.-Y. Lin. ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out, ACL Workshop, 2004, pp. 74--81."),
    ("ref_rag", "P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. K\\\"{u}ttler, M. Lewis, W.-t. Yih, T. Rockt\\\"{a}schel, S. Riedel, and D. Kiela. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems (NeurIPS), 2020. arXiv:2005.11401."),
    ("ref_ragvsft", "O. Ovadia, M. Brief, M. Mishaeli, and O. Elisha. Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs. arXiv:2312.05934, 2024."),
    ("ref_ragft_agri", "A. Balaguer, V. Benara, R. L. de Freitas Cunha, R. de M. Estev\\~{a}o Filho, T. Hendry, D. Holstein, J. Marsman, N. Mecklenburg, S. Malvar, L. O. Nunes, R. Padilha, M. Sharp, B. Silva, S. Sharma, V. Aski, and R. Chandra. RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture. arXiv:2401.08406, 2024."),
    ("ref_ragft_popularity", "H. Soudani, E. Kanoulas, and F. Hasibi. Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge. In Proc. Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP), 2024. arXiv:2403.01432."),
    ("ref_llmjudge", "L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In NeurIPS Datasets and Benchmarks Track, 2023. arXiv:2306.05685."),
    ("ref_nomic", "Z. Nussbaum, J. X. Morris, B. Duderstadt, and A. Mulyar. Nomic Embed: Training a Reproducible Long Context Text Embedder. arXiv:2402.01613, 2024."),
]

def eval_setup_text():
    """EVAL_SETUP with measured values substituted in."""
    return EVAL_SETUP.replace("{refusal_validation}",
                              refusal_validation_sentence())


def discussion_text():
    """DISCUSSION, led by the findings recomputed from the artifacts."""
    return key_findings_text() + "\n\n" + DISCUSSION.strip()


REF_KEYS = [k for k, _ in REFERENCES]
_CITE_RE = re.compile(r"\[\[([a-z_0-9]+)\]\]")


def render_citations(text, style):
    """Render [[ref_key]] markers.

    style='latex' -> \\cite{a,b} (adjacent markers merged)
    style='plain' -> [1, 2] numeric, matching REFERENCES order
    """
    idx = {k: i + 1 for i, k in enumerate(REF_KEYS)}

    def merge(m):
        keys = _CITE_RE.findall(m.group(0))
        unknown = [k for k in keys if k not in idx]
        if unknown:
            raise KeyError(f"unknown citation key(s): {unknown}")
        if style == "latex":
            return "\\cite{" + ",".join(keys) + "}"
        return "[" + ", ".join(str(idx[k]) for k in keys) + "]"

    return re.sub(r"(?:\[\[[a-z_0-9]+\]\])+", merge, text)


def strip_latex(text):
    """Convert the LaTeX-flavoured inline markup in the prose to plain text."""
    text = re.sub(r"\\paragraph\{([^}]*)\}", r"\1 ", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\texttt\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\url\{([^}]*)\}", r"\1", text)
    text = re.sub(r"``([^']*)''", r'"\1"', text)
    text = text.replace("\\%", "%").replace("\\_", "_").replace("\\&", "&")
    text = text.replace("\\~{a}", "a").replace('\\"{u}', "u").replace('\\"{a}', "a")
    text = text.replace("\\`{e}", "e").replace("--", "-")
    return text


_COUNT_PLACEHOLDERS = {
    "N_HELDOUT": lambda: set_size("heldout"),
    "N_SCENARIO": lambda: set_size("scenario"),
    "N_OOS": lambda: set_size("oos"),
    "N_TOTAL": lambda: sum(set_size(s) for s in ("heldout", "scenario", "oos")),
    "N_HELDOUT_GOLD": lambda: gold_count("heldout"),
    "N_SCENARIO_GOLD": lambda: gold_count("scenario"),
    # Named here rather than written into the prose because the paper argues
    # the judge is from a different model family than any system it scores;
    # that argument is only checkable if the name comes from the run record.
    #
    # Substituted text must be format-neutral. The DOCX path runs strip_latex
    # *before* paragraphs(), which is where substitution happens, so any LaTeX
    # emitted here is past the point where it would have been stripped and
    # reaches the Word file verbatim.
    "JUDGE_MODEL": lambda: (load_provenance() or {}).get(
        "judge_model", "a local judge model"),
}
# Matches any all-caps placeholder, but substitutes only known ones - LaTeX
# such as \textbf{ABC} would match the pattern and must survive untouched.
_COUNT_RE = re.compile(r"\{([A-Z][A-Z_0-9]*)\}")


def fill_counts(text):
    """Substitute {N_...} placeholders with counts read from the test sets.

    The prose blocks are module-level constants, so they cannot be f-strings.
    Without this the sizes get typed into the text by hand, which is how six
    sentences came to claim a 150-item hold-out set after it became 146.

    This matches only the specific {N_NAME} spellings rather than running
    str.format over the block. The prose is LaTeX, so it is full of braces --
    \\emph{x}, 10{,}000, @{} -- and handing that to a formatter raises on the
    first one it cannot parse as a field.
    """
    return _COUNT_RE.sub(
        lambda m: str(_COUNT_PLACEHOLDERS[m.group(1)]())
        if m.group(1) in _COUNT_PLACEHOLDERS else m.group(0), text)


def paragraphs(text):
    """Split a prose block into paragraphs."""
    return [re.sub(r"\s+", " ", p).strip()
            for p in fill_counts(text).strip().split("\n\n") if p.strip()]
