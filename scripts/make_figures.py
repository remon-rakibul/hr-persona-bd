#!/usr/bin/env python3
"""
Generate every figure in the paper.

Figure code used to live inside generate_publication_docx.py, which meant the
DOCX build was the only way to refresh a figure and the LaTeX build silently
reused whatever PNGs happened to be on disk. It lives here now, and both
generators consume the output.

Figures:
  1 dataset creation pipeline          (diagram)
  2 fine-tuning and deployment         (diagram)
  3 evaluation design                  (diagram)
  4 training loss                      (from trainer_state.json, else pilot run)
  5 evaluation loss / perplexity       (two panels - see note below)
  6 dataset statistics                 (from the dataset files)
  7 results comparison                 (from results/comparison.csv)
  8 error analysis                     (from results/error_analysis.csv)

Figures 6-8 render only when their source artifact exists, so a partial run
produces the figures it has evidence for rather than inventing the rest.

Two deliberate design choices:
  * Figure 5 is two panels, not one plot with twin y-axes. Perplexity is
    exp(eval loss), so a second axis adds a second scale for a monotone
    transform of the same series - the reader has to decode two axes to read one
    fact.
  * Colours are a CVD-validated categorical palette, and every bar carries its
    value as a direct label so the figures stay readable in greyscale print.

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --trainer-state runs/main_full/trainer_state.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for d in (PROJECT_ROOT, SCRIPT_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import publication_content as P

DEFAULT_FIG_DIR = PROJECT_ROOT / "publication" / "figures"
DPI = 300

# Categorical palette, validated for colour-vision deficiency separation against
# a light surface (adjacent-pair dE >= 9.1 protan, >= 19.6 normal vision).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#1a1a19"
MUTED = "#52514e"
GRID = "#d9d9d6"
BOX_FILL = "#E8F4FC"
BOX_EDGE = "#2E86AB"

plt.rcParams.update({
    "font.size": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig, out_path: Path) -> Path:
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path.name}")
    return out_path


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        if side not in keep:
            ax.spines[side].set_visible(False)


# ---------------------------------------------------------------- diagrams --

def _draw_flowchart_boxes(ax, boxes, box_width=0.72, box_height=0.10, gap=0.14,
                          top=1.0, title=None, title_offset=0.08):
    """Vertical flowchart: centred boxes joined by arrows."""
    x_center = 0.5
    x_left = x_center - box_width / 2
    bottom_y = top - (len(boxes) - 1) * (box_height + gap) - box_height / 2
    if title:
        ax.text(x_center, top + title_offset, title, ha="center", va="center",
                fontsize=11, fontweight="bold")
    for i, label in enumerate(boxes):
        y_center = top - i * (box_height + gap)
        y_bottom = y_center - box_height / 2
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_left, y_bottom), box_width, box_height,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            facecolor=BOX_FILL, edgecolor=BOX_EDGE, linewidth=1.2))
        ax.text(x_center, y_center, label, ha="center", va="center", fontsize=8,
                wrap=True, ma="center")
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x_center, y_center - (box_height + gap) + box_height / 2),
                        xytext=(x_center, y_bottom),
                        arrowprops=dict(arrowstyle="->", color=BOX_EDGE, lw=2))
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom_y - 0.05,
                (top + title_offset + 0.03) if title else top + 0.05)
    ax.axis("off")


def figure1_dataset_pipeline(out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 6.5))
    ax.grid(False)
    _draw_flowchart_boxes(ax, [
        "Source PDF (Bangladesh Labour Act 2006, amended 2018)",
        "Text extraction",
        "Chunking (4000 chars, 200 overlap)",
        "QA generation (per chunk)",
        "Extension (variations, follow-ups, scenarios)",
        "Validation (structure, section refs, PDF verification)",
        "Deduplication and cleaning",
        "Leakage-free split (train pool / hold-out)",
    ], box_width=0.76, box_height=0.095, gap=0.105, top=0.90,
        title="Figure 1: Dataset creation and validation pipeline")
    fig.tight_layout(pad=1.2)
    return _save(fig, out_path)


def figure2_finetuning_pipeline(out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 6.2))
    ax.grid(False)
    _draw_flowchart_boxes(ax, [
        "Llama 3.2 3B Instruct",
        "Load in 4-bit + LoRA (r=16)",
        "SFT on training pool (95/5 train/val)",
        "Early stopping on validation loss",
        "Save best LoRA adapter",
        "Export GGUF Q4_K_M",
        "Ollama Modelfile (system prompt)",
        "Local deployment",
    ], box_width=0.74, box_height=0.10, gap=0.12, top=0.86,
        title="Figure 2: Fine-tuning and deployment pipeline", title_offset=0.14)
    fig.tight_layout(pad=1.2)
    return _save(fig, out_path)


def figure3_evaluation_design(out_path: Path) -> Path:
    """Evaluation design. Names the systems actually compared."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.grid(False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    y_center, box_h, gap = 1.5, 1.25, 0.4
    y_bottom = y_center - box_h / 2
    specs = [
        (0.05, 2.5, ["Test sets", "hold-out (150)", "scenarios (38)",
                     "out-of-scope (20)"], BOX_FILL),
        (2.95, 1.05, ["Identical", "prompt,", "greedy,", "seeded"], BOX_FILL),
        (4.40, 2.5, ["Systems", "base | fine-tuned", "RAG | RAG+fine-tuned",
                     "Qwen2.5 7B"], "#FFF4E6"),
        (7.30, 2.6, ["Metrics", "BLEU, ROUGE-L, grounding",
                     "citation validity / F1",
                     "judge: faithful, complete,", "useful, harm; refusal"], BOX_FILL),
    ]
    for x, w, lines, color in specs:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y_bottom), w, box_h,
            boxstyle="round,pad=0.03,rounding_size=0.05",
            facecolor=color, edgecolor=BOX_EDGE, linewidth=1.2))
        ax.text(x + w / 2, y_center, "\n".join(lines), ha="center", va="center",
                fontsize=7, ma="center")
    for x_end, x_start in [(2.55, 2.95), (4.00, 4.40), (6.90, 7.30)]:
        ax.annotate("", xy=(x_start - 0.04, y_center), xytext=(x_end + 0.04, y_center),
                    arrowprops=dict(arrowstyle="->", color=BOX_EDGE, lw=2))
    ax.set_title("Figure 3: Comparative evaluation design", fontsize=11, pad=10)
    fig.tight_layout(pad=1.2)
    return _save(fig, out_path)


# ------------------------------------------------------------- training ----

def _training_curves(trainer_state_path):
    """(steps, train_loss, eval_steps, eval_loss) from a run, else the pilot."""
    tm = P.load_training_metrics(trainer_state_path)
    if tm:
        st, tl, es, el, seen = [], [], [], [], set()
        for e in tm["log"]:
            if "loss" in e:
                st.append(e.get("step", len(st)))
                tl.append(e["loss"])
            if "eval_loss" in e:
                # load_best_model_at_end re-evaluates the restored checkpoint
                # under the final step number. Plotting that point would draw
                # validation loss dropping at the end of a run that was stopped
                # precisely because validation loss had started rising.
                if e.get("step") in seen:
                    continue
                seen.add(e.get("step"))
                es.append(e.get("step", len(es)))
                el.append(e["eval_loss"])
        if st and es:
            return st, tl, es, el, False
    steps = [s for s, _, _ in P.PILOT_STEPS]
    return (steps, [a for _, a, _ in P.PILOT_STEPS],
            steps, [b for _, _, b in P.PILOT_STEPS], True)


def _early_stop_marks(trainer_state_path=None):
    """(best_step, stop_step) if the run was halted by early stopping."""
    tm = P.load_training_metrics(trainer_state_path)
    if not tm:
        return None, None
    state = tm["state"]
    ck = str(state.get("best_model_checkpoint") or "")
    tail = ck.rsplit("-", 1)
    best = int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None
    ran, budget = state.get("global_step"), state.get("max_steps")
    return best, (ran if ran and budget and ran < budget else None)


def figure4_training_loss(out_path: Path, trainer_state_path=None) -> Path:
    st, tl, es, el, pilot = _training_curves(trainer_state_path)
    best, stop = (None, None) if pilot else _early_stop_marks(trainer_state_path)

    # Two panels. On a single axis the run's whole story is invisible: training
    # loss starts near 3.5, so the divergence between the curves - which happens
    # inside a 0.08 band - is a few pixels tall. The right panel is the same
    # validation curve on its own scale, which is where the reader can actually
    # see the minimum and the rise that stopped the run.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    ax.plot(st, tl, "-o", color=SERIES[0], linewidth=2, markersize=3,
            label="Training loss")
    ax.plot(es, el, "--s", color=SERIES[1], linewidth=2, markersize=4,
            label="Validation loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Full range", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)

    ax2.plot(es, el, "--s", color=SERIES[1], linewidth=2, markersize=4)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Validation loss")
    ax2.set_title("Validation loss, own scale", fontsize=9, color=MUTED)
    if best is not None:
        by = el[es.index(best)] if best in es else None
        if by is not None:
            ax2.plot([best], [by], "o", color=SERIES[3], markersize=8,
                     zorder=5)
            ax2.annotate(f"best: step {best}", (best, by),
                         textcoords="offset points", xytext=(-6, -14),
                         ha="right", fontsize=8, color=SERIES[3])
        if stop is not None:
            ax2.axvspan(best, stop, color=SERIES[3], alpha=0.10, lw=0)
            ax2.annotate("patience window\n(stopped here)",
                         ((best + stop) / 2, max(el)),
                         textcoords="offset points", xytext=(0, -4),
                         ha="center", va="top", fontsize=7, color=MUTED)
    ax2.margins(y=0.28)
    _despine(ax2)

    fig.suptitle("Figure 4: Training and validation loss"
                 + (" (pilot run)" if pilot else ""), fontsize=10)
    fig.tight_layout()
    return _save(fig, out_path)


def figure5_eval_loss_perplexity(out_path: Path, trainer_state_path=None) -> Path:
    """Two panels rather than twin y-axes.

    Perplexity is exp(eval loss); plotting both against a shared x on two y-scales
    asks the reader to decode two axes to read one quantity.
    """
    _, _, es, el, pilot = _training_curves(trainer_state_path)
    pp = [math.exp(x) for x in el]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, ys, lab, col in ((axes[0], el, "Validation loss", SERIES[0]),
                             (axes[1], pp, "Perplexity", SERIES[2])):
        ax.plot(es, ys, "-o", color=col, linewidth=2, markersize=5)
        ax.set_xlabel("Step")
        ax.set_ylabel(lab)
        # Label every point only while there are few of them. The pilot logged
        # 4 evaluations; the full run logs 17, and labelling all of those
        # produces overlapping text that hides the curve it annotates.
        if len(es) <= 6:
            mark = list(zip(es, ys))
        else:
            lo = min(range(len(ys)), key=lambda i: ys[i])
            mark = [(es[i], ys[i]) for i in sorted({0, lo, len(ys) - 1})]
        for x, y in mark:
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=MUTED)
        ax.margins(y=0.22)
        _despine(ax)
    fig.suptitle("Figure 5: Validation loss and perplexity"
                 + (" (pilot run)" if pilot else ""), fontsize=10)
    fig.tight_layout()
    return _save(fig, out_path)


# ---------------------------------------------------------------- results --

def figure6_dataset_statistics(out_path: Path) -> Path:
    """Topic coverage of the two in-scope test sets, and the split sizes."""
    import collections
    eval_dir = PROJECT_ROOT / "data/eval"
    try:
        held = json.load(open(eval_dir / "heldout_test.json", encoding="utf-8"))
        scen = json.load(open(eval_dir / "scenario_test.json", encoding="utf-8"))
    except Exception as e:
        print(f"  skip figure6 ({e})")
        return out_path

    hc = collections.Counter(P.canon_topic(i.get("topic")) for i in held)
    sc = collections.Counter(P.canon_topic(i.get("topic")) for i in scen)
    topics = sorted(set(hc) | set(sc), key=lambda t: -(hc[t] + sc[t]))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    y = range(len(topics))
    h = 0.38
    ax.barh([i + h / 2 for i in y], [hc[t] for t in topics], height=h,
            color=SERIES[0], label=f"Hold-out (n={len(held)})")
    ax.barh([i - h / 2 for i in y], [sc[t] for t in topics], height=h,
            color=SERIES[1], label=f"Scenarios (n={len(scen)})")
    for i, t in enumerate(topics):
        if hc[t]:
            ax.text(hc[t] + 0.7, i + h / 2, str(hc[t]), va="center", fontsize=6.5,
                    color=MUTED)
        if sc[t]:
            ax.text(sc[t] + 0.7, i - h / 2, str(sc[t]), va="center", fontsize=6.5,
                    color=MUTED)
    ax.set_yticks(list(y))
    ax.set_yticklabels([t.replace("_", " ") for t in topics], fontsize=7)
    ax.set_xlabel("Questions")
    ax.set_title("Topic coverage of the test sets", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", visible=False)
    _despine(ax)

    ax = axes[1]
    _, _, rows, _ = P.table_dataset()
    labels = [r[0].strip() for r in rows if not r[0].startswith("  ")]
    values = [int(r[1]) for r in rows if not r[0].startswith("  ") and r[1].isdigit()]
    bars = ax.bar(range(len(values)), values,
                  color=[SERIES[2]] + [SERIES[0]] * (len(values) - 1), width=0.62)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center",
                va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labels], fontsize=7)
    ax.set_ylabel("Items")
    ax.set_yscale("log")
    ax.set_title("Split sizes (log scale)", fontsize=9)
    ax.grid(axis="x", visible=False)
    _despine(ax)

    fig.suptitle("Figure 6: Dataset statistics", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_path)


# Each metric gets its own panel: BLEU is 0-100, grounding 0-1 and the judge
# scores 1-5, so a single shared axis would compress most of them to nothing.
RESULT_PANELS = [
    ("rougeL", "ROUGE-L (0-100)", None),
    ("grounding", "Grounding (0-1)", None),
    ("cited_valid", "Cited sections valid (0-1)", None),
    ("faithfulness", "Faithfulness (1-5)", (0, 5)),
    ("usefulness", "Usefulness (1-5)", (0, 5)),
    ("refusal_rate", "Out-of-scope refusal (0-1)", (0, 1)),
]


def figure7_results_comparison(out_path: Path, setname="heldout") -> Path:
    comp = P.load_comparison()
    if not comp:
        print("  skip figure7 (no results/comparison.csv)")
        return out_path

    systems = [(k, lab) for k, lab in P.SYSTEM_LABELS
               if (k, setname) in comp or (k, "oos") in comp]
    if not systems:
        print("  skip figure7 (no rows)")
        return out_path

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.4))
    for ax, (metric, title, ylim) in zip(axes.flat, RESULT_PANELS):
        src = "oos" if metric == "refusal_rate" else setname
        vals, labs, cols = [], [], []
        for i, (k, lab) in enumerate(systems):
            row = comp.get((k, src))
            v = None
            if row:
                try:
                    v = float(row.get(metric) or "nan")
                except ValueError:
                    v = None
            if v is None or v != v:
                continue
            vals.append(v)
            labs.append(lab)
            cols.append(SERIES[i % len(SERIES)])
        if not vals:
            ax.axis("off")
            continue
        bars = ax.bar(range(len(vals)), vals, color=cols, width=0.64)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=6.5, color=MUTED)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels([l.replace(" ", "\n") for l in labs], fontsize=6)
        ax.set_title(title, fontsize=8.5)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.margins(y=0.18)
        ax.grid(axis="x", visible=False)
        _despine(ax)
    fig.suptitle(f"Figure 7: System comparison ({setname} set; "
                 "refusal panel is the out-of-scope set)", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_path)


def figure8_error_analysis(out_path: Path, setname="heldout") -> Path:
    rows = P.load_error_analysis()
    if not rows:
        print("  skip figure8 (no results/error_analysis.csv)")
        return out_path
    by_sys = {r["system"]: r for r in rows if r["set"] == setname}
    systems = [(k, lab) for k, lab in P.SYSTEM_LABELS if k in by_sys]
    if not systems:
        print("  skip figure8 (no rows)")
        return out_path

    cats = [c for c, _ in P.ERROR_COLS]
    labels = [lab for _, lab in P.ERROR_COLS]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    n = len(systems)
    width = 0.8 / n
    for i, (k, lab) in enumerate(systems):
        r = by_sys[k]
        total = float(r.get("n") or 1)
        vals = [100.0 * float(r.get(c) or 0) / total for c in cats]
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(cats))]
        bars = ax.bar(xs, vals, width=width * 0.92,
                      color=SERIES[i % len(SERIES)], label=lab)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center",
                        va="bottom", fontsize=6, color=MUTED)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("% of answers in the set")
    ax.set_title(f"Figure 8: Failure modes by system ({setname} set)", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="x", visible=False)
    _despine(ax)
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------

def build_all(fig_dir: Path, trainer_state_path=None):
    _ensure_dir(fig_dir)
    print(f"Writing figures to {fig_dir}")
    figure1_dataset_pipeline(fig_dir / "figure1_dataset_pipeline.png")
    figure2_finetuning_pipeline(fig_dir / "figure2_finetuning_pipeline.png")
    figure3_evaluation_design(fig_dir / "figure3_evaluation_design.png")
    figure4_training_loss(fig_dir / "figure4_training_loss.png", trainer_state_path)
    figure5_eval_loss_perplexity(fig_dir / "figure5_eval_loss_perplexity.png",
                                 trainer_state_path)
    figure6_dataset_statistics(fig_dir / "figure6_dataset_statistics.png")
    figure7_results_comparison(fig_dir / "figure7_results_comparison.png")
    figure8_error_analysis(fig_dir / "figure8_error_analysis.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--trainer-state", type=Path, default=None)
    args = ap.parse_args()
    build_all(args.fig_dir, args.trainer_state)


if __name__ == "__main__":
    main()
