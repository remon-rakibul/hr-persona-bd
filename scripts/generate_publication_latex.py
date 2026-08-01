#!/usr/bin/env python3
"""
Generate the LaTeX version of the publication for Overleaf.

Output: publication/overleaf/main.tex, publication/overleaf/figures/*.png, and
        publication/Haque_etal_LLM_Bangladesh_Labour_Law_overleaf.zip

All prose and every table come from scripts/publication_content.py, which loads
the results from the measurement artifacts. This script only formats them; it
contains no research numbers of its own, so a value cannot drift between the
LaTeX and DOCX versions or be edited into the paper by hand.

Usage:
    python scripts/generate_publication_latex.py
    python scripts/generate_publication_latex.py --trainer-state runs/main_full/trainer_state.json
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for d in (PROJECT_ROOT, SCRIPT_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import publication_content as P

DEFAULT_FIG_DIR = PROJECT_ROOT / "publication" / "figures"
DEFAULT_OVERLEAF_DIR = PROJECT_ROOT / "publication" / "overleaf"
DEFAULT_ZIP_PATH = (PROJECT_ROOT / "publication"
                    / "Haque_etal_LLM_Bangladesh_Labour_Law_overleaf.zip")

FIGURES = [
    ("figure1_dataset_pipeline.png", "Dataset creation and validation pipeline.", "fig:data"),
    ("figure2_finetuning_pipeline.png", "Fine-tuning and deployment pipeline.", "fig:train"),
    ("figure3_evaluation_design.png", "Comparative evaluation design.", "fig:eval"),
    ("figure4_training_loss.png", "Training loss curve.", "fig:loss"),
    ("figure5_eval_loss_perplexity.png", "Evaluation loss and perplexity over training.", "fig:perp"),
    ("figure6_dataset_statistics.png", "Dataset statistics: topic distribution across the training pool and test sets.", "fig:datastats"),
    ("figure7_results_comparison.png", "Comparison of the evaluated systems.", "fig:results"),
    ("figure8_error_analysis.png", "Distribution of failure modes by system.", "fig:errors"),
]


def esc(s: str) -> str:
    """Escape LaTeX specials in plain text (leaves intentional markup alone)."""
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
             .replace("_", "\\_").replace("{", "\\{").replace("}", "\\}")
             .replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")
             .replace("$", "\\$"))


def prose(block: str) -> str:
    """Render a prose block: citations to \\cite, paragraphs separated."""
    return "\n\n".join(P.paragraphs(P.render_citations(block, "latex")))


def table(caption, header, rows, note, label=None, small=True):
    """Render a booktabs table, or the note if there are no measured rows."""
    if not rows:
        return ("\\begin{quote}\\itshape " + esc(note or P.NOT_MEASURED)
                + "\\end{quote}\n")
    align = "@{}l" + "c" * (len(header) - 1) + "@{}"
    out = ["\\begin{table}[htbp]", "\\centering",
           "\\renewcommand{\\arraystretch}{1.15}"]
    if caption:
        out.append("\\caption{" + caption + "}")
    if label:
        out.append("\\label{" + label + "}")
    if small:
        out.append("\\small")
    out += ["\\begin{tabular}{" + align + "}", "\\toprule",
            " & ".join("\\textbf{" + esc(h) + "}" for h in header) + " \\\\",
            "\\midrule"]
    for r in rows:
        out.append(" & ".join(esc(str(c)) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def figure(name, caption, label, fig_dir, width="0.85"):
    if not (fig_dir / name).exists():
        return ""
    return ("\n".join([
        "\\begin{figure}[htbp]", "\\centering",
        f"\\includegraphics[width={width}\\textwidth,height=0.42\\textheight,"
        f"keepaspectratio]{{figures/{name}}}",
        "\\caption{" + caption + "}", "\\label{" + label + "}",
        "\\end{figure}", ""]))


def build_main_tex(trainer_state_path, fig_dir) -> str:
    figs = {name: (cap, lab) for name, cap, lab in FIGURES}

    def fig(name, width="0.85"):
        cap, lab = figs[name]
        return figure(name, cap, lab, fig_dir, width)

    L = []
    add = L.append

    add(r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage[margin=2.5cm]{geometry}
\usepackage{setspace}
\onehalfspacing
\usepackage{float}
\usepackage{graphicx}
\usepackage[font=small,labelfont=bf]{caption}
\captionsetup{skip=4pt}
\usepackage{booktabs}
\usepackage{array}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
""")
    add("\\title{" + P.TITLE + "}")
    add("\\author{" + " \\\\ ".join(P.AUTHORS) + "}")
    add("\\date{}\n")
    add(r"\begin{document}" + "\n" + r"\maketitle" + "\n" + r"\vspace{1em}" + "\n")

    add(r"\begin{abstract}" + "\n\\noindent " + prose(P.ABSTRACT) + "\n" + r"\end{abstract}")
    add("")

    add(r"\section{Introduction}")
    add(prose(P.INTRO))

    add(r"\section{Related Work}")
    add(prose(P.RELATED))

    add(r"\section{Methodology}")
    add(r"\subsection{Dataset creation and validation}")
    add(prose(P.METHOD_DATASET))
    add(fig("figure1_dataset_pipeline.png", "0.78"))
    c, h, r, n = P.table_dataset()
    add(table(c, h, r, n, "tab:dataset"))
    add(fig("figure6_dataset_statistics.png"))

    add(r"\subsection{Applied HR scenarios}")
    add(prose(P.METHOD_SCENARIOS))
    c, h, r, n = P.table_scenario_verification()
    add(table(c, h, r, n, "tab:scenverify"))

    examples = P.example_qa_pairs(3)
    if examples:
        add("Three representative items, one per topic, are shown below.\n")
        add(r"\begin{quote}\small")
        for ex in examples:
            add("\\textbf{[" + esc(str(ex["topic"])) + "]} "
                + esc(ex["question"]) + "\\\\[2pt]")
            add("\\textit{Reference:} " + esc(ex["reference"])
                + " \\quad \\textit{Gold sections:} "
                + esc(", ".join(str(s) for s in ex["gold_sections"]))
                + "\\\\[6pt]")
        add(r"\end{quote}")

    add(r"\subsection{Model and training}")
    add(prose(P.METHOD_TRAINING))
    add(fig("figure2_finetuning_pipeline.png"))

    add(r"\subsection{Deployment}")
    add(prose(P.METHOD_DEPLOY))

    add(r"\section{Experimental setup}")
    add(prose(P.eval_setup_text()))
    add(fig("figure3_evaluation_design.png"))

    add(r"\section{Results}")

    add(r"\subsection{Training}")
    c, h, r, n = P.table_training(trainer_state_path)
    add(table(c, h, r, n, "tab:training", small=False))
    c, h, r, n = P.table_training_steps(trainer_state_path)
    add(table(c, h, r, n, "tab:trainsteps", small=False))
    add(fig("figure4_training_loss.png", "0.88"))
    add(fig("figure5_eval_loss_perplexity.png", "0.88"))

    add(r"\subsection{Answer quality}")
    c, h, r, n = P.table_main_comparison("heldout")
    add(table(c, h, r, n, "tab:heldout"))
    c, h, r, n = P.table_main_comparison("scenario")
    add(table(c, h, r, n, "tab:scenario"))
    add(fig("figure7_results_comparison.png"))

    add(r"\subsection{Statistical comparison}")
    c, h, r, n = P.table_significance("heldout")
    add(table(c, h, r, n, "tab:significance"))
    c, h, r, n = P.table_significance("scenario")
    add(table(c, h, r, n, "tab:significance_scn"))

    add(r"\subsection{Citation accuracy}")
    c, h, r, n = P.table_citation("scenario")
    add(table(c, h, r, n, "tab:citation"))

    add(r"\subsection{Scope discipline}")
    c, h, r, n = P.table_refusal()
    add(table(c, h, r, n, "tab:refusal"))

    add(r"\subsection{Error analysis}")
    c, h, r, n = P.table_error_analysis("heldout")
    add(table(c, h, r, n, "tab:errors"))
    add(fig("figure8_error_analysis.png"))

    add(r"\subsection{Ablation}")
    c, h, r, n = P.table_ablation()
    add(table(c, h, r, n, "tab:ablation"))
    add(prose(P.training_findings_text()))

    add(r"\section{Discussion}")
    add(prose(P.discussion_text()))

    add(r"\section{Limitations}")
    add(prose(P.LIMITATIONS))

    add(r"\section{Conclusion}")
    add(prose(P.CONCLUSION))

    add(r"\section*{Appendix: reproducibility}")
    add(esc(P.reproducibility_note()))

    add("\n" + r"\begin{thebibliography}{99}")
    for key, text in P.REFERENCES:
        add("\\bibitem{" + key + "} " + text)
    add(r"\end{thebibliography}")
    add(r"\end{document}")

    return "\n".join(x for x in L if x is not None) + "\n"


def check_output(tex: str) -> list[str]:
    """Catch the mistakes that survive a successful build but ruin the PDF.

    Neither of these is visible in the generated file unless looked for, and
    both shipped once. A stray ``%`` is the worse of the two: it comments out
    the rest of its line, so the paper silently loses a sentence rather than
    failing to compile. An unrendered ``[[key]]`` means a citation marker was
    written in a form render_citations does not match - it merges *adjacent*
    ``[[a]][[b]]`` markers, so any separator between them stops the match and
    the raw marker is typeset.
    """
    problems = []

    # Command arguments are not typeset, so specials inside them are fine.
    masked = re.sub(
        r"\\(cite|label|ref|bibitem|includegraphics|url)\s*(\[[^\]]*\])?\{[^}]*\}",
        lambda m: "X" * len(m.group(0)), tex)
    for i, line in enumerate(masked.split("\n"), 1):
        if re.search(r"(?<!\\)%", line):
            problems.append(f"line {i}: unescaped % (comments out the rest of "
                            f"the line in the PDF)")
    for m in re.finditer(r"\[\[[a-z_0-9]+\]?", tex):
        problems.append(f"unrendered citation marker {m.group(0)!r} - adjacent "
                        f"markers merge, separators between them do not")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overleaf-dir", type=Path, default=DEFAULT_OVERLEAF_DIR)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    ap.add_argument("--trainer-state", type=Path, default=None)
    args = ap.parse_args()

    overleaf_dir = args.overleaf_dir.resolve()
    figures_dst = overleaf_dir / "figures"
    figures_dst.mkdir(parents=True, exist_ok=True)

    tex = build_main_tex(args.trainer_state, args.fig_dir)

    problems = check_output(tex)
    if problems:
        print("Refusing to write: the generated LaTeX has problems that would "
              "reach the PDF.", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    (overleaf_dir / "main.tex").write_text(tex, encoding="utf-8")
    print(f"Wrote {overleaf_dir / 'main.tex'} ({len(tex.splitlines())} lines)")

    copied = []
    for name, _, _ in FIGURES:
        src = args.fig_dir / name
        if src.exists():
            shutil.copy2(src, figures_dst / name)
            copied.append(name)
    print(f"Copied {len(copied)} figures")

    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(overleaf_dir / "main.tex", "main.tex")
        for name in copied:
            zf.write(figures_dst / name, f"figures/{name}")
    print(f"Wrote {args.zip_path}")
    print("Upload the ZIP to Overleaf (New Project -> Upload Project), set "
          "main.tex as the main file, and recompile.")


if __name__ == "__main__":
    main()
