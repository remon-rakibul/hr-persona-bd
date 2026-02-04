#!/usr/bin/env python3
"""
Generate submission-ready DOCX for Q1 publication:
"Fine-Tuning Large Language Models for Bangladesh Labour Law: An HR-Oriented Legal QA System"
First author: Md. Rakibul Haque.

Creates the full document with sections, tables, and five figures (3 pipeline diagrams +
2 training metrics figures). Training curves are plotted from trainer_state.json if provided,
otherwise placeholder curves or placeholders are used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Add project root for imports if needed
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
except ImportError:
    print("Install python-docx: pip install python-docx", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None

# Default paths
DEFAULT_FIG_DIR = PROJECT_ROOT / "publication" / "figures"
DEFAULT_OUTPUT_DOCX = PROJECT_ROOT / "publication" / "Haque_etal_LLM_Bangladesh_Labour_Law.docx"
DPI = 300  # Publication quality


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _draw_flowchart_boxes(ax, boxes, box_width=0.72, box_height=0.10, gap=0.14, top=1.0):
    """Draw a vertical flowchart: centered boxes with consistent alignment and arrows."""
    x_center = 0.5
    x_left = x_center - box_width / 2
    for i, label in enumerate(boxes):
        y_center = top - i * (box_height + gap)
        y_bottom = y_center - box_height / 2
        box = mpatches.FancyBboxPatch(
            (x_left, y_bottom), box_width, box_height,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            facecolor="#E8F4FC", edgecolor="#2E86AB", linewidth=1.2
        )
        ax.add_patch(box)
        ax.text(x_center, y_center, label, ha="center", va="center", fontsize=8,
                wrap=True, ma="center")
        if i < len(boxes) - 1:
            arrow_start = y_bottom
            arrow_end = y_center - (box_height + gap) + box_height / 2
            ax.annotate("", xy=(x_center, arrow_end), xytext=(x_center, arrow_start),
                        arrowprops=dict(arrowstyle="->", color="#2E86AB", lw=2))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, top + 0.05)
    ax.axis("off")


def create_figure1_dataset_pipeline(out_path: Path) -> Path:
    """Figure 1: Dataset creation pipeline flowchart."""
    if plt is None:
        (out_path.parent).mkdir(parents=True, exist_ok=True)
        (out_path.parent / (out_path.stem + "_PLACEHOLDER.txt")).write_text(
            "Figure 1: Dataset pipeline – Source PDF → Text extraction → Chunking (4000, 200 overlap) "
            "→ QA generation → Extension → Validation → Final ChatML dataset. Generate with draw.io or matplotlib."
        )
        return out_path
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 6.5))
    boxes = [
        "Source PDF",
        "Text extraction",
        "Chunking (4000 chars, 200 overlap)",
        "QA generation (per chunk)",
        "Extension (variations, follow-up, scenarios)",
        "Validation (structure, section refs, PDF verification)",
        "Final ChatML dataset",
    ]
    _draw_flowchart_boxes(ax, boxes, box_width=0.74, box_height=0.11, gap=0.13, top=0.98)
    ax.set_title("Figure 1: Dataset creation pipeline", fontsize=11, pad=12)
    fig.tight_layout(pad=1.2)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def create_figure2_finetuning_pipeline(out_path: Path) -> Path:
    """Figure 2: Fine-tuning and deployment pipeline."""
    if plt is None:
        (out_path.parent).mkdir(parents=True, exist_ok=True)
        (out_path.parent / (out_path.stem + "_PLACEHOLDER.txt")).write_text(
            "Figure 2: Fine-tuning pipeline – Llama 3.2 3B → 4-bit + LoRA → SFT → Save LoRA → Merge (optional) "
            "→ Export GGUF Q4_K_M → Ollama Modelfile → Local deployment."
        )
        return out_path
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 6.2))
    boxes = [
        "Llama 3.2 3B Instruct",
        "Load in 4-bit + LoRA config",
        "SFT on validated dataset (train/val split)",
        "Save LoRA",
        "Merge (optional)",
        "Export GGUF Q4_K_M",
        "Ollama Modelfile",
        "Local deployment",
    ]
    _draw_flowchart_boxes(ax, boxes, box_width=0.74, box_height=0.10, gap=0.12, top=0.98)
    ax.set_title("Figure 2: Fine-tuning and deployment pipeline", fontsize=11, pad=12)
    fig.tight_layout(pad=1.2)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def create_figure3_evaluation_design(out_path: Path) -> Path:
    """Figure 3: Comparative evaluation design — horizontal flow, text contained and centered."""
    if plt is None:
        (out_path.parent).mkdir(parents=True, exist_ok=True)
        (out_path.parent / (out_path.stem + "_PLACEHOLDER.txt")).write_text(
            "Figure 3: Hold-out test set → same questions → [Fine-tuned | Base Llama | ChatGPT | Claude] "
            "→ metrics (BLEU, ROUGE, EM / human) → comparison table."
        )
        return out_path
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    # Boxes with clear gaps so arrows sit only in gap (no overlap with text)
    y_center = 1.5
    box_h = 1.05
    y_bottom = y_center - box_h / 2
    gap = 0.4  # space between boxes for arrows only
    # (x_start, width, label_lines, facecolor) — label as list of lines for explicit wrap
    box_specs = [
        (0.05, 2.35, ["Hold-out test set", "(Bangladesh Labour Act QA)"], "#E8F4FC"),
        (2.35 + gap, 1.15, ["Same questions"], "#E8F4FC"),
        (2.35 + gap + 1.15 + gap, 2.2, ["Fine-tuned | Base Llama", "ChatGPT | Claude"], "#FFF4E6"),
        (2.35 + gap + 1.15 + gap + 2.2 + gap, 2.0, ["Metrics: BLEU, ROUGE, EM", "Comparison table"], "#E8F4FC"),
    ]
    fontsize = 7.5
    for x, w, lines, color in box_specs:
        rect = mpatches.FancyBboxPatch(
            (x, y_bottom), w, box_h,
            boxstyle="round,pad=0.03,rounding_size=0.05",
            facecolor=color, edgecolor="#2E86AB", linewidth=1.2
        )
        ax.add_patch(rect)
        label = "\n".join(lines)
        ax.text(x + w / 2, y_center, label, ha="center", va="center", fontsize=fontsize,
                ma="center")
    # Arrow endpoints: end of box i + small inset, start of box i+1 - small inset (arrows in gap only)
    arrow_inset = 0.06
    arrow_y = y_center
    box_ends = [0.05 + 2.35, 2.35 + gap + 1.15, 2.35 + gap + 1.15 + gap + 2.2]
    box_starts = [2.35 + gap, 2.35 + gap + 1.15 + gap, 2.35 + gap + 1.15 + gap + 2.2 + gap]
    for x_end, x_start in zip(box_ends, box_starts):
        ax.annotate("", xy=(x_start + arrow_inset, arrow_y), xytext=(x_end - arrow_inset, arrow_y),
                    arrowprops=dict(arrowstyle="->", color="#2E86AB", lw=2))
    ax.set_title("Figure 3: Comparative evaluation design", fontsize=11, pad=10)
    fig.tight_layout(pad=1.2)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def load_trainer_state(path: Path) -> list[dict] | None:
    """Load Hugging Face trainer_state.json; return log_history or None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("log_history") or data.get("state", {}).get("log_history")
    except Exception:
        return None


def create_figure4_training_loss(out_path: Path, trainer_state_path: Path | None = None) -> Path:
    """Figure 4: Training loss curve. Uses trainer_state.json if provided."""
    if plt is None:
        _ensure_dir(out_path.parent)
        (out_path.parent / (out_path.stem + "_PLACEHOLDER.txt")).write_text(
            "Figure 4: Training loss vs step. Add PNG from TensorBoard or from trainer_state.json (see script)."
        )
        return out_path
    steps, loss_values = [], []
    if trainer_state_path and trainer_state_path.exists():
        history = load_trainer_state(trainer_state_path)
        if history:
            for entry in history:
                if "loss" in entry:
                    steps.append(entry.get("step", len(steps)))
                    loss_values.append(entry["loss"])
    if not steps:
        # From Colab run: Step 25/50/75/100 and training loss
        steps = [25, 50, 75, 100]
        loss_values = [1.8568, 1.2597, 1.0083, 0.8798]
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    ax.plot(steps, loss_values, "b-", linewidth=2, label="Training loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Figure 4: Training loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def create_figure5_eval_loss_perplexity(out_path: Path, trainer_state_path: Path | None = None) -> Path:
    """Figure 5: Evaluation loss and perplexity over training."""
    if plt is None:
        _ensure_dir(out_path.parent)
        (out_path.parent / (out_path.stem + "_PLACEHOLDER.txt")).write_text(
            "Figure 5: Eval loss and perplexity vs step. Add PNG from TensorBoard or trainer_state.json."
        )
        return out_path
    steps, eval_loss, perplexity = [], [], []
    if trainer_state_path and trainer_state_path.exists():
        history = load_trainer_state(trainer_state_path)
        if history:
            for entry in history:
                if "eval_loss" in entry:
                    step = entry.get("step", len(steps))
                    steps.append(step)
                    el = entry["eval_loss"]
                    eval_loss.append(el)
                    perplexity.append(math.exp(el))
    if not steps:
        # From Colab run: eval loss at steps 25, 50, 75, 100
        steps = [25, 50, 75, 100]
        eval_loss = [1.3950, 1.1217, 0.9772, 0.9522]
        perplexity = [math.exp(x) for x in eval_loss]
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax1.set_xlabel("Step")
    ax1.plot(steps, eval_loss, "b-", linewidth=2, label="Eval loss")
    ax1.set_ylabel("Eval loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(steps, perplexity, "g--", linewidth=2, label="Perplexity")
    ax2.set_ylabel("Perplexity", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    ax1.set_title("Figure 5: Evaluation loss and perplexity")
    ax1.legend(loc="upper right")
    ax2.legend(loc="center right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def get_training_metrics_table_data(trainer_state_path: Path | None) -> list[list[str]]:
    """Build rows for training metrics summary table from trainer_state.json or placeholders."""
    headers = ["Metric", "Value"]
    rows = [headers]
    if trainer_state_path and trainer_state_path.exists():
        try:
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            hist = data.get("log_history") or []
            train_losses = [e["loss"] for e in hist if "loss" in e]
            eval_entries = [e for e in hist if "eval_loss" in e]
            if train_losses:
                rows.append(["Final training loss", f"{train_losses[-1]:.4f}"])
            if eval_entries:
                last_eval = eval_entries[-1]
                rows.append(["Final eval loss", f"{last_eval['eval_loss']:.4f}"])
                rows.append(["Final perplexity", f"{math.exp(last_eval['eval_loss']):.2f}"])
                rows.append(["Best checkpoint (step)", str(last_eval.get("step", "—"))])
            if "train_runtime" in data:
                rows.append(["Total training time (s)", f"{data['train_runtime']:.1f}"])
        except Exception:
            pass
    if len(rows) == 1:
        # Default from Colab run (Llama 3.2 3B, 100 steps, Tesla T4)
        rows.extend([
            ["Final training loss", "1.3637"],
            ["Final eval loss", "0.9522"],
            ["Final perplexity", "2.59"],
            ["Best checkpoint (step)", "100"],
            ["Total training time (s)", "465.9"],
            ["GPU", "Tesla T4 (peak 5.33 GB / 14.74 GB)"],
        ])
    return rows


def add_paragraph(doc: Document, text: str, style: str = "Normal"):
    p = doc.add_paragraph(text, style=style)
    return p


def add_figure_with_caption(doc: Document, image_path: Path, caption: str, width_inches: float = 5.5):
    if image_path.exists() and image_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        doc.add_picture(str(image_path), width=Inches(width_inches))
    else:
        doc.add_paragraph(f"[{caption}] — Placeholder: add image at {image_path}", style="Intense Quote")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(caption)
    run.italic = True
    run.font.size = Pt(10)


def build_document(
    output_docx: Path,
    figures_dir: Path,
    trainer_state_path: Path | None = None,
) -> None:
    doc = Document()
    # Title
    doc.add_heading("Fine-Tuning Large Language Models for Bangladesh Labour Law: An HR-Oriented Legal QA System", 0)
    doc.add_paragraph()
    doc.add_paragraph("Md. Rakibul Haque (first author)")
    doc.add_paragraph("[Co-authors and affiliations to be added]")
    doc.add_paragraph()

    # Abstract
    doc.add_heading("Abstract", level=1)
    abstract = (
        "Human resource professionals in Bangladesh require accurate, readily accessible answers to questions "
        "about the Bangladesh Labour Act 2006 (amended to 2018). General-purpose large language models (LLMs) "
        "may lack up-to-date domain knowledge and raise privacy concerns when used with sensitive HR data. "
        "We present an end-to-end pipeline to create a curated, validated question-answering dataset from the "
        "official PDF of the Act, and we fine-tune Llama 3.2 3B Instruct using low-rank adaptation (LoRA) with "
        "Unsloth for efficient training. The resulting model is exported to GGUF (Q4_K_M) for local deployment "
        "via Ollama, providing an HR consultant–style assistant. Our dataset pipeline includes PDF chunking, "
        "LLM-based QA generation, extension with variations and scenarios, and validation against the source "
        "document (99.7% verification rate). We describe the training setup, report training and evaluation "
        "metrics, and provide a comparative evaluation framework against base Llama 3.2 3B and commercial APIs "
        "(e.g., ChatGPT, Claude). The contribution is a reproducible, privacy-preserving, domain-specific legal "
        "QA system suitable for local deployment and future extension to additional laws and Bengali."
    )
    add_paragraph(doc, abstract)
    doc.add_paragraph()

    # 1. Introduction
    doc.add_heading("1. Introduction", level=1)
    add_paragraph(doc,
        "Accurate and timely answers to questions about labour law are essential for HR practitioners in Bangladesh. "
        "The Bangladesh Labour Act 2006, as amended up to 2018, is the primary source, but navigating it manually "
        "is time-consuming. General-purpose LLMs can assist but may lack precise knowledge of this jurisdiction and "
        "raise concerns about data privacy when used with confidential HR information. There is a gap in domain-specific "
        "LLMs tailored to Bangladesh labour law that can be deployed locally.")
    add_paragraph(doc,
        "This work contributes: (1) a curated, PDF-verified QA dataset derived from the Act (510 → 3,220 → 3,219 "
        "pairs after validation); (2) a reproducible dataset-creation and validation pipeline; (3) a fine-tuned "
        "Llama 3.2 3B model using LoRA and Unsloth; and (4) a comparative evaluation framework with baselines "
        "and metrics so that results can be filled after running the evaluation protocol.")
    doc.add_paragraph()

    # 2. Related Work
    doc.add_heading("2. Related Work", level=1)
    add_paragraph(doc,
        "Legal NLP and legal QA have been studied for contract analysis, legal search, and compliance. Domain adaptation "
        "and fine-tuning of LLMs for specialized corpora are well established; LoRA and related parameter-efficient "
        "methods enable training on consumer hardware. Low-resource and local deployment scenarios favour small models "
        "and quantized formats such as GGUF. We are not aware of prior work targeting Bangladesh Labour Act–specific "
        "QA with a fine-tuned, locally deployable LLM; this work fills that gap.")
    doc.add_paragraph()

    # 3. Methodology
    doc.add_heading("3. Methodology", level=1)
    add_paragraph(doc, "The end-to-end pipeline comprises dataset creation, model and training configuration, and deployment. Figures 1 and 2 illustrate the main workflows.")
    doc.add_paragraph()

    doc.add_heading("3.1 Dataset creation", level=2)
    add_paragraph(doc,
        "The source document is the Bangladesh Labour Act 2006 (English), amended up to 2018. Text is extracted from "
        "the PDF and split into chunks (e.g., 4,000 characters with 200-character overlap). For each chunk, question–answer "
        "pairs are generated using an LLM (Ollama). The dataset is extended with variations, follow-up questions, and "
        "scenarios to increase coverage. A validation step checks structure, section references, and consistency with the "
        "source PDF. The final ChatML-formatted dataset grows from 510 initial pairs to 3,220 after extension and 3,219 "
        "after validation, with a 99.7% verification rate against the PDF. Scripts: pdf_to_qa_direct.py, "
        "extend_dataset_ollama.py, validate_and_improve_dataset.py.")
    doc.add_paragraph()
    add_figure_with_caption(doc, figures_dir / "figure1_dataset_pipeline.png", "Figure 1: Dataset creation pipeline.")
    doc.add_paragraph()

    doc.add_heading("3.2 Model and training", level=2)
    add_paragraph(doc,
        "Base model: Llama 3.2 3B Instruct. We use Unsloth with 4-bit quantization. LoRA is applied with r=16, alpha=16, "
        "target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj. Max sequence length is 2,048. "
        "Training uses per_device_train_batch_size=2, gradient_accumulation_steps=4 (effective batch size 8), "
        "learning_rate=2e-4, optimizer adamw_8bit, and SFTTrainer. A 90/10 train/validation split is used; evaluation "
        "runs every 25 steps; the best model is selected by eval_loss. For a quick run, max_steps=100 can be used; "
        "for full training, num_train_epochs=1. The model is exported to GGUF Q4_K_M for Ollama. All details are from "
        "notebooks/finetune_llama32_3b.ipynb.")
    add_paragraph(doc,
        "Unsloth patches 28 layers (28 QKV, 28 O, 28 MLP). Trainable parameters: 24,313,856 of 3,237,063,680 (0.75%). "
        "The dataset is formatted with the llama-3.2 chat template; we use 3,219 conversations, split into 2,897 training "
        "and 322 validation samples. Training was run on a single Tesla T4 GPU (peak memory 5.33 GB of 14.74 GB).")
    doc.add_paragraph()
    add_figure_with_caption(doc, figures_dir / "figure2_finetuning_pipeline.png", "Figure 2: Fine-tuning and deployment pipeline.")
    doc.add_paragraph()

    doc.add_heading("3.3 Deployment", level=2)
    add_paragraph(doc,
        "The fine-tuned model is deployed locally via Ollama using a Modelfile and a system prompt that defines an "
        "HR consultant persona for Bangladesh Labour Act questions.")
    doc.add_paragraph()

    # 4. Experimental setup and comparative evaluation
    doc.add_heading("4. Experimental setup and comparative evaluation", level=1)
    add_paragraph(doc, "4.1 Baselines: Base Llama 3.2 3B (no fine-tuning), ChatGPT (e.g. GPT-4o-mini), Claude (e.g. Claude 3 Haiku), and optionally one other (e.g. Gemini).")
    add_paragraph(doc,
        "4.2 Evaluation protocol: A hold-out test set of 100–200 QA pairs (not used in training) is drawn from the "
        "validated dataset. The same questions are posed to all models with a fixed system prompt (e.g. HR consultant for "
        "Bangladesh Labour Act). Metrics include at least one automatic metric (e.g. BLEU or ROUGE-L) and, if feasible, "
        "human or expert ratings (e.g. 1–5 for accuracy/relevance).")
    add_paragraph(doc, "4.3 Comparison table (to be filled from evaluation):")
    # Comparison table
    table = doc.add_table(rows=5, cols=4)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    hdr[0].text = "Model"
    hdr[1].text = "BLEU / ROUGE"
    hdr[2].text = "Exact match / Human"
    hdr[3].text = "Notes"
    for i, name in enumerate(["Fine-tuned Llama 3.2 3B", "Base Llama 3.2 3B", "ChatGPT (e.g. GPT-4o-mini)", "Claude (e.g. Claude 3 Haiku)"], start=1):
        row = table.rows[i].cells
        row[0].text = name
        row[1].text = "[To be filled from evaluation]"
        row[2].text = "[To be filled from evaluation]"
        row[3].text = ""
    doc.add_paragraph()
    add_figure_with_caption(doc, figures_dir / "figure3_evaluation_design.png", "Figure 3: Comparative evaluation design.")
    doc.add_paragraph()
    add_paragraph(doc,
        "4.4 Rationale: Domain fine-tuning on a validated, jurisdiction-specific dataset can outperform general-purpose "
        "LLMs on accuracy and citation correctness for the Bangladesh Labour Act, while enabling local deployment and "
        "privacy. Actual comparison numbers will be added once the evaluation protocol is run.")
    doc.add_paragraph()

    # 5. Results
    doc.add_heading("5. Results", level=1)
    doc.add_heading("5.1 Training metrics", level=2)
    add_paragraph(doc,
        "Figure 4 shows the training loss curve over steps. Figure 5 shows evaluation loss and perplexity over training, "
        "illustrating convergence and the absence of severe overfitting. The following table summarizes key training metrics "
        "(sourced from notebook outputs, trainer_state.json, or TensorBoard logs).")
    doc.add_paragraph()
    add_figure_with_caption(doc, figures_dir / "figure4_training_loss.png", "Figure 4: Training loss curve.")
    doc.add_paragraph()
    add_figure_with_caption(doc, figures_dir / "figure5_eval_loss_perplexity.png", "Figure 5: Evaluation loss and perplexity over training.")
    doc.add_paragraph()
    add_paragraph(doc, "Training metrics summary (from Colab run):")
    metrics_rows = get_training_metrics_table_data(trainer_state_path)
    tbl = doc.add_table(rows=len(metrics_rows), cols=2)
    tbl.style = "Table Grid"
    for i, row_data in enumerate(metrics_rows):
        tbl.rows[i].cells[0].text = row_data[0]
        tbl.rows[i].cells[1].text = row_data[1]
    doc.add_paragraph()
    add_paragraph(doc, "Loss at evaluation steps (from Colab run):")
    step_table = doc.add_table(rows=5, cols=3)
    step_table.style = "Table Grid"
    step_table.rows[0].cells[0].text = "Step"
    step_table.rows[0].cells[1].text = "Training loss"
    step_table.rows[0].cells[2].text = "Validation loss"
    for i, (step, tl, vl) in enumerate([(25, 1.8568, 1.3950), (50, 1.2597, 1.1217), (75, 1.0083, 0.9772), (100, 0.8798, 0.9522)], start=1):
        step_table.rows[i].cells[0].text = str(step)
        step_table.rows[i].cells[1].text = f"{tl:.4f}"
        step_table.rows[i].cells[2].text = f"{vl:.4f}"
    doc.add_paragraph()
    add_paragraph(doc,
        "Final training loss: 1.3637. Eval loss: 0.9522; eval perplexity: 2.59 (excellent, <10). "
        "Training time: 465.9 s; eval runtime 38.9 s (8.28 samples/s).")
    doc.add_paragraph()

    doc.add_heading("5.2 Comparative results", level=2)
    add_paragraph(doc,
        "Comparative results against the baselines listed in Section 4 will be filled after running the evaluation "
        "protocol (hold-out set, same questions, BLEU/ROUGE and/or human ratings). See the comparison table in Section 4.3.")
    doc.add_paragraph()

    # 6. Discussion
    doc.add_heading("6. Discussion", level=1)
    add_paragraph(doc,
        "Strengths include a reproducible pipeline from PDF to validated dataset and fine-tuned model, a small deployable "
        "model suitable for local use, and a clear evaluation framework. Limitations: the scope is limited to one act and "
        "English; the system is not a substitute for legal advice. Ethical use and disclaimers should be stated when "
        "deploying the model.")
    doc.add_paragraph()

    # 7. Conclusion
    doc.add_heading("7. Conclusion", level=1)
    add_paragraph(doc,
        "We presented a pipeline for building a domain-specific legal QA system for the Bangladesh Labour Act: curated "
        "and validated dataset creation, LoRA fine-tuning of Llama 3.2 3B with Unsloth, and deployment via Ollama. "
        "Training and evaluation metrics were reported; comparative results can be added after running the evaluation "
        "protocol. Future work may include additional laws, Bengali language support, and human expert evaluation.")
    doc.add_paragraph()

    # References
    doc.add_heading("References", level=1)
    refs = [
        "[1] Legal NLP / legal QA literature (to be completed).",
        "[2] LoRA: Hu et al.; Unsloth (to be completed).",
        "[3] Bangladesh Labour Act 2006 (amended to 2018).",
        "[4] BLEU/ROUGE and evaluation metrics (to be completed).",
    ]
    for r in refs:
        add_paragraph(doc, r)
    doc.add_paragraph()

    # Appendices (optional)
    doc.add_heading("Appendices", level=1)
    add_paragraph(doc, "Dataset statistics: 510 initial QA pairs → 3,220 after extension → 3,219 after validation (99.7% PDF-verified).")
    add_paragraph(doc, "Example prompts and outputs, and links to code/dataset availability, can be added here.")
    doc.add_paragraph()
    add_paragraph(doc, "Data and code availability: Dataset and code will be made available at [URL to be added].")

    _ensure_dir(output_docx.parent)
    doc.save(str(output_docx))
    print(f"Saved: {output_docx}")


def main():
    parser = argparse.ArgumentParser(description="Generate Q1 publication DOCX for Bangladesh Labour Law LLM.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DOCX, help="Output DOCX path")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIG_DIR, help="Directory for figure PNGs")
    parser.add_argument("--trainer-state", type=Path, default=None, help="Path to trainer_state.json for training curves")
    args = parser.parse_args()

    # Resolve trainer_state: optional path or look in outputs/
    trainer_state = args.trainer_state
    if trainer_state is None and (PROJECT_ROOT / "outputs" / "trainer_state.json").exists():
        trainer_state = PROJECT_ROOT / "outputs" / "trainer_state.json"

    fig_dir = args.figures_dir
    _ensure_dir(fig_dir)

    # Generate all five figures
    create_figure1_dataset_pipeline(fig_dir / "figure1_dataset_pipeline.png")
    create_figure2_finetuning_pipeline(fig_dir / "figure2_finetuning_pipeline.png")
    create_figure3_evaluation_design(fig_dir / "figure3_evaluation_design.png")
    create_figure4_training_loss(fig_dir / "figure4_training_loss.png", trainer_state)
    create_figure5_eval_loss_perplexity(fig_dir / "figure5_eval_loss_perplexity.png", trainer_state)

    build_document(args.output, fig_dir, trainer_state)
    print("Done. Next: add co-authors/affiliations, fill comparison table from evaluation, polish references.")


if __name__ == "__main__":
    main()
