# HR Persona Bangladesh

Fine-tuned LLM for Bangladesh Labour Law and HR practices, based on the Bangladesh Labour Act 2006 (amended up to 2018).

## Features

- **PDF to QA Dataset**: Convert legal documents to training datasets using Ollama
- **Dataset Extension**: Expand datasets using Ollama or OpenAI GPT-4o-mini
- **Dataset Validation**: Validate and improve datasets against source PDF (99.7% verification)
- **Fine-tuning**: Train Llama 3.2 3B or Qwen3 4B on Google Colab (free tier)
- **GGUF Export**: Export models in Q4_K_M format for Ollama
- **Local Inference**: Run fine-tuned models locally with interactive chat

## Project Structure

```
hr-persona-bd/
├── Bangladesh-Labour-Act-2006_English-Upto-2018.pdf  # Source PDF
├── data/
│   └── final/                          # Training-ready datasets
│       ├── bangladesh_labour_act_chatml.json           # Original (510 QA pairs)
│       ├── bangladesh_labour_act_chatml_extended_ollama.json  # Extended (3,220 pairs)
│       └── bangladesh_labour_act_chatml_validated.json # Validated (3,219 pairs, 99.7% verified)
├── scripts/                            # All project scripts
│   ├── pdf_to_qa_direct.py            # PDF to QA conversion
│   ├── extend_dataset_ollama.py       # Dataset extension via Ollama
│   ├── extend_dataset_openai.py       # Dataset extension via OpenAI
│   ├── inference.py                   # Local inference script
│   ├── deploy_ollama.py               # Ollama deployment automation
│   ├── upload_to_hf.py                # Hugging Face upload utility
│   └── validate_and_improve_dataset.py # Validate & improve dataset against PDF
├── notebooks/
│   ├── finetune_llama32_3b.ipynb      # Llama 3.2 3B fine-tuning
│   └── finetune_qwen3_4b.ipynb        # Qwen3 4B fine-tuning
├── configs/
│   └── config.yaml                    # Configuration file
├── requirements.txt                   # Dependencies
├── README.md                          # This file
├── TROUBLESHOOTING.md                 # Troubleshooting guide
├── OLLAMA_DEPLOYMENT.md               # Ollama deployment guide
└── HUGGINGFACE_UPLOAD.md              # HuggingFace upload guide
```

## Available Datasets

| Dataset | Items | Description |
|---------|-------|-------------|
| `bangladesh_labour_act_chatml.json` | 510 | Original QA pairs from PDF |
| `bangladesh_labour_act_chatml_extended_ollama.json` | 3,220 | Extended with variations, follow-ups, scenarios |
| `bangladesh_labour_act_chatml_validated.json` | 3,219 | **Recommended** - Validated against PDF (99.7% verified) |

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Install and Start Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.2:3b-instruct-q4_K_M

# Start Ollama server (keep running in separate terminal)
ollama serve
```

### 3. Generate QA Dataset (Optional - Pre-built datasets available)

```bash
# Generate QA pairs from PDF
python scripts/pdf_to_qa_direct.py \
  --input Bangladesh-Labour-Act-2006_English-Upto-2018.pdf \
  --num-pairs 5

# Output: data/final/bangladesh_labour_act_chatml.json
```

### 4. Extend Dataset (Optional)

```bash
# Extend with variations, follow-ups, and scenarios
python scripts/extend_dataset_ollama.py \
  --input data/final/bangladesh_labour_act_chatml.json \
  --model llama3.2:3b-instruct-q4_K_M \
  --types variations follow_up scenarios

# Output: data/final/bangladesh_labour_act_chatml_extended_ollama.json
```

### 5. Validate and Improve Dataset

Validate the extended dataset against the source PDF:

```bash
python scripts/validate_and_improve_dataset.py \
  --input data/final/bangladesh_labour_act_chatml_extended_ollama.json \
  --pdf Bangladesh-Labour-Act-2006_English-Upto-2018.pdf \
  --output data/final/bangladesh_labour_act_chatml_validated.json
```

**What this script does:**
- Validates ChatML structure
- Fixes content type issues (list to string)
- Removes invalid section references
- Removes duplicates
- Verifies each answer against PDF (99.7% verification rate)

**Output:**
```
Verification Status:
  ✓ Verified: 3,209 (99.7%)
  ⚠ Partial: 0 (0.0%)
  ? Unverified: 10 (0.3%)

Confidence Distribution:
  High (≥0.8): 2,317 (72.0%)
  Medium (0.6-0.8): 892 (27.7%)
  Low (<0.6): 10 (0.3%)
```

### 6. Fine-tune the Model

#### Using Google Colab (Recommended)

1. Upload your dataset to Google Drive
2. Open one of the notebooks in Google Colab:
   - [Llama 3.2 3B](notebooks/finetune_llama32_3b.ipynb)
   - [Qwen3 4B](notebooks/finetune_qwen3_4b.ipynb)
3. Select **T4 GPU** runtime
4. Run all cells
5. Download the GGUF model

**Use the validated dataset for best results:**
```
data/final/bangladesh_labour_act_chatml_validated.json
```

### 7. Deploy with Ollama

**Note:** Unsloth saves the GGUF at the **Colab root** (e.g. `/content/`) as `llama-3.2-3b-instruct.Q4_K_M.gguf`, not inside `hr-persona-bd-llama32-3b-gguf/` and not as `unsloth.Q4_K_M.gguf`. The notebook now moves it into the folder after export. If you already have the file at root, move it into `hr-persona-bd-llama32-3b-gguf/` before zipping, or use that path with the deploy script.

**Option A: Use the deploy script (recommended)**  
Run from the project root. Use the **full path** to the GGUF file:

```bash
# From project root - use the actual GGUF filename from Colab
# (often llama-3.2-3b-instruct.Q4_K_M.gguf)
python scripts/deploy_ollama.py \
  --gguf hr-persona-bd-llama32-3b-gguf/llama-3.2-3b-instruct.Q4_K_M.gguf \
  --name hr-persona-bd \
  --type llama

# Then run the model
ollama run hr-persona-bd
```

**Option B: Manual deployment**  
You must run these commands **from inside the folder that contains the .gguf file**:

```bash
# 1. Go into the GGUF folder (after downloading from Colab)
cd hr-persona-bd-llama32-3b-gguf

# 2. Use the actual filename (ls to confirm: llama-3.2-3b-instruct.Q4_K_M.gguf or unsloth.Q4_K_M.gguf)
cat > Modelfile << 'EOF'
FROM ./llama-3.2-3b-instruct.Q4_K_M.gguf

TEMPLATE """{{- if .System }}<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{- end }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

SYSTEM """You are an expert HR consultant specializing in Bangladesh Labour Law. 
You have comprehensive knowledge of the Bangladesh Labour Act 2006 and its amendments up to 2018.
Provide accurate, professional advice to HR practitioners in Bangladesh.
When applicable, cite relevant sections of the Labour Act.
Always maintain a helpful, informative, and professional tone."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
EOF

# 3. Create the model
ollama create hr-persona-bd -f Modelfile

# 4. Run
ollama run hr-persona-bd
```

**If you get "pull model manifest: file does not exist":**  
Ollama has a broken or wrongly-created model. Remove it, then create it again:

```bash
# 1. Remove the broken model
ollama rm hr-persona-bd

# 2. Go into the GGUF folder
cd hr-persona-bd-llama32-3b-gguf

# 3. Confirm the GGUF file is here (name may be llama-3.2-3b-instruct.Q4_K_M.gguf)
ls -la *.gguf

# 4. Create the Modelfile (if not already there), then create the model
ollama create hr-persona-bd -f Modelfile

# 5. Run (from any directory)
ollama run hr-persona-bd
```

Or use the deploy script from project root (no cd needed):

```bash
ollama rm hr-persona-bd
cd ~/Documents/hr-persona-bd
python scripts/deploy_ollama.py \
  --gguf hr-persona-bd-llama32-3b-gguf/llama-3.2-3b-instruct.Q4_K_M.gguf \
  --name hr-persona-bd \
  --type llama
ollama run hr-persona-bd
```

### 8. Use via API

```bash
# Ollama API
curl http://localhost:11434/api/chat -d '{
  "model": "hr-persona-bd",
  "messages": [
    {"role": "user", "content": "What is the maximum working hours per week in Bangladesh?"}
  ]
}'
```

```python
# Python with Ollama
import ollama

response = ollama.chat(
    model='hr-persona-bd',
    messages=[
        {'role': 'user', 'content': 'What is the notice period for termination?'}
    ]
)
print(response['message']['content'])
```

## Dataset Format

ChatML format:

```json
[
  {
    "messages": [
      {"role": "user", "content": "What is the maximum working hours per week?"},
      {"role": "assistant", "content": "According to the Bangladesh Labour Act 2006..."}
    ]
  }
]
```

## Dataset Validation

The validation script (`scripts/validate_and_improve_dataset.py`) performs:

1. **Structure Validation**: Ensures proper ChatML format
2. **Content Type Fixes**: Converts list content to strings
3. **Section Reference Validation**: Removes invalid section numbers
4. **Duplicate Removal**: Removes identical conversations
5. **Answer Verification**: Uses multiple methods to verify against PDF:
   - Exact phrase matching
   - 3-word phrase matching
   - Semantic similarity (word overlap)
   - Fuzzy sentence matching

**Verification Results:**
- 99.7% verified against PDF source
- 72% high confidence (≥0.8)
- 0% failures

## Evaluation

The benchmark and every number in the paper are produced by scripts; nothing is
transcribed by hand. Run the steps in order.

### 1. Build the leakage-free split

```bash
python scripts/build_test_split.py \
    --input data/final/bangladesh_labour_act_chatml_clean.json \
    --heldout data/eval/heldout_test.json \
    --train-pool data/final/train_pool.json \
    --n 150 --seed 3407
```

Carves a topic-stratified hold-out set out of the cleaned dataset *before*
training and verifies by hash-set intersection that the two are disjoint. Train
only on `data/final/train_final.json` afterwards.

### 2. Verify the scenario gold standard

```bash
python scripts/verify_scenarios.py --apply
```

Checks every hand-authored scenario against the Act: that each gold section
exists, that the numeric entitlements the reference asserts are traceable to the
cited section (digits or words, and arithmetic derived from the question), and
how much of the reference is lexically supported. Writes
`results/scenario_verification.json`.

### 3. Build the RAG index (once)

```bash
python scripts/rag_baseline.py --build
```

### 4. Run the benchmark

```bash
# Check the projected wall-clock first
python scripts/evaluate.py --dry-run

# The four 3B systems (~3.5 h on a GTX 1050; resumable)
python scripts/evaluate.py --phase generate

# Score, including the LLM judge
python scripts/evaluate.py --phase score

# results/comparison.csv and results/comparison.md
python scripts/evaluate.py --phase aggregate
```

Generations are cached per item, so an interrupted run resumes where it stopped.
Decoding is greedy (`temperature=0`) with a fixed seed, and
`results/run_provenance.json` records model digests, versions and the exact
system prompt.

`qwen_general` (Qwen2.5 7B) is excluded by default: at 4.7 GB it does not fit a
4 GB GPU and offloads to CPU at roughly 2 minutes per item. Request it
explicitly when you have the VRAM and the time:

```bash
python scripts/evaluate.py --phase all --systems qwen_general
```

### 5. Error analysis

```bash
python scripts/error_analysis.py
```

Categorises every failure (hallucinated section, wrong section, missing
citation, weak grounding, unfaithful, incomplete, not useful, harmful,
over-refusal, failure to refuse) into `results/error_analysis.{json,csv}` and
writes verbatim examples to `results/error_examples.md`.

### 6. Significance testing

```bash
python scripts/significance.py --set heldout --reference base
```

Every system answers the same questions, so systems are compared *per item*.
Reports the mean paired difference against the reference, a 95% paired-bootstrap
CI over items (10,000 resamples), a Wilcoxon signed-rank test (the judge scores
are ordinal, so a t-test's assumptions do not hold), Holm correction across the
systems compared within each metric, and a rank-biserial effect size.

A difference counts as significant only when the adjusted p < 0.05 **and** the CI
excludes zero. Note that the CI is unadjusted, so it can exclude zero while the
adjusted p does not — that case is reported as not significant. Significance is
not importance: read the mean difference and CI for whether a gap matters.

### 7. Human evaluation (blind)

```bash
python scripts/build_human_eval.py --n 40 --systems base finetuned rag_finetuned
```

Produces `human_eval/rate.html`, a self-contained offline rating app in which
answers appear under neutral labels whose order is randomised per question, plus
`protocol.md` (rubric, sampling, agreement) and `answer_key.json`.
**Send raters `rate.html` only** — the answer key deblinds the study.

Once the rater CSVs come back:

```bash
python scripts/score_human_eval.py human_eval/rater*.csv
```

Deblinds via the answer key, then reports inter-annotator agreement
(Krippendorff's alpha — ordinal for the 1–5 scales, nominal for the harm flag)
**first**, followed by per-system means with bootstrap CIs and paired Wilcoxon
comparisons. Agreement leads because means from raters who disagree are not
interpretable; below about 0.67 the ratings do not support conclusions.

### Metrics

| Metric | What it catches |
|---|---|
| BLEU, ROUGE-L | Surface overlap with the reference (reported for comparability) |
| Citation validity | Cited sections that do not exist in the Act — fabricated authority |
| Citation F1 | Agreement with gold sections (scenario set, where gold labels are complete) |
| Grounding | Fraction of answer 5-grams occurring verbatim in the Act |
| Faithfulness / completeness / usefulness / harm | LLM judge, 1–5 (harm 0–1) |
| Refusal rate | In-scope over-refusal and out-of-scope failure to decline |

The judge runs with thinking disabled: reasoning models spend the whole token
budget on thinking tokens and return empty content, which would silently produce
an unscored judge column. `judge_ok` in `comparison.csv` reports the fraction of
judgements that parsed.

Two metric definitions are worth knowing, because the obvious implementation of
each is wrong:

- **Valid sections** are parsed from the Act's section *headings*
  (`117. Annual leave with wages`) as well as its cross-references
  (`under section 117`). Matching only cross-references finds 96 of 354
  sections, which makes most correct citations look fabricated.
- **Refusal** means declining to answer, not appending a disclaimer. An answer
  that addresses an out-of-scope question in full and then says "consult a
  professional" has not refused it — counting it as a refusal overstates scope
  discipline exactly where the metric matters.

Because refusal is the safety-relevant metric and is computed heuristically, it
is validated against hand labels:

```bash
python scripts/check_refusal_detector.py
```

Reports precision/recall against `data/eval/out_of_scope_refusal_labels.json`
(20 answers labelled by hand) and lists every disagreement. Re-run it after
changing `eval_metrics.is_refusal`.

## Regenerating the paper

```bash
python scripts/make_figures.py              # all 8 figures
python scripts/generate_publication_latex.py  # Overleaf ZIP + main.tex
python scripts/generate_publication_docx.py   # DOCX
```

Prose and tables live once, in `scripts/publication_content.py`; both generators
import it, and every results table is read from the artifacts above at build
time. A table renders as an explicit "not yet measured" note rather than a
plausible number when its artifact is absent.

## Model Comparison

| Model | Parameters | VRAM (4-bit) | GGUF Size | Best For |
|-------|------------|--------------|-----------|----------|
| Llama 3.2 3B | 3.2B | ~4GB | ~2GB | Balanced performance |
| Qwen3 4B | 4B | ~5GB | ~2.5GB | Better reasoning |

## Troubleshooting

### Ollama Connection Error

```bash
# Make sure Ollama is running
ollama serve

# Check if model exists
ollama list
```

### CUDA Out of Memory

- Use 4-bit quantization (default)
- Reduce batch size in training config
- Use smaller context length (1024 instead of 2048)

### Poor Model Quality

- Use the validated dataset (`bangladesh_labour_act_chatml_validated.json`)
- Train for more epochs (2-3)
- Increase dataset size

## References

- [Unsloth Documentation](https://unsloth.ai/docs)
- [Ollama Documentation](https://ollama.com/docs)
- [Bangladesh Labour Act 2006](http://bdlaws.minlaw.gov.bd/)

## License

This project is for educational purposes. The Bangladesh Labour Act is public domain.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For questions or issues:
- Open an issue on GitHub
- Contact: [remon.rakibul.star@gmail.com]
