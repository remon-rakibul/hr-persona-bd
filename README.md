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
