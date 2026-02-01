# HR Persona Bangladesh

Fine-tuned LLM for Bangladesh Labour Law and HR practices, based on the Bangladesh Labour Act 2006 (amended up to 2018).

## Features

- **PDF to QA Dataset**: Convert legal documents to training datasets using Meta's Synthetic Data Kit
- **Dataset Extension**: Expand datasets using Ollama or OpenAI GPT-4o-mini
- **Fine-tuning**: Train Llama 3.2 3B or Qwen3 4B on Google Colab (free tier)
- **GGUF Export**: Export models in Q4_K_M format for Ollama
- **Local Inference**: Run fine-tuned models locally with interactive chat

## Project Structure

```
hr-persona-bd/
├── Bangladesh-Labour-Act-2006_English-Upto-2018.pdf  # Source PDF (at root)
├── data/
│   ├── output/                   # Extracted text (if using synthetic-data-kit)
│   ├── generated/                # Initial QA pairs
│   ├── curated/                  # Quality-filtered QA pairs
│   └── final/                    # Training-ready datasets
├── scripts/
│   ├── pdf_to_qa_direct.py       # PDF to QA conversion (recommended)
│   ├── extend_dataset_ollama.py  # Dataset extension via Ollama
│   ├── extend_dataset_openai.py  # Dataset extension via OpenAI
│   ├── inference.py              # Local inference script
│   ├── deploy_ollama.py          # Ollama deployment automation
│   └── upload_to_hf.py           # Hugging Face upload utility
├── notebooks/
│   ├── finetune_llama32_3b.ipynb # Llama 3.2 3B fine-tuning
│   └── finetune_qwen3_4b.ipynb   # Qwen3 4B fine-tuning
├── configs/
│   └── config.yaml               # Configuration file
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── TROUBLESHOOTING.md            # Troubleshooting guide
├── OLLAMA_DEPLOYMENT.md          # Ollama deployment guide
├── HUGGINGFACE_UPLOAD.md         # HuggingFace upload guide
└── .gitignore                    # Git ignore rules
```

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

# Start Ollama server (keep this running in a separate terminal)
ollama serve
```

### 3. Generate QA Dataset

```bash
# RECOMMENDED: Start with a test run (20 chunks, ~100 QA pairs, ~30 minutes)
python scripts/pdf_to_qa_direct.py \
  --input Bangladesh-Labour-Act-2006_English-Upto-2018.pdf \
  --num-pairs 5 \
  --max-chunks 20

# FULL DATASET: Generate complete dataset (all ~115 chunks, ~575 QA pairs, ~2-3 hours)
python scripts/pdf_to_qa_direct.py \
  --input Bangladesh-Labour-Act-2006_English-Upto-2018.pdf \
  --num-pairs 5

# OR: Run full dataset in background (recommended for long generation)
nohup python scripts/pdf_to_qa_direct.py \
  --input Bangladesh-Labour-Act-2006_English-Upto-2018.pdf \
  --num-pairs 5 \
  > qa_generation.log 2>&1 &

# Check background progress:
tail -f qa_generation.log
```

The dataset will be saved to `data/final/bangladesh_labour_act_chatml.json`

**Performance Notes:**
- The script automatically extracts text from PDF
- Generation time: ~60-80 seconds per chunk (on GTX 1050 4GB)
- Test run (20 chunks): ~25-30 minutes → ~100 QA pairs
- Full dataset (115 chunks): ~2-3 hours → ~575 QA pairs
- Recommended: Start with test run, then run full dataset overnight

**IMPORTANT: Start Ollama before running the script!**

```bash
# Step 1: Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Step 2: Pull a model for QA generation
ollama pull llama3.2:3b-instruct-q4_K_M

# Step 3: Start Ollama server (in a separate terminal or background)
ollama serve

# Step 4: Verify Ollama is running
curl http://localhost:11434/api/tags
```

#### Option A: Direct QA Generation with Ollama (Recommended - Fast & Reliable)

This method directly uses Ollama without intermediate tools. It's simpler and more reliable.

```bash
# Generate a test dataset (20 chunks, ~100 QA pairs, ~25-30 minutes)
python scripts/pdf_to_qa_direct.py \
  --input data/parsed/Bangladesh-Labour-Act-2006_English-Upto-2018.txt \
  --num-pairs 5 \
  --max-chunks 20

# Generate full dataset (all chunks, ~575 QA pairs, ~2-3 hours)
python scripts/pdf_to_qa_direct.py \
  --input data/parsed/Bangladesh-Labour-Act-2006_English-Upto-2018.txt \
  --num-pairs 5

# Or run it in background (overnight)
nohup python scripts/pdf_to_qa_direct.py \
  --input data/parsed/Bangladesh-Labour-Act-2006_English-Upto-2018.txt \
  --num-pairs 5 \
  > qa_generation.log 2>&1 &

# Check progress
tail -f qa_generation.log
```

**Note**: The script automatically extracts text from PDF files. It works with:
- PDF files (`.pdf`) - Automatically extracts text
- Text files (`.txt`) - Reads directly

The script:
1. Extracts/reads text from input file
2. Splits text into chunks
3. Generates QA pairs using Ollama
4. Saves in ChatML format to `data/final/bangladesh_labour_act_chatml.json`

**Performance Tips:**
- Use `--num-pairs 5` for faster generation with quality pairs
- Use `--max-chunks 20` to test with a smaller dataset first
- Generation time: ~60-80 seconds per chunk on GTX 1050 4GB
- Full dataset (~115 chunks) takes ~2-3 hours

#### Option B: Using Synthetic Data Kit (Advanced - May Have Issues)

If you prefer the full synthetic-data-kit pipeline:

```bash
# Parse PDF
synthetic-data-kit -c configs/config.yaml ingest Bangladesh-Labour-Act-2006_English-Upto-2018.pdf

# Generate QA pairs
synthetic-data-kit -c configs/config.yaml create data/output/Bangladesh-Labour-Act-2006_English-Upto-2018.txt \
  --type qa --num-pairs 50 --chunk-size 4000

# Curate with quality threshold
synthetic-data-kit -c configs/config.yaml curate data/generated/*.json --threshold 7.0

# Save as ChatML format
synthetic-data-kit -c configs/config.yaml save-as data/curated/*.json --format chatml --storage hf
```

**Note**: synthetic-data-kit may have configuration issues. Use Option A for reliability.

### 4. Extend Dataset (Optional)

Expand the dataset with additional QA pairs:

#### Using Ollama (Free, Local)

```bash
python scripts/extend_dataset_ollama.py \
  --input data/final/bangladesh_labour_act_chatml.json \
  --model llama3.2:3b-instruct-q4_K_M \
  --types variations follow_up scenarios
```

#### Using OpenAI (Paid, Higher Quality)

```bash
export OPENAI_API_KEY="your-api-key"
python scripts/extend_dataset_openai.py \
  --input data/final/bangladesh_labour_act_chatml.json \
  --model gpt-4o-mini \
  --types variations follow_up scenarios edge_cases
```

### 5. Fine-tune the Model

#### Using Google Colab (Recommended)

1. Upload your dataset to Google Drive or prepare to upload during training
2. Open one of the notebooks in Google Colab:
   - [Llama 3.2 3B](notebooks/finetune_llama32_3b.ipynb)
   - [Qwen3 4B](notebooks/finetune_qwen3_4b.ipynb)
3. Make sure to select **T4 GPU** runtime
4. Run all cells and follow the instructions
5. Download the GGUF model when complete

#### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_seq_length` | 2048 | Context length |
| `load_in_4bit` | True | Memory-efficient training |
| `r` | 16 | LoRA rank |
| `lora_alpha` | 16 | LoRA scaling |
| `learning_rate` | 2e-4 | Training speed |
| `batch_size` | 2 | Per-device batch size |
| `gradient_accumulation` | 4 | Effective batch size = 8 |

### 6. Deploy with Ollama

After downloading the GGUF model from Colab:

#### Install Ollama

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

#### Create Modelfile

For Llama 3.2:

```bash
cd hr-persona-bd-llama32-3b-gguf

cat > Modelfile << 'EOF'
FROM ./unsloth.Q4_K_M.gguf

TEMPLATE """{{- if .System }}<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{- end }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

SYSTEM """You are an expert HR consultant specializing in Bangladesh Labour Law. You have comprehensive knowledge of the Bangladesh Labour Act 2006 and its amendments. Provide accurate, professional advice to HR practitioners."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
EOF
```

For Qwen3:

```bash
cd hr-persona-bd-qwen3-4b-gguf

cat > Modelfile << 'EOF'
FROM ./unsloth.Q4_K_M.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """You are an expert HR consultant specializing in Bangladesh Labour Law. You have comprehensive knowledge of the Bangladesh Labour Act 2006 and its amendments. Provide accurate, professional advice to HR practitioners."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
EOF
```

#### Create and Run the Model

```bash
# Create model in Ollama
ollama create hr-persona-bd -f Modelfile

# Run interactive chat
ollama run hr-persona-bd
```

### 7. Use via API

#### Ollama API

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "hr-persona-bd",
  "messages": [
    {"role": "user", "content": "What is the maximum working hours per week in Bangladesh?"}
  ]
}'
```

#### Python with Ollama

```python
import ollama

response = ollama.chat(
    model='hr-persona-bd',
    messages=[
        {'role': 'user', 'content': 'What is the notice period for termination?'}
    ]
)
print(response['message']['content'])
```

### 8. Local Inference Script

Use the provided inference script for more control:

```bash
# Interactive chat with Ollama
python scripts/inference.py --backend ollama --model hr-persona-bd --interactive

# Single query
python scripts/inference.py --backend ollama --model hr-persona-bd \
  --query "How many days of annual leave is an employee entitled to?"

# Batch inference
python scripts/inference.py --backend ollama --model hr-persona-bd \
  --batch queries.json --output results.json
```

## Dataset Format

The training dataset uses ChatML format:

```json
[
  {
    "messages": [
      {"role": "user", "content": "What is the maximum working hours per week?"},
      {"role": "assistant", "content": "According to Section 100 of the Bangladesh Labour Act 2006..."}
    ]
  }
]
```

## Model Comparison

| Model | Parameters | VRAM (4-bit) | GGUF Size | Best For |
|-------|------------|--------------|-----------|----------|
| Llama 3.2 3B | 3.2B | ~4GB | ~2GB | Balanced performance |
| Qwen3 4B | 4B | ~5GB | ~2.5GB | Better reasoning |

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
# LLM Provider for dataset generation
llm:
  provider: "ollama"  # or "vllm", "api-endpoint"

# Ollama settings
ollama:
  model: "llama3.2:3b-instruct-q4_K_M"

# Generation settings
generation:
  temperature: 0.7
  chunk_size: 4000
  num_pairs: 50

# Quality curation
curate:
  threshold: 7.0
```

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

- Increase dataset size (aim for 1000+ QA pairs)
- Raise curation threshold (8.0+)
- Train for more epochs (2-3)
- Use larger model for dataset generation

## References

- [Unsloth Documentation](https://unsloth.ai/docs)
- [Meta Synthetic Data Kit](https://github.com/meta-llama/synthetic-data-kit)
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
