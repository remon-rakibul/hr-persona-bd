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
├── data/
│   ├── input/                    # Source PDFs
│   ├── parsed/                   # Parsed text from PDFs
│   ├── generated/                # Initial QA pairs
│   ├── curated/                  # Quality-filtered QA pairs
│   └── final/                    # Training-ready datasets
├── scripts/
│   ├── pdf_to_qa.py              # PDF to QA conversion
│   ├── extend_dataset_ollama.py  # Dataset extension via Ollama
│   ├── extend_dataset_openai.py  # Dataset extension via OpenAI
│   └── inference.py              # Local inference script
├── notebooks/
│   ├── finetune_llama32_3b.ipynb # Llama 3.2 3B fine-tuning
│   └── finetune_qwen3_4b.ipynb   # Qwen3 4B fine-tuning
├── configs/
│   └── config.yaml               # Configuration file
├── requirements.txt              # Dependencies
└── README.md                     # This file
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

### 2. Generate QA Dataset from PDF

#### Option A: Using Meta's Synthetic Data Kit (Recommended)

First, start a local LLM with Ollama:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model for QA generation
ollama pull llama3.2:3b-instruct-q4_K_M

# Start Ollama server
ollama serve
```

Then run the PDF to QA conversion:

```bash
python scripts/pdf_to_qa.py --pdf data/input/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf
```

This will:
1. Parse the PDF to extract text
2. Generate QA pairs using the LLM
3. Curate pairs for quality
4. Save in ChatML format for training

#### Option B: Using Synthetic Data Kit CLI Directly

```bash
# Install synthetic-data-kit
pip install synthetic-data-kit

# Parse PDF
synthetic-data-kit ingest data/input/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf

# Generate QA pairs (50 per chunk)
synthetic-data-kit create data/parsed/Bangladesh-Labour-Act-2006_English-Upto-2018.txt \
  --type qa --num-pairs 50 --chunk-size 4000

# Curate with quality threshold
synthetic-data-kit curate data/generated/*.json --threshold 7.0

# Save as ChatML format
synthetic-data-kit save-as data/curated/*.json --format chatml --storage hf
```

### 3. Extend Dataset (Optional)

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

### 4. Fine-tune the Model

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

### 5. Deploy with Ollama

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

### 6. Use via API

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

### 7. Local Inference Script

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
- Contact: [your-email@example.com]
