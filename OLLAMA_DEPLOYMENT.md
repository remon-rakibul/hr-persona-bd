# Ollama Deployment Guide

Complete guide for deploying HR Persona BD fine-tuned models with Ollama.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Install Ollama](#install-ollama)
3. [Download Your Fine-tuned Model](#download-your-fine-tuned-model)
4. [Deploy with Script (Recommended)](#deploy-with-script-recommended)
5. [Manual Deployment](#manual-deployment)
6. [Using the Model](#using-the-model)
7. [API Integration](#api-integration)
8. [Advanced Configuration](#advanced-configuration)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Fine-tuned GGUF model file (exported from Colab notebook)
- At least 4GB RAM for 3B model, 6GB for 4B model
- Ollama installed on your system

---

## Install Ollama

### Linux / macOS

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

Download the installer from [ollama.com/download](https://ollama.com/download)

### Verify Installation

```bash
ollama --version
```

### Start Ollama Server

```bash
# Start the server (runs in background)
ollama serve

# Or start as a service (Linux with systemd)
sudo systemctl start ollama
sudo systemctl enable ollama  # Auto-start on boot
```

---

## Download Your Fine-tuned Model

After running the Colab notebook, you'll have a folder containing:

```
hr-persona-bd-llama32-3b-gguf/
├── llama-3.2-3b-instruct.Q4_K_M.gguf   # The GGUF file (~2GB) - actual name from Colab
│   # (Some exports may name it unsloth.Q4_K_M.gguf - use the name you see in Colab.)
├── config.json
└── tokenizer files...
```

1. Download the zip from Colab
2. Extract to a directory:

```bash
unzip hr-persona-bd-llama32-3b-gguf.zip -d ~/models/
```

---

## Deploy with Script (Recommended)

Use the provided deployment script for automatic setup:

### For Llama 3.2 3B

```bash
# From project root - use the GGUF filename from Colab (often llama-3.2-3b-instruct.Q4_K_M.gguf)
python scripts/deploy_ollama.py \
    --gguf hr-persona-bd-llama32-3b-gguf/llama-3.2-3b-instruct.Q4_K_M.gguf \
    --name hr-persona-bd \
    --type llama

# Or with full path
python scripts/deploy_ollama.py \
    --gguf /path/to/hr-persona-bd-llama32-3b-gguf/llama-3.2-3b-instruct.Q4_K_M.gguf \
    --name hr-persona-bd \
    --type llama
```

### For Qwen3 4B

```bash
python scripts/deploy_ollama.py \
    --gguf hr-persona-bd-qwen3-4b-gguf/qwen3-4b-instruct.Q4_K_M.gguf \
    --name hr-persona-bd-qwen \
    --type qwen
```

### Script Options

```bash
python scripts/deploy_ollama.py --help

Options:
  --gguf, -g         Path to GGUF model file (required)
  --name, -n         Name for Ollama model (default: hr-persona-bd)
  --type, -t         Model type: llama or qwen (default: llama)
  --temperature      Sampling temperature (default: 0.7)
  --context-length   Context window size (default: 2048)
  --list, -l         List all installed Ollama models
  --remove, -r       Remove a model from Ollama
```

---

## Manual Deployment

If you prefer manual setup or the script doesn't work:

**Important:** You must run `ollama create` from the **same directory that contains the .gguf file**. Otherwise you get "file does not exist" when running the model.

### Step 1: Navigate to Model Directory

```bash
# Go into the folder that contains the .gguf file (after downloading from Colab)
cd hr-persona-bd-llama32-3b-gguf/
# Or: cd ~/models/hr-persona-bd-llama32-3b-gguf/
# List to confirm filename: ls *.gguf  (often llama-3.2-3b-instruct.Q4_K_M.gguf)
```

### Step 2: Create Modelfile

#### For Llama 3.2 Models

Create a file named `Modelfile` in that same directory. Use the **exact GGUF filename** from your folder (e.g. `llama-3.2-3b-instruct.Q4_K_M.gguf`):

```
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
PARAMETER num_ctx 2048
```

#### For Qwen3 Models

Create a file named `Modelfile`. Use the exact GGUF filename from Colab (e.g. `qwen3-4b-instruct.Q4_K_M.gguf` or similar):

```
FROM ./qwen3-4b-instruct.Q4_K_M.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """You are an expert HR consultant specializing in Bangladesh Labour Law. 
You have comprehensive knowledge of the Bangladesh Labour Act 2006 and its amendments up to 2018.
Provide accurate, professional advice to HR practitioners in Bangladesh.
When applicable, cite relevant sections of the Labour Act.
Always maintain a helpful, informative, and professional tone."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 2048
```

### Step 3: Create Model in Ollama

```bash
ollama create hr-persona-bd -f Modelfile
```

Wait for the model to be registered (may take a minute).

### Step 4: Verify Installation

```bash
# List models
ollama list

# Should show:
# NAME                    SIZE      MODIFIED
# hr-persona-bd:latest    2.0 GB    Just now
```

---

## Using the Model

### Interactive Chat

```bash
ollama run hr-persona-bd
```

This opens an interactive chat session:

```
>>> What is the maximum working hours per week in Bangladesh?

According to Section 100 of the Bangladesh Labour Act 2006, no adult worker 
shall ordinarily be required or allowed to work in an establishment for more 
than 48 hours in any week. However, this can be extended to 60 hours per week 
with overtime, subject to the worker's consent and payment of overtime wages 
at twice the normal rate...

>>> What about overtime pay?

Section 108 of the Labour Act specifies that overtime work must be compensated 
at twice the ordinary rate of basic wages and dearness allowance...

>>> /bye
```

### Single Query

```bash
ollama run hr-persona-bd "How many days of annual leave is an employee entitled to?"
```

---

## API Integration

### REST API

Ollama runs a local API server on port 11434.

#### Chat Completion

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "hr-persona-bd",
  "messages": [
    {
      "role": "user",
      "content": "What is the notice period for termination in Bangladesh?"
    }
  ],
  "stream": false
}'
```

#### Streaming Response

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "hr-persona-bd",
  "messages": [
    {"role": "user", "content": "Explain maternity leave benefits"}
  ],
  "stream": true
}'
```

### Python Integration

```python
import ollama

# Simple query
response = ollama.chat(
    model='hr-persona-bd',
    messages=[
        {'role': 'user', 'content': 'What are the rules for child labor in Bangladesh?'}
    ]
)
print(response['message']['content'])

# Streaming
for chunk in ollama.chat(
    model='hr-persona-bd',
    messages=[{'role': 'user', 'content': 'Explain worker compensation'}],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
```

### JavaScript/Node.js Integration

```javascript
// Using fetch
const response = await fetch('http://localhost:11434/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'hr-persona-bd',
    messages: [
      { role: 'user', content: 'What is gratuity in Bangladesh Labour Act?' }
    ],
    stream: false
  })
});

const data = await response.json();
console.log(data.message.content);
```

### Using with LangChain

```python
from langchain_community.llms import Ollama

llm = Ollama(model="hr-persona-bd")

response = llm.invoke("What are the trade union rights in Bangladesh?")
print(response)
```

---

## Advanced Configuration

### Custom System Prompts

Create different versions for different use cases:

#### For General HR Queries

```
SYSTEM """You are an HR assistant for Bangladesh. Answer questions about 
employment, leave policies, and workplace regulations based on the 
Bangladesh Labour Act 2006."""
```

#### For Legal Compliance

```
SYSTEM """You are a labour law compliance advisor. Provide precise legal 
guidance with section references from the Bangladesh Labour Act 2006. 
Always recommend consulting a legal professional for complex matters."""
```

#### For Employee Relations

```
SYSTEM """You are an employee relations specialist. Help resolve workplace 
issues while ensuring compliance with Bangladesh Labour Act 2006. 
Be empathetic but professional."""
```

### Performance Tuning

Edit Modelfile parameters:

```
# Lower temperature for more consistent responses
PARAMETER temperature 0.3

# Higher context for longer conversations
PARAMETER num_ctx 4096

# Limit response length
PARAMETER num_predict 512

# Use more threads (for faster inference)
PARAMETER num_thread 8
```

### Running Multiple Models

```bash
# Create specialized versions
ollama create hr-persona-bd-legal -f Modelfile.legal
ollama create hr-persona-bd-employee -f Modelfile.employee

# Run specific version
ollama run hr-persona-bd-legal
```

---

## Troubleshooting

### Model Not Found

```
Error: model 'hr-persona-bd' not found
```

**Solution:**
```bash
# Check if model is registered
ollama list

# Re-create from inside the GGUF folder
cd hr-persona-bd-llama32-3b-gguf
ollama create hr-persona-bd -f Modelfile
```

### Ollama Not Running

```
Error: could not connect to ollama server
```

**Solution:**
```bash
# Start Ollama
ollama serve

# Or restart service
sudo systemctl restart ollama
```

### Out of Memory

```
Error: CUDA out of memory / not enough RAM
```

**Solutions:**
1. Close other applications
2. Use a smaller quantization (Q4_K_S instead of Q4_K_M)
3. Reduce context length:
   ```
   PARAMETER num_ctx 1024
   ```

### Slow Responses

**Solutions:**
1. Enable GPU acceleration (if available)
2. Increase CPU threads:
   ```
   PARAMETER num_thread 8
   ```
3. Use smaller model (3B instead of 4B)

### Wrong Template Format

If responses are garbled or include template tokens:

**Solution:** Verify you're using the correct template for your model type:
- Llama 3.x: Uses `<|begin_of_text|>`, `<|eot_id|>` tokens
- Qwen: Uses `<|im_start|>`, `<|im_end|>` tokens

### "Invalid model name" (400 Bad Request)

**Solution:** Some Ollama versions reject certain names. Try creating from inside the GGUF folder, or use the deploy script with full path to the GGUF file.

### "File does not exist" when running model

**Cause:** You ran `ollama create` from the wrong directory, or the Modelfile's `FROM` line uses a filename that doesn't match your GGUF file (e.g. Colab saves `llama-3.2-3b-instruct.Q4_K_M.gguf`, not `unsloth.Q4_K_M.gguf`).

**Solution:**
```bash
# Must run from the folder that contains the .gguf file; FROM must match actual filename
cd hr-persona-bd-llama32-3b-gguf
ls *.gguf   # use this name in Modelfile FROM line
ollama create hr-persona-bd -f Modelfile
```

Or use the deploy script with the full path to the GGUF file (no need to cd):
```bash
python scripts/deploy_ollama.py --gguf hr-persona-bd-llama32-3b-gguf/llama-3.2-3b-instruct.Q4_K_M.gguf --name hr-persona-bd --type llama
```

### Check Logs

```bash
# View Ollama logs
journalctl -u ollama -f

# Or if running manually
ollama serve 2>&1 | tee ollama.log
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `ollama serve` | Start Ollama server |
| `ollama list` | List installed models |
| `ollama run MODEL` | Interactive chat |
| `ollama create NAME -f FILE` | Create model from Modelfile (run from GGUF folder) |
| `ollama rm MODEL` | Remove a model |
| `ollama show MODEL` | Show model details |
| `ollama cp MODEL NEW` | Copy/rename a model |

---

## Model Files Location

Ollama stores models in:

- **Linux:** `~/.ollama/models/`
- **macOS:** `~/.ollama/models/`
- **Windows:** `C:\Users\<user>\.ollama\models\`

---

## Next Steps

1. Test the model with sample HR questions
2. Integrate with your HR application
3. Fine-tune further if needed with additional data
4. Consider deploying behind an API gateway for production use
