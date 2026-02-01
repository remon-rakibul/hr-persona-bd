# Troubleshooting Guide

Common issues and solutions for the HR Persona BD project.

## PDF to QA Conversion Issues

### Error: "Authentication Error" (401 Unauthorized)

**Symptoms:**
```
ERROR: api-endpoint API error (attempt 1/3): Error code: 401
{'title': 'Authentication Error', 'detail': 'Authentication Error', 'status': 401}
```

**Cause:** The synthetic-data-kit is trying to use an external API instead of your local Ollama instance.

**Solution:**

1. **Make sure Ollama is installed and running:**
   ```bash
   # Check if Ollama is installed
   ollama --version
   
   # Start Ollama server
   ollama serve
   
   # In another terminal, verify it's running
   curl http://localhost:11434/api/tags
   ```

2. **Pull the required model:**
   ```bash
   ollama pull llama3.2:3b-instruct-q4_K_M
   
   # Verify it's available
   ollama list
   ```

3. **Check your config file** (`configs/config.yaml`):
   ```yaml
   llm:
     provider: "ollama"  # Must be "ollama", not "api-endpoint"
   
   ollama:
     api_base: "http://localhost:11434/v1"
     model: "llama3.2:3b-instruct-q4_K_M"
   ```

4. **Re-run the script:**
   ```bash
   python scripts/pdf_to_qa.py --pdf data/input/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf
   ```

---

### Error: "Connection refused" or "Cannot connect to Ollama"

**Cause:** Ollama server is not running.

**Solution:**
```bash
# Start Ollama in background
ollama serve &

# Or in a separate terminal
ollama serve
```

---

### Error: "Model not found"

**Cause:** The model specified in config hasn't been pulled yet.

**Solution:**
```bash
# Pull the model
ollama pull llama3.2:3b-instruct-q4_K_M

# Alternative: use a different model
ollama pull llama3.3:70b-instruct  # Larger, better quality
```

Then update `configs/config.yaml` if using a different model:
```yaml
ollama:
  model: "llama3.3:70b-instruct"
```

---

## Dataset Extension Issues

### Ollama Connection Error in `extend_dataset_ollama.py`

**Solution:**
```bash
# Check Ollama status
systemctl status ollama  # Linux with systemd

# Or restart manually
pkill ollama
ollama serve
```

---

### OpenAI API Key Error in `extend_dataset_openai.py`

**Symptoms:**
```
Error: OPENAI_API_KEY environment variable not set
```

**Solution:**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Or add to .env file
echo 'OPENAI_API_KEY="sk-..."' > .env
```

---

## Colab Training Issues

### CUDA Out of Memory

**Solution:**
1. **Use 4-bit quantization** (already enabled by default):
   ```python
   load_in_4bit = True
   ```

2. **Reduce batch size:**
   ```python
   per_device_train_batch_size = 1  # Instead of 2
   ```

3. **Reduce context length:**
   ```python
   max_seq_length = 1024  # Instead of 2048
   ```

4. **Restart runtime** in Colab:
   - Runtime → Restart runtime

---

### Model Not Loading in Colab

**Cause:** Model name changed or incorrect.

**Solution:**

For Llama 3.2:
```python
model_name = "unsloth/Llama-3.2-3B-Instruct"
# Or use pre-quantized version
model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
```

For Qwen3:
```python
model_name = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
```

---

### Training Loss Not Decreasing

**Causes & Solutions:**

1. **Learning rate too high:**
   ```python
   learning_rate = 1e-4  # Try smaller (default is 2e-4)
   ```

2. **Dataset too small:**
   - Aim for 1000+ QA pairs
   - Extend dataset using `extend_dataset_ollama.py`

3. **Bad data quality:**
   - Increase curation threshold in `configs/config.yaml`:
     ```yaml
     curate:
       threshold: 8.0  # Higher = stricter
     ```

---

## Ollama Deployment Issues

### Model Not Found After Creating

**Symptoms:**
```bash
ollama run hr-persona-bd
Error: model 'hr-persona-bd' not found
```

**Solution:**
```bash
# List models
ollama list

# If not there, recreate
cd /path/to/gguf/directory
ollama create hr-persona-bd -f Modelfile

# Verify
ollama list | grep hr-persona-bd
```

---

### Incorrect Response Format (Template Issues)

**Cause:** Wrong chat template for model type.

**Solution:**

For **Llama 3.x** models, use:
```
TEMPLATE """{{- if .System }}<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{- end }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
```

For **Qwen** models, use:
```
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
```

---

## Inference Issues

### HuggingFace Model Not Loading

**Cause:** Model path incorrect or unsloth not installed.

**Solution:**
```bash
# Install unsloth for local inference
pip install unsloth

# Use absolute path or correct relative path
python scripts/inference.py --model ./hr-persona-bd-llama32-3b-lora --interactive
```

---

### Slow Inference

**Solutions:**

1. **Use GGUF with Ollama instead** (faster):
   ```bash
   python scripts/inference.py --backend ollama --model hr-persona-bd --interactive
   ```

2. **Reduce max tokens:**
   ```bash
   python scripts/inference.py --model ./model --max-tokens 256 --interactive
   ```

3. **Use GPU if available:**
   - Check: `nvidia-smi`
   - Make sure PyTorch detects GPU: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Hugging Face Upload Issues

### Authentication Failed

**Solution:**
```bash
# Login to Hugging Face
huggingface-cli login

# Or use token directly
python scripts/upload_to_hf.py --login
```

---

### Upload Timeout

**Cause:** Large files with slow internet.

**Solution:**

1. **Upload GGUF instead of full model** (smaller):
   ```bash
   python scripts/upload_to_hf.py --type gguf --model-path ./model-gguf --repo username/model
   ```

2. **Use Colab's faster connection:**
   - Upload directly from Colab notebook

---

## General Issues

### Package Installation Errors

**Solution:**
```bash
# Use fresh virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Upgrade pip first
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

---

### "synthetic-data-kit" Command Not Found

**Solution:**
```bash
# Install it
pip install synthetic-data-kit

# Verify installation
synthetic-data-kit --help
```

---

## Getting Help

If you're still stuck:

1. **Check logs:** Most scripts have verbose output showing what went wrong
2. **Verify environment:**
   ```bash
   python --version  # Should be 3.10+
   pip list | grep -E "unsloth|ollama|transformers"
   ```
3. **Check system resources:**
   ```bash
   nvidia-smi  # For GPU info
   free -h     # For RAM
   df -h       # For disk space
   ```
4. **Open an issue** on GitHub with:
   - Error message (full output)
   - Your OS and Python version
   - Steps to reproduce

---

## Quick Checklist

Before running the pipeline:

- [ ] Python 3.10+ installed
- [ ] Virtual environment activated
- [ ] `requirements.txt` dependencies installed
- [ ] Ollama installed and running (`ollama serve`)
- [ ] Model pulled (`ollama pull llama3.2:3b-instruct-q4_K_M`)
- [ ] Config file uses `provider: "ollama"`
- [ ] PDF file exists in `data/input/`
- [ ] Data directories exist (`data/parsed`, `data/generated`, etc.)
