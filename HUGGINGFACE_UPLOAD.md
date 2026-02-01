# Hugging Face Upload Guide

Complete guide for uploading your fine-tuned models and datasets to Hugging Face Hub.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Authentication](#authentication)
3. [Upload Models](#upload-models)
4. [Upload Datasets](#upload-datasets)
5. [Upload from Colab](#upload-from-colab)
6. [Repository Management](#repository-management)
7. [Best Practices](#best-practices)

---

## Prerequisites

### Install Required Packages

```bash
pip install huggingface_hub datasets transformers
```

### Create Hugging Face Account

1. Go to [huggingface.co](https://huggingface.co)
2. Click "Sign Up" and create an account
3. Verify your email

### Get Access Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name it (e.g., "hr-persona-bd")
4. Select "Write" permission
5. Copy the token (you'll only see it once!)

---

## Authentication

### Method 1: Using the Upload Script

```bash
python scripts/upload_to_hf.py --login
```

This will prompt you to enter your token interactively.

### Method 2: CLI Login

```bash
huggingface-cli login
```

Paste your token when prompted.

### Method 3: Environment Variable

```bash
export HF_TOKEN="hf_your_token_here"
```

Add to `~/.bashrc` or `~/.zshrc` for persistence.

### Method 4: Python Login

```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

### Verify Login

```bash
huggingface-cli whoami
```

---

## Upload Models

### Upload LoRA Adapters (Recommended)

LoRA adapters are small (~100-200MB) and efficient to share.

```bash
python scripts/upload_to_hf.py \
    --type lora \
    --model-path ./hr-persona-bd-llama32-3b-lora \
    --repo YOUR_USERNAME/hr-persona-bd-llama-lora \
    --base-model unsloth/Llama-3.2-3B-Instruct
```

### Upload GGUF Model

GGUF models are ready for Ollama and llama.cpp.

```bash
python scripts/upload_to_hf.py \
    --type gguf \
    --model-path ./hr-persona-bd-llama32-3b-gguf \
    --repo YOUR_USERNAME/hr-persona-bd-llama-gguf
```

### Upload Merged Model

Full merged model (larger, ~6-8GB for 3B model).

```bash
python scripts/upload_to_hf.py \
    --type merged \
    --model-path ./hr-persona-bd-llama32-3b-merged \
    --repo YOUR_USERNAME/hr-persona-bd-llama
```

### Upload Private Model

Add `--private` flag for private repositories:

```bash
python scripts/upload_to_hf.py \
    --type lora \
    --model-path ./model \
    --repo YOUR_USERNAME/private-model \
    --private
```

---

## Upload Datasets

### Upload Training Dataset

```bash
python scripts/upload_to_hf.py \
    --type dataset \
    --dataset-path ./data/final/bangladesh_labour_act_chatml.json \
    --repo YOUR_USERNAME/hr-persona-bd-dataset \
    --description "QA dataset for Bangladesh Labour Law fine-tuning"
```

### Upload Extended Dataset

```bash
python scripts/upload_to_hf.py \
    --type dataset \
    --dataset-path ./data/final/extended_dataset.json \
    --repo YOUR_USERNAME/hr-persona-bd-extended \
    --description "Extended QA dataset with synthetic data"
```

### Manual Dataset Upload (Python)

```python
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import json

# Load your dataset
with open("data/final/dataset.json") as f:
    data = json.load(f)

# Create Dataset object
dataset = Dataset.from_list(data)

# Optional: Split into train/test
dataset_dict = dataset.train_test_split(test_size=0.1)

# Push to Hub
dataset_dict.push_to_hub(
    "YOUR_USERNAME/hr-persona-bd-dataset",
    private=False
)

print("Dataset uploaded!")
```

---

## Upload from Colab

### Option 1: Use Unsloth's Built-in Methods

The notebooks already include upload code. Just uncomment and run:

```python
# In Colab notebook (already included in the notebooks)

# Login to Hugging Face
from huggingface_hub import login
login(token="hf_your_token_here")

# Push LoRA adapters
model.push_to_hub("YOUR_USERNAME/hr-persona-bd-llama-lora")
tokenizer.push_to_hub("YOUR_USERNAME/hr-persona-bd-llama-lora")

# Push GGUF model
model.push_to_hub_gguf(
    "YOUR_USERNAME/hr-persona-bd-llama-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

### Option 2: Upload After Training Completes

```python
# After training, in the same Colab session

from huggingface_hub import HfApi, login

# Login
login(token="hf_your_token_here")

# Upload LoRA
api = HfApi()
api.upload_folder(
    folder_path="hr-persona-bd-llama32-3b-lora",
    repo_id="YOUR_USERNAME/hr-persona-bd-llama-lora",
    commit_message="Upload fine-tuned LoRA adapters"
)

# Upload GGUF
api.upload_folder(
    folder_path="hr-persona-bd-llama32-3b-gguf",
    repo_id="YOUR_USERNAME/hr-persona-bd-llama-gguf",
    commit_message="Upload GGUF Q4_K_M model"
)
```

### Option 3: Download and Upload Later

1. Download the model from Colab
2. Extract on local machine
3. Use the upload script:

```bash
python scripts/upload_to_hf.py --type gguf --model-path ./extracted-model --repo YOUR_USERNAME/model
```

---

## Repository Management

### List Your Repositories

```python
from huggingface_hub import HfApi

api = HfApi()
repos = api.list_repos_objs(author="YOUR_USERNAME")

for repo in repos:
    print(f"{repo.id} - {repo.private}")
```

### Delete a Repository

```python
from huggingface_hub import delete_repo

delete_repo("YOUR_USERNAME/repo-to-delete")
# or for datasets:
delete_repo("YOUR_USERNAME/dataset-to-delete", repo_type="dataset")
```

### Update Repository Settings

```python
from huggingface_hub import update_repo_visibility

# Make public
update_repo_visibility("YOUR_USERNAME/repo-name", private=False)

# Make private
update_repo_visibility("YOUR_USERNAME/repo-name", private=True)
```

### Add Collaborators

1. Go to your repository on huggingface.co
2. Click "Settings"
3. Go to "Access" tab
4. Add collaborators by username

---

## Best Practices

### Model Naming Convention

Use consistent naming:

```
username/hr-persona-bd-{model}-{type}

Examples:
- username/hr-persona-bd-llama32-3b-lora
- username/hr-persona-bd-llama32-3b-gguf
- username/hr-persona-bd-qwen3-4b-lora
```

### Dataset Naming Convention

```
username/hr-persona-bd-{description}

Examples:
- username/hr-persona-bd-labour-act-qa
- username/hr-persona-bd-extended-dataset
```

### Include Good Documentation

Always include:
- Model card (README.md) with usage instructions
- Training details (hyperparameters, base model)
- Limitations and intended use
- Citation information

### Version Your Uploads

Use commit messages to track versions:

```bash
python scripts/upload_to_hf.py \
    --type lora \
    --model-path ./model-v2 \
    --repo username/model \
    --commit-message "v2.0: Trained on extended dataset"
```

### Test Before Public Release

1. Upload as private first
2. Test loading and inference
3. Make public when ready:

```python
from huggingface_hub import update_repo_visibility
update_repo_visibility("username/repo", private=False)
```

---

## Troubleshooting

### "Repository not found"

- Check you're logged in: `huggingface-cli whoami`
- Verify the repository name is correct
- Ensure you have write permissions

### "Token is invalid"

- Generate a new token at huggingface.co/settings/tokens
- Make sure you selected "Write" permission
- Try logging in again

### Upload Timeout

For large models:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./large-model",
    repo_id="username/model",
    commit_message="Upload large model",
    multi_commits=True,  # Upload in multiple commits
    multi_commits_verbose=True,
)
```

### Slow Upload

- Use a stable internet connection
- Upload GGUF (smaller) instead of full model
- Consider using Colab's faster upload speeds

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `python scripts/upload_to_hf.py --login` | Login to Hugging Face |
| `--type lora` | Upload LoRA adapters |
| `--type gguf` | Upload GGUF model |
| `--type merged` | Upload merged model |
| `--type dataset` | Upload dataset |
| `--private` | Make repository private |
| `--skip-card` | Skip creating model card |

---

## Example: Complete Upload Workflow

```bash
# 1. Login
python scripts/upload_to_hf.py --login

# 2. Upload dataset
python scripts/upload_to_hf.py \
    --type dataset \
    --dataset-path ./data/final/bangladesh_labour_act_chatml.json \
    --repo myusername/hr-bd-dataset \
    --description "Bangladesh Labour Act QA Dataset"

# 3. Upload LoRA model
python scripts/upload_to_hf.py \
    --type lora \
    --model-path ./hr-persona-bd-llama32-3b-lora \
    --repo myusername/hr-bd-llama-lora \
    --base-model unsloth/Llama-3.2-3B-Instruct

# 4. Upload GGUF model
python scripts/upload_to_hf.py \
    --type gguf \
    --model-path ./hr-persona-bd-llama32-3b-gguf \
    --repo myusername/hr-bd-llama-gguf

# Done! Your models are now on Hugging Face:
# - https://huggingface.co/datasets/myusername/hr-bd-dataset
# - https://huggingface.co/myusername/hr-bd-llama-lora
# - https://huggingface.co/myusername/hr-bd-llama-gguf
```
