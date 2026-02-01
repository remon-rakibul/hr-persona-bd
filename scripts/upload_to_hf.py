#!/usr/bin/env python3
"""
Upload Models and Datasets to Hugging Face Hub

This script handles uploading:
1. Fine-tuned LoRA adapters
2. Merged models (16-bit or 4-bit)
3. GGUF quantized models
4. Training datasets

Usage:
    # Upload LoRA adapters
    python scripts/upload_to_hf.py --type lora --model-path ./hr-persona-bd-llama32-3b-lora --repo your-username/hr-persona-bd-llama-lora
    
    # Upload GGUF model
    python scripts/upload_to_hf.py --type gguf --model-path ./hr-persona-bd-llama32-3b-gguf --repo your-username/hr-persona-bd-llama-gguf
    
    # Upload dataset
    python scripts/upload_to_hf.py --type dataset --dataset-path ./data/final/dataset.json --repo your-username/hr-persona-bd-dataset
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import json


def check_huggingface_login():
    """Check if user is logged into Hugging Face."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        print("✗ Not logged into Hugging Face")
        print("\nTo login, run one of:")
        print("  1. huggingface-cli login")
        print("  2. python -c \"from huggingface_hub import login; login()\"")
        print("\nOr set the HF_TOKEN environment variable:")
        print("  export HF_TOKEN='your_token_here'")
        return False


def login_to_huggingface(token: Optional[str] = None):
    """Login to Hugging Face Hub."""
    from huggingface_hub import login
    
    if token:
        login(token=token)
    elif os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    else:
        # Interactive login
        login()
    
    return check_huggingface_login()


def upload_lora_model(
    model_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload LoRA adapters"
):
    """
    Upload LoRA adapter model to Hugging Face.
    
    Args:
        model_path: Path to the LoRA model directory
        repo_id: Hugging Face repository ID (username/repo-name)
        private: Whether to make the repo private
        commit_message: Commit message for the upload
    """
    from huggingface_hub import HfApi, create_repo
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload the entire directory
    print(f"\nUploading LoRA model from: {model_path}")
    print(f"To repository: {repo_id}")
    
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    
    print(f"\n✓ LoRA model uploaded successfully!")
    print(f"  URL: https://huggingface.co/{repo_id}")
    
    return f"https://huggingface.co/{repo_id}"


def upload_gguf_model(
    model_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload GGUF model"
):
    """
    Upload GGUF quantized model to Hugging Face.
    
    Args:
        model_path: Path to the GGUF model directory or file
        repo_id: Hugging Face repository ID (username/repo-name)
        private: Whether to make the repo private
        commit_message: Commit message for the upload
    """
    from huggingface_hub import HfApi, create_repo
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    model_path = Path(model_path)
    
    print(f"\nUploading GGUF model from: {model_path}")
    print(f"To repository: {repo_id}")
    
    if model_path.is_dir():
        # Upload entire directory
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            commit_message=commit_message,
        )
    else:
        # Upload single file
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=model_path.name,
            repo_id=repo_id,
            commit_message=commit_message,
        )
    
    print(f"\n✓ GGUF model uploaded successfully!")
    print(f"  URL: https://huggingface.co/{repo_id}")
    
    return f"https://huggingface.co/{repo_id}"


def upload_merged_model(
    model_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload merged model"
):
    """
    Upload merged (full) model to Hugging Face.
    
    Args:
        model_path: Path to the merged model directory
        repo_id: Hugging Face repository ID (username/repo-name)
        private: Whether to make the repo private
        commit_message: Commit message for the upload
    """
    from huggingface_hub import HfApi, create_repo
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    print(f"\nUploading merged model from: {model_path}")
    print(f"To repository: {repo_id}")
    print("This may take a while for large models...")
    
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    
    print(f"\n✓ Merged model uploaded successfully!")
    print(f"  URL: https://huggingface.co/{repo_id}")
    
    return f"https://huggingface.co/{repo_id}"


def upload_dataset(
    dataset_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload training dataset",
    dataset_name: str = "train",
    description: str = None
):
    """
    Upload dataset to Hugging Face.
    
    Args:
        dataset_path: Path to the dataset file (JSON) or directory
        repo_id: Hugging Face repository ID (username/repo-name)
        private: Whether to make the repo private
        commit_message: Commit message for the upload
        dataset_name: Name for the dataset split
        description: Dataset description
    """
    from huggingface_hub import HfApi, create_repo
    from datasets import Dataset, DatasetDict
    
    api = HfApi()
    
    # Create repo as dataset type
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"✓ Dataset repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    dataset_path = Path(dataset_path)
    
    print(f"\nLoading dataset from: {dataset_path}")
    
    # Load dataset
    if dataset_path.suffix == '.json':
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, list):
            dataset = Dataset.from_list(data)
        elif isinstance(data, dict) and 'data' in data:
            dataset = Dataset.from_list(data['data'])
        else:
            dataset = Dataset.from_dict(data)
    elif dataset_path.suffix == '.jsonl':
        dataset = Dataset.from_json(str(dataset_path))
    elif dataset_path.is_dir():
        # Load from directory with multiple files
        from datasets import load_dataset
        dataset = load_dataset(str(dataset_path))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")
    
    # Create dataset dict with train split
    if isinstance(dataset, Dataset):
        dataset_dict = DatasetDict({dataset_name: dataset})
    else:
        dataset_dict = dataset
    
    # Push to hub
    print(f"\nUploading dataset to: {repo_id}")
    dataset_dict.push_to_hub(
        repo_id,
        private=private,
        commit_message=commit_message,
    )
    
    # Create README if description provided
    if description:
        readme_content = f"""---
license: mit
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- hr
- bangladesh
- labour-law
- legal
size_categories:
- 1K<n<10K
---

# {repo_id.split('/')[-1]}

{description}

## Dataset Details

- **Samples**: {len(dataset)}
- **Format**: ChatML / Conversational
- **Language**: English
- **Domain**: Bangladesh Labour Law, HR Practices

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
print(dataset)
```

## Citation

If you use this dataset, please cite:

```
@dataset{{hr_persona_bd,
  title={{HR Persona Bangladesh Dataset}},
  year={{2026}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""
        # Upload README
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset README",
        )
    
    print(f"\n✓ Dataset uploaded successfully!")
    print(f"  URL: https://huggingface.co/datasets/{repo_id}")
    
    return f"https://huggingface.co/datasets/{repo_id}"


def create_model_card(
    repo_id: str,
    base_model: str,
    model_type: str = "lora",
    description: str = None
):
    """
    Create and upload a model card (README.md) for the model.
    
    Args:
        repo_id: Hugging Face repository ID
        base_model: Base model name (e.g., "unsloth/Llama-3.2-3B-Instruct")
        model_type: Type of model (lora, gguf, merged)
        description: Model description
    """
    from huggingface_hub import HfApi
    
    api = HfApi()
    
    if description is None:
        description = "Fine-tuned model for Bangladesh Labour Law and HR practices."
    
    model_card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- unsloth
- hr
- bangladesh
- labour-law
- legal
- llama
- fine-tuned
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

# {repo_id.split('/')[-1]}

{description}

## Model Details

- **Base Model**: [{base_model}](https://huggingface.co/{base_model})
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Framework**: Unsloth
- **Model Type**: {model_type.upper()}

## Intended Use

This model is designed for:
- Answering questions about Bangladesh Labour Act 2006
- Providing HR guidance for Bangladesh workplaces
- Explaining worker rights and employer obligations
- General HR policy assistance

## Usage

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

messages = [
    {{"role": "user", "content": "What is the maximum working hours per week in Bangladesh?"}}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### With Unsloth (Faster)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("{repo_id}")
FastLanguageModel.for_inference(model)
```

### With Ollama (GGUF version)

```bash
ollama run hr-persona-bd
```

## Training Details

- **Dataset**: Bangladesh Labour Act 2006 QA pairs
- **Epochs**: 1-3
- **Learning Rate**: 2e-4
- **LoRA Rank**: 16
- **LoRA Alpha**: 16

## Limitations

- Knowledge is limited to Bangladesh Labour Act 2006 (amended up to 2018)
- Should not be used as sole legal advice - consult legal professionals
- May not cover all edge cases or recent amendments

## Citation

```
@model{{hr_persona_bd,
  title={{HR Persona Bangladesh}},
  year={{2026}},
  base_model={{{base_model}}},
  url={{https://huggingface.co/{repo_id}}}
}}
```
"""
    
    # Upload model card
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )
    
    print(f"✓ Model card created for {repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload models and datasets to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Login to Hugging Face (do this first)
    python scripts/upload_to_hf.py --login
    
    # Upload LoRA adapters
    python scripts/upload_to_hf.py --type lora \\
        --model-path ./hr-persona-bd-llama32-3b-lora \\
        --repo your-username/hr-persona-bd-llama-lora \\
        --base-model unsloth/Llama-3.2-3B-Instruct
    
    # Upload GGUF model
    python scripts/upload_to_hf.py --type gguf \\
        --model-path ./hr-persona-bd-llama32-3b-gguf \\
        --repo your-username/hr-persona-bd-llama-gguf
    
    # Upload merged model
    python scripts/upload_to_hf.py --type merged \\
        --model-path ./hr-persona-bd-llama32-3b-merged \\
        --repo your-username/hr-persona-bd-llama
    
    # Upload dataset
    python scripts/upload_to_hf.py --type dataset \\
        --dataset-path ./data/final/dataset.json \\
        --repo your-username/hr-persona-bd-dataset \\
        --description "QA dataset for Bangladesh Labour Law"
    
    # Upload with private repo
    python scripts/upload_to_hf.py --type lora \\
        --model-path ./model \\
        --repo username/model \\
        --private
        """
    )
    
    parser.add_argument(
        "--login",
        action="store_true",
        help="Login to Hugging Face Hub"
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        choices=["lora", "gguf", "merged", "dataset"],
        help="Type of upload: lora, gguf, merged, or dataset"
    )
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--dataset-path", "-d",
        type=str,
        help="Path to the dataset file (for dataset upload)"
    )
    parser.add_argument(
        "--repo", "-r",
        type=str,
        help="Hugging Face repository ID (username/repo-name)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/Llama-3.2-3B-Instruct",
        help="Base model name for model card"
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Description for model card or dataset"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--skip-card",
        action="store_true",
        help="Skip creating model card"
    )
    
    args = parser.parse_args()
    
    # Handle login
    if args.login:
        login_to_huggingface(args.token)
        sys.exit(0)
    
    # Check login status
    if not check_huggingface_login():
        if args.token:
            login_to_huggingface(args.token)
        else:
            print("\nPlease login first: python scripts/upload_to_hf.py --login")
            sys.exit(1)
    
    # Validate arguments
    if not args.type:
        print("Error: --type is required")
        parser.print_help()
        sys.exit(1)
    
    if not args.repo:
        print("Error: --repo is required")
        parser.print_help()
        sys.exit(1)
    
    # Perform upload based on type
    if args.type == "dataset":
        if not args.dataset_path:
            print("Error: --dataset-path is required for dataset upload")
            sys.exit(1)
        
        if not Path(args.dataset_path).exists():
            print(f"Error: Dataset not found: {args.dataset_path}")
            sys.exit(1)
        
        upload_dataset(
            dataset_path=args.dataset_path,
            repo_id=args.repo,
            private=args.private,
            description=args.description or "HR Persona Bangladesh - Labour Law QA Dataset"
        )
    
    else:  # Model upload
        if not args.model_path:
            print("Error: --model-path is required for model upload")
            sys.exit(1)
        
        if not Path(args.model_path).exists():
            print(f"Error: Model not found: {args.model_path}")
            sys.exit(1)
        
        if args.type == "lora":
            upload_lora_model(
                model_path=args.model_path,
                repo_id=args.repo,
                private=args.private
            )
        elif args.type == "gguf":
            upload_gguf_model(
                model_path=args.model_path,
                repo_id=args.repo,
                private=args.private
            )
        elif args.type == "merged":
            upload_merged_model(
                model_path=args.model_path,
                repo_id=args.repo,
                private=args.private
            )
        
        # Create model card
        if not args.skip_card:
            create_model_card(
                repo_id=args.repo,
                base_model=args.base_model,
                model_type=args.type,
                description=args.description
            )
    
    print("\n" + "="*50)
    print("Upload Complete!")
    print("="*50)


if __name__ == "__main__":
    main()
