#!/usr/bin/env python3
"""
Deploy Fine-tuned Model to Ollama

This script automates the deployment of fine-tuned GGUF models to Ollama:
1. Creates the appropriate Modelfile based on model type
2. Registers the model with Ollama
3. Verifies the deployment

Usage:
    python scripts/deploy_ollama.py --gguf path/to/model.gguf --name hr-persona-bd --type llama
    python scripts/deploy_ollama.py --gguf path/to/model.gguf --name hr-persona-bd-qwen --type qwen
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional


# Modelfile templates for different model architectures
MODELFILE_TEMPLATES = {
    "llama": '''FROM {gguf_path}

TEMPLATE """{{{{- if .System }}}}<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{- end }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>"""

SYSTEM """{system_prompt}"""

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER stop "<|eot_id|>"
PARAMETER num_ctx {context_length}
''',

    "qwen": '''FROM {gguf_path}

TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
"""

SYSTEM """{system_prompt}"""

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx {context_length}
''',
}

# Default system prompt for HR Persona BD
DEFAULT_SYSTEM_PROMPT = """You are an expert HR consultant specializing in Bangladesh Labour Law. 
You have comprehensive knowledge of the Bangladesh Labour Act 2006 and its amendments up to 2018.
Provide accurate, professional advice to HR practitioners in Bangladesh.
When applicable, cite relevant sections of the Labour Act.
Always maintain a helpful, informative, and professional tone."""


def check_ollama_installed() -> bool:
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("Error: Ollama is not installed.")
        print("Install with: curl -fsSL https://ollama.com/install.sh | sh")
        return False
    except subprocess.TimeoutExpired:
        print("Error: Ollama is not responding. Make sure it's running.")
        print("Start with: ollama serve")
        return False


def create_modelfile(
    gguf_path: str,
    model_type: str,
    output_dir: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    top_p: float = 0.9,
    context_length: int = 2048,
) -> str:
    """
    Create a Modelfile for Ollama.
    
    Args:
        gguf_path: Path to the GGUF model file
        model_type: Model type (llama or qwen)
        output_dir: Directory to save the Modelfile
        system_prompt: System prompt for the model
        temperature: Sampling temperature
        top_p: Top-p sampling
        context_length: Context window size
        
    Returns:
        Path to the created Modelfile
    """
    if model_type not in MODELFILE_TEMPLATES:
        raise ValueError(f"Unknown model type: {model_type}. Use 'llama' or 'qwen'")
    
    # Get absolute path to GGUF file
    gguf_abs_path = os.path.abspath(gguf_path)
    
    # Create Modelfile content
    template = MODELFILE_TEMPLATES[model_type]
    modelfile_content = template.format(
        gguf_path=gguf_abs_path,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        context_length=context_length,
    )
    
    # Save Modelfile
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = output_dir / "Modelfile"
    
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"Created Modelfile at: {modelfile_path}")
    return str(modelfile_path)


def register_model(model_name: str, modelfile_path: str) -> bool:
    """
    Register the model with Ollama.
    
    Args:
        model_name: Name for the Ollama model
        modelfile_path: Path to the Modelfile
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\nRegistering model '{model_name}' with Ollama...")
    
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout for large models
        )
        
        if result.returncode == 0:
            print(f"Successfully registered model: {model_name}")
            return True
        else:
            print(f"Error registering model: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Error: Model registration timed out.")
        return False


def verify_deployment(model_name: str) -> bool:
    """
    Verify the model is deployed and working.
    
    Args:
        model_name: Name of the Ollama model
        
    Returns:
        True if verification successful
    """
    print(f"\nVerifying deployment of '{model_name}'...")
    
    # Check if model is in the list
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        
        if model_name not in result.stdout:
            print(f"Warning: Model '{model_name}' not found in Ollama list")
            return False
        
        print(f"Model '{model_name}' is registered.")
        
        # Test with a simple query
        print("\nTesting model with a sample query...")
        
        test_result = subprocess.run(
            ["ollama", "run", model_name, "What is the maximum working hours per week in Bangladesh?"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if test_result.returncode == 0 and test_result.stdout.strip():
            print("\nTest response:")
            print("-" * 40)
            print(test_result.stdout[:500] + ("..." if len(test_result.stdout) > 500 else ""))
            print("-" * 40)
            print("\nDeployment verified successfully!")
            return True
        else:
            print("Warning: Model did not respond as expected")
            return False
            
    except subprocess.TimeoutExpired:
        print("Warning: Test query timed out")
        return False


def list_models():
    """List all Ollama models."""
    print("\nInstalled Ollama models:")
    print("-" * 40)
    subprocess.run(["ollama", "list"])


def remove_model(model_name: str) -> bool:
    """
    Remove a model from Ollama.
    
    Args:
        model_name: Name of the model to remove
        
    Returns:
        True if successful
    """
    print(f"\nRemoving model '{model_name}'...")
    
    try:
        result = subprocess.run(
            ["ollama", "rm", model_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"Successfully removed model: {model_name}")
            return True
        else:
            print(f"Error removing model: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Deploy fine-tuned GGUF model to Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Deploy Llama model
    python scripts/deploy_ollama.py --gguf hr-persona-bd-llama32-3b-gguf/unsloth.Q4_K_M.gguf \\
        --name hr-persona-bd-llama --type llama
    
    # Deploy Qwen model
    python scripts/deploy_ollama.py --gguf hr-persona-bd-qwen3-4b-gguf/unsloth.Q4_K_M.gguf \\
        --name hr-persona-bd-qwen --type qwen
    
    # Deploy with custom settings
    python scripts/deploy_ollama.py --gguf model.gguf --name my-model --type llama \\
        --temperature 0.5 --context-length 4096
    
    # List models
    python scripts/deploy_ollama.py --list
    
    # Remove a model
    python scripts/deploy_ollama.py --remove hr-persona-bd-llama
        """
    )
    
    parser.add_argument(
        "--gguf", "-g",
        type=str,
        help="Path to the GGUF model file"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="hr-persona-bd",
        help="Name for the Ollama model (default: hr-persona-bd)"
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        choices=["llama", "qwen"],
        default="llama",
        help="Model architecture type (default: llama)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="Context window size (default: 2048)"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Custom system prompt"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        help="Directory to save Modelfile (default: current directory)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all Ollama models"
    )
    parser.add_argument(
        "--remove", "-r",
        type=str,
        metavar="MODEL_NAME",
        help="Remove a model from Ollama"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip deployment verification"
    )
    parser.add_argument(
        "--modelfile-only",
        action="store_true",
        help="Only create Modelfile, don't register with Ollama"
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        if not check_ollama_installed():
            sys.exit(1)
        list_models()
        sys.exit(0)
    
    # Handle remove command
    if args.remove:
        if not check_ollama_installed():
            sys.exit(1)
        success = remove_model(args.remove)
        sys.exit(0 if success else 1)
    
    # Validate GGUF path for deployment
    if not args.gguf:
        print("Error: --gguf is required for deployment")
        parser.print_help()
        sys.exit(1)
    
    if not Path(args.gguf).exists():
        print(f"Error: GGUF file not found: {args.gguf}")
        sys.exit(1)
    
    # Check Ollama installation (unless modelfile-only)
    if not args.modelfile_only:
        if not check_ollama_installed():
            sys.exit(1)
    
    print("=" * 60)
    print("HR Persona BD - Ollama Deployment")
    print("=" * 60)
    print(f"GGUF Model: {args.gguf}")
    print(f"Model Name: {args.name}")
    print(f"Model Type: {args.type}")
    print(f"Temperature: {args.temperature}")
    print(f"Context Length: {args.context_length}")
    
    # Create Modelfile
    modelfile_path = create_modelfile(
        gguf_path=args.gguf,
        model_type=args.type,
        output_dir=args.output_dir,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        context_length=args.context_length,
    )
    
    if args.modelfile_only:
        print(f"\nModelfile created at: {modelfile_path}")
        print("\nTo register with Ollama manually, run:")
        print(f"  ollama create {args.name} -f {modelfile_path}")
        sys.exit(0)
    
    # Register with Ollama
    if not register_model(args.name, modelfile_path):
        print("\nDeployment failed.")
        sys.exit(1)
    
    # Verify deployment
    if not args.skip_verify:
        verify_deployment(args.name)
    
    print("\n" + "=" * 60)
    print("Deployment Complete!")
    print("=" * 60)
    print(f"\nTo use the model:")
    print(f"  ollama run {args.name}")
    print(f"\nTo use via API:")
    print(f'  curl http://localhost:11434/api/chat -d \'{{"model": "{args.name}", "messages": [{{"role": "user", "content": "Your question"}}]}}\'')


if __name__ == "__main__":
    main()
