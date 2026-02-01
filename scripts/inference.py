#!/usr/bin/env python3
"""
Local Inference Script for HR Persona BD Fine-tuned Models

Supports:
- HuggingFace format models (LoRA adapters or merged)
- Ollama models (via API)
- Interactive chat mode
- Single query mode

Usage:
    # Interactive chat with HuggingFace model
    python scripts/inference.py --model hr-persona-bd-llama32-3b-lora --interactive
    
    # Single query
    python scripts/inference.py --model hr-persona-bd-llama32-3b-lora --query "What is annual leave?"
    
    # Use Ollama backend
    python scripts/inference.py --backend ollama --model hr-persona-bd-llama --interactive
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict
import json


def load_huggingface_model(model_path: str, load_in_4bit: bool = True):
    """
    Load a fine-tuned model from HuggingFace format.
    
    Args:
        model_path: Path to the model directory
        load_in_4bit: Whether to use 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Error: unsloth not installed. Run: pip install unsloth")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer


def generate_response_hf(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate a response using HuggingFace model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        messages: List of message dicts with 'role' and 'content'
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Generated response text
    """
    import torch
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    # This depends on the chat template used
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()
    
    return response


def generate_response_ollama(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
) -> str:
    """
    Generate a response using Ollama API.
    
    Args:
        model_name: Name of the Ollama model
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        
    Returns:
        Generated response text
    """
    try:
        import ollama
    except ImportError:
        print("Error: ollama package not installed. Run: pip install ollama")
        sys.exit(1)
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={
                "temperature": temperature,
            }
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return None


def interactive_chat(
    backend: str,
    model=None,
    tokenizer=None,
    model_name: str = None,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
):
    """
    Run interactive chat session.
    
    Args:
        backend: Either 'huggingface' or 'ollama'
        model: HuggingFace model (if backend is 'huggingface')
        tokenizer: HuggingFace tokenizer (if backend is 'huggingface')
        model_name: Ollama model name (if backend is 'ollama')
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
    """
    print("\n" + "="*60)
    print("HR Persona Bangladesh - Interactive Chat")
    print("="*60)
    print("Ask questions about Bangladesh Labour Law and HR practices.")
    print("Type 'quit', 'exit', or 'q' to exit.")
    print("Type 'clear' to clear conversation history.")
    print("="*60 + "\n")
    
    # System message for context
    system_message = {
        "role": "system",
        "content": """You are an expert HR consultant specializing in Bangladesh Labour Law. 
You have comprehensive knowledge of the Bangladesh Labour Act 2006 and its amendments up to 2018.
Provide accurate, professional advice to HR practitioners in Bangladesh.
When applicable, cite relevant sections of the Labour Act."""
    }
    
    conversation_history = [system_message]
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'clear':
            conversation_history = [system_message]
            print("\nConversation cleared.")
            continue
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        print("\nAssistant: ", end="", flush=True)
        
        if backend == 'huggingface':
            response = generate_response_hf(
                model, tokenizer,
                conversation_history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:  # ollama
            response = generate_response_ollama(
                model_name,
                conversation_history,
                temperature=temperature,
            )
        
        if response:
            print(response)
            
            # Add assistant response to history
            conversation_history.append({
                "role": "assistant",
                "content": response
            })
        else:
            print("[Error generating response]")
            # Remove the failed user message
            conversation_history.pop()


def single_query(
    backend: str,
    query: str,
    model=None,
    tokenizer=None,
    model_name: str = None,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> str:
    """
    Process a single query.
    
    Args:
        backend: Either 'huggingface' or 'ollama'
        query: The user's question
        model: HuggingFace model (if backend is 'huggingface')
        tokenizer: HuggingFace tokenizer (if backend is 'huggingface')
        model_name: Ollama model name (if backend is 'ollama')
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        The model's response
    """
    messages = [
        {
            "role": "system",
            "content": """You are an expert HR consultant specializing in Bangladesh Labour Law. 
You have comprehensive knowledge of the Bangladesh Labour Act 2006 and its amendments.
Provide accurate, professional advice to HR practitioners."""
        },
        {
            "role": "user",
            "content": query
        }
    ]
    
    if backend == 'huggingface':
        return generate_response_hf(
            model, tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:  # ollama
        return generate_response_ollama(
            model_name,
            messages,
            temperature=temperature,
        )


def batch_inference(
    backend: str,
    input_file: str,
    output_file: str,
    model=None,
    tokenizer=None,
    model_name: str = None,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
):
    """
    Process a batch of queries from a JSON file.
    
    Args:
        backend: Either 'huggingface' or 'ollama'
        input_file: Path to JSON file with queries
        output_file: Path to save results
        model: HuggingFace model (if backend is 'huggingface')
        tokenizer: HuggingFace tokenizer (if backend is 'huggingface')
        model_name: Ollama model name (if backend is 'ollama')
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
    """
    from tqdm import tqdm
    
    # Load queries
    with open(input_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    if not isinstance(queries, list):
        queries = [queries]
    
    results = []
    
    print(f"Processing {len(queries)} queries...")
    
    for item in tqdm(queries):
        # Get query from various possible formats
        if isinstance(item, str):
            query = item
        elif isinstance(item, dict):
            query = item.get('query') or item.get('question') or item.get('input')
        else:
            continue
        
        if not query:
            continue
        
        response = single_query(
            backend=backend,
            query=query,
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        
        results.append({
            'query': query,
            'response': response
        })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on fine-tuned HR Persona BD models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive chat with local HuggingFace model
    python scripts/inference.py --model ./hr-persona-bd-llama32-3b-lora --interactive
    
    # Single query
    python scripts/inference.py --model ./model --query "What is maternity leave duration?"
    
    # Use Ollama backend
    python scripts/inference.py --backend ollama --model hr-persona-bd-llama --interactive
    
    # Batch inference
    python scripts/inference.py --model ./model --batch queries.json --output results.json
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model path (HuggingFace) or model name (Ollama)"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["huggingface", "ollama"],
        default="huggingface",
        help="Inference backend (default: huggingface)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive chat mode"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single query to process"
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Path to JSON file with batch queries"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="inference_results.json",
        help="Output file for batch inference (default: inference_results.json)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (HuggingFace only)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.query and not args.batch:
        print("Error: Must specify --interactive, --query, or --batch")
        parser.print_help()
        sys.exit(1)
    
    # Initialize model
    model = None
    tokenizer = None
    
    if args.backend == 'huggingface':
        # Check if model path exists
        if not Path(args.model).exists():
            print(f"Error: Model path not found: {args.model}")
            sys.exit(1)
        
        model, tokenizer = load_huggingface_model(
            args.model,
            load_in_4bit=not args.no_4bit
        )
        print("Model loaded successfully!")
    else:  # ollama
        # Test Ollama connection
        try:
            import ollama
            ollama.list()
            print(f"Using Ollama model: {args.model}")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            sys.exit(1)
    
    # Run appropriate mode
    if args.interactive:
        interactive_chat(
            backend=args.backend,
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
        )
    elif args.query:
        response = single_query(
            backend=args.backend,
            query=args.query,
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
        )
        print(f"\nQuery: {args.query}")
        print(f"\nResponse: {response}")
    elif args.batch:
        if not Path(args.batch).exists():
            print(f"Error: Batch file not found: {args.batch}")
            sys.exit(1)
        
        batch_inference(
            backend=args.backend,
            input_file=args.batch,
            output_file=args.output,
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
