#!/usr/bin/env python3
"""
Extend QA Dataset using Ollama Local LLM
Generates additional QA pairs and multi-turn conversations from existing dataset.

Usage:
    python scripts/extend_dataset_ollama.py --input data/final/bangladesh_labour_act_chatml.json
    python scripts/extend_dataset_ollama.py --input data/generated/*.json --model llama3.3:70b
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
from tqdm import tqdm

try:
    import ollama
except ImportError:
    print("Error: ollama package not installed. Run: pip install ollama")
    sys.exit(1)


# System prompt for HR Bangladesh context
SYSTEM_PROMPT = """You are an expert HR consultant specializing in Bangladesh Labour Law and workplace practices. 
You have comprehensive knowledge of the Bangladesh Labour Act 2006 (amended up to 2018) and its practical applications.
Your responses should be accurate, professional, and helpful for HR practitioners in Bangladesh."""

# Prompts for different types of dataset extension
EXTENSION_PROMPTS = {
    "variations": """Based on this question-answer pair about Bangladesh Labour Law:

Question: {question}
Answer: {answer}

Generate 3 variations of this question that ask about the same topic but from different angles or with different wording. 
For each variation, provide an appropriate answer.

Return ONLY valid JSON in this format:
[
    {{"question": "variation 1", "answer": "answer 1"}},
    {{"question": "variation 2", "answer": "answer 2"}},
    {{"question": "variation 3", "answer": "answer 3"}}
]""",

    "follow_up": """Based on this question-answer pair about Bangladesh Labour Law:

Question: {question}
Answer: {answer}

Generate 2 natural follow-up questions that someone might ask after receiving this answer, along with appropriate answers.

Return ONLY valid JSON in this format:
[
    {{"question": "follow-up question 1", "answer": "answer 1"}},
    {{"question": "follow-up question 2", "answer": "answer 2"}}
]""",

    "scenarios": """Based on this information from Bangladesh Labour Law:

Question: {question}
Answer: {answer}

Create 2 practical scenario-based questions that HR professionals might encounter in real workplace situations in Bangladesh.
Include specific details like company size, industry, or employee circumstances.

Return ONLY valid JSON in this format:
[
    {{"question": "Scenario question 1", "answer": "Detailed answer 1"}},
    {{"question": "Scenario question 2", "answer": "Detailed answer 2"}}
]""",

    "multi_turn": """Based on this question-answer pair about Bangladesh Labour Law:

Question: {question}
Answer: {answer}

Create a natural 4-turn conversation between an HR professional and an AI assistant about this topic.
The conversation should start with a related question and progress naturally.

Return ONLY valid JSON in this format:
{{
    "conversations": [
        {{"from": "human", "value": "initial question"}},
        {{"from": "gpt", "value": "response"}},
        {{"from": "human", "value": "follow-up question"}},
        {{"from": "gpt", "value": "detailed response"}}
    ]
}}"""
}


def check_ollama_connection(model: str) -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        # Try to list models
        models = ollama.list()
        available_models = [m['name'] for m in models.get('models', [])]
        
        # Check if our model is available (handle partial matches)
        model_base = model.split(':')[0]
        if not any(model_base in m for m in available_models):
            print(f"Warning: Model '{model}' not found. Available models:")
            for m in available_models:
                print(f"  - {m}")
            print(f"\nTrying to pull {model}...")
            ollama.pull(model)
        return True
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return False


def generate_with_ollama(
    prompt: str,
    model: str = "llama3.2:3b-instruct-q4_K_M",
    temperature: float = 0.7,
    max_retries: int = 3
) -> Optional[str]:
    """
    Generate text using Ollama.
    
    Args:
        prompt: The prompt to send to the model
        model: Ollama model name
        temperature: Sampling temperature
        max_retries: Number of retries on failure
        
    Returns:
        Generated text or None if failed
    """
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": temperature,
                    "num_predict": 2048,
                }
            )
            return response['message']['content']
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    return None


def parse_json_response(response: str) -> Optional[Any]:
    """Parse JSON from LLM response, handling common issues."""
    if not response:
        return None
    
    # Try to extract JSON from the response
    response = response.strip()
    
    # Try direct parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in the response
    import re
    
    # Look for JSON array
    array_match = re.search(r'\[[\s\S]*\]', response)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass
    
    # Look for JSON object
    obj_match = re.search(r'\{[\s\S]*\}', response)
    if obj_match:
        try:
            return json.loads(obj_match.group())
        except json.JSONDecodeError:
            pass
    
    return None


def load_dataset(input_path: str) -> List[Dict]:
    """Load QA dataset from JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'data' in data:
        return data['data']
    else:
        return [data]


def extract_qa_pairs(data: List[Dict]) -> List[Dict]:
    """Extract question-answer pairs from various formats."""
    qa_pairs = []
    
    for item in data:
        # Direct QA format
        if 'question' in item and 'answer' in item:
            qa_pairs.append({
                'question': item['question'],
                'answer': item['answer']
            })
        # ChatML format
        elif 'messages' in item:
            messages = item['messages']
            for i in range(0, len(messages) - 1, 2):
                if messages[i].get('role') == 'user' and messages[i+1].get('role') == 'assistant':
                    qa_pairs.append({
                        'question': messages[i]['content'],
                        'answer': messages[i+1]['content']
                    })
        # ShareGPT format
        elif 'conversations' in item:
            convs = item['conversations']
            for i in range(0, len(convs) - 1, 2):
                if convs[i].get('from') == 'human' and convs[i+1].get('from') == 'gpt':
                    qa_pairs.append({
                        'question': convs[i]['value'],
                        'answer': convs[i+1]['value']
                    })
    
    return qa_pairs


def extend_single_qa(
    qa: Dict,
    model: str,
    extension_types: List[str],
    temperature: float = 0.7
) -> List[Dict]:
    """
    Extend a single QA pair with multiple extension types.
    
    Args:
        qa: Question-answer pair
        model: Ollama model name
        extension_types: List of extension types to apply
        temperature: Sampling temperature
        
    Returns:
        List of new QA pairs/conversations
    """
    extended = []
    
    for ext_type in extension_types:
        if ext_type not in EXTENSION_PROMPTS:
            continue
        
        prompt = EXTENSION_PROMPTS[ext_type].format(
            question=qa['question'],
            answer=qa['answer']
        )
        
        response = generate_with_ollama(prompt, model, temperature)
        parsed = parse_json_response(response)
        
        if parsed:
            if isinstance(parsed, list):
                extended.extend(parsed)
            elif isinstance(parsed, dict):
                extended.append(parsed)
    
    return extended


def convert_to_chatml(qa_pairs: List[Dict]) -> List[Dict]:
    """Convert QA pairs to ChatML format."""
    chatml_data = []
    
    for item in qa_pairs:
        # Already in conversation format
        if 'conversations' in item:
            # Convert ShareGPT to ChatML
            messages = []
            for conv in item['conversations']:
                role = 'user' if conv['from'] == 'human' else 'assistant'
                messages.append({
                    'role': role,
                    'content': conv['value']
                })
            chatml_data.append({'messages': messages})
        # Simple QA format
        elif 'question' in item and 'answer' in item:
            chatml_data.append({
                'messages': [
                    {'role': 'user', 'content': item['question']},
                    {'role': 'assistant', 'content': item['answer']}
                ]
            })
    
    return chatml_data


def extend_dataset(
    input_path: str,
    output_path: str,
    model: str = "llama3.2:3b-instruct-q4_K_M",
    extension_types: List[str] = None,
    max_samples: Optional[int] = None,
    temperature: float = 0.7
) -> str:
    """
    Extend a QA dataset with additional generated content.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to save extended dataset
        model: Ollama model name
        extension_types: Types of extensions to generate
        max_samples: Maximum number of samples to process (None for all)
        temperature: Sampling temperature
        
    Returns:
        Path to the extended dataset
    """
    if extension_types is None:
        extension_types = ["variations", "follow_up", "scenarios"]
    
    print(f"\n{'='*60}")
    print("HR Persona BD - Dataset Extension with Ollama")
    print('='*60)
    print(f"Input: {input_path}")
    print(f"Model: {model}")
    print(f"Extensions: {', '.join(extension_types)}")
    
    # Check Ollama connection
    if not check_ollama_connection(model):
        return None
    
    # Load and extract QA pairs
    print("\nLoading dataset...")
    data = load_dataset(input_path)
    qa_pairs = extract_qa_pairs(data)
    print(f"Found {len(qa_pairs)} QA pairs")
    
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
        print(f"Processing first {max_samples} samples")
    
    # Extend each QA pair
    all_extended = []
    original_count = len(qa_pairs)
    
    print("\nGenerating extensions...")
    for qa in tqdm(qa_pairs, desc="Extending"):
        # Keep original
        all_extended.append(qa)
        
        # Generate extensions
        extensions = extend_single_qa(qa, model, extension_types, temperature)
        all_extended.extend(extensions)
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    # Convert to ChatML format
    chatml_data = convert_to_chatml(all_extended)
    
    # Save extended dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chatml_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Extension Complete!")
    print('='*60)
    print(f"Original pairs: {original_count}")
    print(f"Extended pairs: {len(chatml_data)}")
    print(f"Saved to: {output_path}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extend QA dataset using Ollama local LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic extension with default model
    python scripts/extend_dataset_ollama.py --input data/final/dataset.json
    
    # Use larger model for better quality
    python scripts/extend_dataset_ollama.py --input data/final/dataset.json --model llama3.3:70b
    
    # Generate only variations and scenarios
    python scripts/extend_dataset_ollama.py --input data/final/dataset.json --types variations scenarios
    
    # Process only first 100 samples
    python scripts/extend_dataset_ollama.py --input data/final/dataset.json --max-samples 100
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input QA dataset JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save extended dataset (default: data/final/extended_ollama.json)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama3.2:3b-instruct-q4_K_M",
        help="Ollama model name (default: llama3.2:3b-instruct-q4_K_M)"
    )
    parser.add_argument(
        "--types", "-t",
        nargs="+",
        choices=["variations", "follow_up", "scenarios", "multi_turn"],
        default=["variations", "follow_up", "scenarios"],
        help="Types of extensions to generate"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        input_name = Path(args.input).stem
        args.output = f"data/final/{input_name}_extended_ollama.json"
    
    # Check input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Run extension
    result = extend_dataset(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        extension_types=args.types,
        max_samples=args.max_samples,
        temperature=args.temperature
    )
    
    if result:
        print(f"\n✓ Success! Extended dataset saved to: {result}")
        sys.exit(0)
    else:
        print("\n✗ Extension failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
