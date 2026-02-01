#!/usr/bin/env python3
"""
Extend QA Dataset using OpenAI GPT-4o-mini API
Generates additional QA pairs and multi-turn conversations from existing dataset.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python scripts/extend_dataset_openai.py --input data/final/bangladesh_labour_act_chatml.json
    python scripts/extend_dataset_openai.py --input data/generated/*.json --model gpt-4o
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
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)


# System prompt for HR Bangladesh context
SYSTEM_PROMPT = """You are an expert HR consultant specializing in Bangladesh Labour Law and workplace practices. 
You have comprehensive knowledge of the Bangladesh Labour Act 2006 (amended up to 2018) and its practical applications.
Your responses should be accurate, professional, and helpful for HR practitioners in Bangladesh.
Always provide practical, actionable advice based on the legal framework."""

# Prompts for different types of dataset extension
EXTENSION_PROMPTS = {
    "variations": """Based on this question-answer pair about Bangladesh Labour Law:

Question: {question}
Answer: {answer}

Generate 3 variations of this question that ask about the same topic but from different angles or with different wording. 
For each variation, provide an appropriate answer that is accurate according to Bangladesh Labour Act 2006.

Return ONLY valid JSON in this format:
[
    {{"question": "variation 1", "answer": "answer 1"}},
    {{"question": "variation 2", "answer": "answer 2"}},
    {{"question": "variation 3", "answer": "answer 3"}}
]""",

    "follow_up": """Based on this question-answer pair about Bangladesh Labour Law:

Question: {question}
Answer: {answer}

Generate 2 natural follow-up questions that an HR professional might ask after receiving this answer, along with detailed appropriate answers.

Return ONLY valid JSON in this format:
[
    {{"question": "follow-up question 1", "answer": "detailed answer 1"}},
    {{"question": "follow-up question 2", "answer": "detailed answer 2"}}
]""",

    "scenarios": """Based on this information from Bangladesh Labour Law:

Question: {question}
Answer: {answer}

Create 2 practical scenario-based questions that HR professionals might encounter in real workplace situations in Bangladesh.
Include specific details like company size, industry (RMG, IT, manufacturing, etc.), or employee circumstances.
Answers should reference relevant sections of the Bangladesh Labour Act when applicable.

Return ONLY valid JSON in this format:
[
    {{"question": "Scenario question 1 with specific details", "answer": "Detailed answer with legal references"}},
    {{"question": "Scenario question 2 with specific details", "answer": "Detailed answer with legal references"}}
]""",

    "multi_turn": """Based on this question-answer pair about Bangladesh Labour Law:

Question: {question}
Answer: {answer}

Create a natural 4-turn conversation between an HR professional seeking advice and an AI assistant about this topic.
The conversation should:
1. Start with a related but different question
2. Progress naturally with follow-ups
3. Include practical advice and legal references
4. Be professional and informative

Return ONLY valid JSON in this format:
{{
    "conversations": [
        {{"from": "human", "value": "initial question"}},
        {{"from": "gpt", "value": "helpful response"}},
        {{"from": "human", "value": "follow-up question"}},
        {{"from": "gpt", "value": "detailed response with practical advice"}}
    ]
}}""",

    "edge_cases": """Based on this question-answer pair about Bangladesh Labour Law:

Question: {question}
Answer: {answer}

Generate 2 edge case or complex scenario questions related to this topic. These should cover:
- Unusual but legally valid situations
- Conflicts between different provisions
- Practical implementation challenges
- Common misunderstandings about the law

Return ONLY valid JSON in this format:
[
    {{"question": "Edge case question 1", "answer": "Nuanced answer addressing complexity"}},
    {{"question": "Edge case question 2", "answer": "Nuanced answer addressing complexity"}}
]"""
}


def get_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client with API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Run: export OPENAI_API_KEY='your-api-key'")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        # Test connection
        client.models.list()
        return client
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return None


def generate_with_openai(
    client: OpenAI,
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_retries: int = 3
) -> Optional[str]:
    """
    Generate text using OpenAI API.
    
    Args:
        client: OpenAI client instance
        prompt: The prompt to send to the model
        model: OpenAI model name
        temperature: Sampling temperature
        max_retries: Number of retries on failure
        
    Returns:
        Generated text or None if failed
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2048,
                response_format={"type": "json_object"} if "json" in prompt.lower() else None
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if "rate_limit" in str(e).lower():
                wait_time = 2 ** (attempt + 2)  # Longer wait for rate limits
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    return None


def parse_json_response(response: str) -> Optional[Any]:
    """Parse JSON from LLM response, handling common issues."""
    if not response:
        return None
    
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
    client: OpenAI,
    qa: Dict,
    model: str,
    extension_types: List[str],
    temperature: float = 0.7
) -> List[Dict]:
    """
    Extend a single QA pair with multiple extension types.
    
    Args:
        client: OpenAI client instance
        qa: Question-answer pair
        model: OpenAI model name
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
        
        response = generate_with_openai(client, prompt, model, temperature)
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


def estimate_cost(num_samples: int, extension_types: List[str], model: str) -> float:
    """Estimate API cost for the extension job."""
    # Approximate tokens per extension type
    tokens_per_type = {
        "variations": 800,
        "follow_up": 600,
        "scenarios": 900,
        "multi_turn": 700,
        "edge_cases": 800
    }
    
    # GPT-4o-mini pricing (as of 2024)
    prices = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }
    
    price = prices.get(model, prices["gpt-4o-mini"])
    
    total_tokens = sum(tokens_per_type.get(t, 700) for t in extension_types)
    total_tokens *= num_samples
    
    # Rough estimate: 30% input, 70% output
    input_cost = (total_tokens * 0.3 / 1000) * price["input"]
    output_cost = (total_tokens * 0.7 / 1000) * price["output"]
    
    return input_cost + output_cost


def extend_dataset(
    input_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    extension_types: List[str] = None,
    max_samples: Optional[int] = None,
    temperature: float = 0.7,
    batch_delay: float = 0.5
) -> Optional[str]:
    """
    Extend a QA dataset with additional generated content.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to save extended dataset
        model: OpenAI model name
        extension_types: Types of extensions to generate
        max_samples: Maximum number of samples to process (None for all)
        temperature: Sampling temperature
        batch_delay: Delay between API calls (seconds)
        
    Returns:
        Path to the extended dataset
    """
    if extension_types is None:
        extension_types = ["variations", "follow_up", "scenarios"]
    
    print(f"\n{'='*60}")
    print("HR Persona BD - Dataset Extension with OpenAI")
    print('='*60)
    print(f"Input: {input_path}")
    print(f"Model: {model}")
    print(f"Extensions: {', '.join(extension_types)}")
    
    # Initialize OpenAI client
    client = get_openai_client()
    if not client:
        return None
    
    # Load and extract QA pairs
    print("\nLoading dataset...")
    data = load_dataset(input_path)
    qa_pairs = extract_qa_pairs(data)
    print(f"Found {len(qa_pairs)} QA pairs")
    
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
        print(f"Processing first {max_samples} samples")
    
    # Estimate cost
    estimated_cost = estimate_cost(len(qa_pairs), extension_types, model)
    print(f"\nEstimated cost: ${estimated_cost:.4f}")
    
    # Confirm before proceeding
    response = input("Proceed? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return None
    
    # Extend each QA pair
    all_extended = []
    original_count = len(qa_pairs)
    
    print("\nGenerating extensions...")
    for qa in tqdm(qa_pairs, desc="Extending"):
        # Keep original
        all_extended.append(qa)
        
        # Generate extensions
        extensions = extend_single_qa(client, qa, model, extension_types, temperature)
        all_extended.extend(extensions)
        
        # Rate limiting delay
        time.sleep(batch_delay)
    
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
        description="Extend QA dataset using OpenAI GPT-4o-mini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Set API key first
    export OPENAI_API_KEY="your-api-key"
    
    # Basic extension with gpt-4o-mini
    python scripts/extend_dataset_openai.py --input data/final/dataset.json
    
    # Use GPT-4o for higher quality
    python scripts/extend_dataset_openai.py --input data/final/dataset.json --model gpt-4o
    
    # Generate all extension types including edge cases
    python scripts/extend_dataset_openai.py --input data/final/dataset.json --types variations follow_up scenarios edge_cases
    
    # Process only first 50 samples
    python scripts/extend_dataset_openai.py --input data/final/dataset.json --max-samples 50
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
        help="Path to save extended dataset (default: data/final/extended_openai.json)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        help="OpenAI model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--types", "-t",
        nargs="+",
        choices=["variations", "follow_up", "scenarios", "multi_turn", "edge_cases"],
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
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        input_name = Path(args.input).stem
        args.output = f"data/final/{input_name}_extended_openai.json"
    
    # Check input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Run: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Run extension
    result = extend_dataset(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        extension_types=args.types,
        max_samples=args.max_samples,
        temperature=args.temperature,
        batch_delay=args.delay
    )
    
    if result:
        print(f"\n✓ Success! Extended dataset saved to: {result}")
        sys.exit(0)
    else:
        print("\n✗ Extension failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
