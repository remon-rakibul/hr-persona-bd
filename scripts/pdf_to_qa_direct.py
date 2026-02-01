#!/usr/bin/env python3
"""
Direct PDF to QA Conversion using Ollama (Bypass synthetic-data-kit)

This script directly uses Ollama to generate QA pairs from PDF or text files.
It's a simpler alternative when synthetic-data-kit has issues.

Usage:
    python scripts/pdf_to_qa_direct.py --input data/input/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf
    python scripts/pdf_to_qa_direct.py --input data/output/document.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import time
from tqdm import tqdm

try:
    import ollama
except ImportError:
    print("Error: ollama package not installed. Run: pip install ollama")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    try:
        from pdfminer.high_level import extract_text
        print(f"Extracting text from PDF: {pdf_path}")
        text = extract_text(pdf_path)
        print(f"Extracted {len(text)} characters")
        return text
    except ImportError:
        print("Error: pdfminer.six not installed. Run: pip install pdfminer.six")
        sys.exit(1)
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        sys.exit(1)


SYSTEM_PROMPT = """You are an expert in Bangladesh Labour Law and HR practices. You are creating question-answer pairs for training an HR assistant chatbot."""

QA_GENERATION_PROMPT = """Based on the following text from the Bangladesh Labour Act 2006, create {num_pairs} high-quality question-answer pairs.

Focus on:
- Labour rights and worker protections
- Employment contracts and termination procedures
- Wages, benefits, and leave policies
- Workplace safety and health regulations
- Trade unions and collective bargaining

Requirements:
- Questions should be practical and relevant to HR professionals in Bangladesh
- Answers should be accurate and cite specific sections when applicable
- Include both simple factual questions and complex scenario-based questions

Text:
{text}

Return ONLY valid JSON formatted as:
[
  {{"question": "...", "answer": "..."}},
  ...
]
"""


def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        
        if end >= len(text):
            break
    
    return chunks


def generate_qa_from_chunk(chunk: str, num_pairs: int, model: str) -> List[Dict]:
    """Generate QA pairs from a text chunk using Ollama."""
    prompt = QA_GENERATION_PROMPT.format(text=chunk, num_pairs=num_pairs)
    
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.7,
                "num_predict": 2048,
            }
        )
        
        content = response['message']['content']
        
        # Try to parse JSON
        import re
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            qa_pairs = json.loads(json_match.group())
            return qa_pairs if isinstance(qa_pairs, list) else []
        
        return []
        
    except Exception as e:
        print(f"Error generating QA pairs: {e}")
        return []


def convert_to_chatml(qa_pairs: List[Dict]) -> List[Dict]:
    """Convert QA pairs to ChatML format."""
    chatml_data = []
    
    for qa in qa_pairs:
        if 'question' in qa and 'answer' in qa:
            chatml_data.append({
                "messages": [
                    {"role": "user", "content": qa['question']},
                    {"role": "assistant", "content": qa['answer']}
                ]
            })
    
    return chatml_data


def main():
    parser = argparse.ArgumentParser(description="Direct PDF/TXT to QA conversion using Ollama")
    parser.add_argument("--input", "-i", required=True, help="Input PDF or text file")
    parser.add_argument("--output", "-o", default="data/final/bangladesh_labour_act_chatml.json", 
                       help="Output file (default: data/final/bangladesh_labour_act_chatml.json)")
    parser.add_argument("--model", "-m", default="llama3.2:3b-instruct-q4_K_M",
                       help="Ollama model (default: llama3.2:3b-instruct-q4_K_M)")
    parser.add_argument("--chunk-size", type=int, default=4000, help="Chunk size (default: 4000)")
    parser.add_argument("--num-pairs", type=int, default=5, 
                       help="QA pairs per chunk (default: 5, recommend lower for better quality)")
    parser.add_argument("--max-chunks", type=int, default=None, 
                       help="Maximum chunks to process (default: all)")
    
    args = parser.parse_args()
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Check Ollama connection
    try:
        ollama.list()
    except Exception as e:
        print(f"Error: Cannot connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    
    print("=" * 60)
    print("HR Persona BD - Direct QA Generation with Ollama")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"QA pairs per chunk: {args.num_pairs}")
    
    # Read input file (PDF or TXT)
    if input_path.suffix.lower() == '.pdf':
        text = extract_text_from_pdf(args.input)
    else:
        print(f"\nReading {args.input}...")
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
    
    # Split into chunks
    print(f"Splitting into chunks...")
    chunks = chunk_text(text, args.chunk_size)
    
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    
    print(f"Processing {len(chunks)} chunks...")
    
    # Generate QA pairs
    all_qa_pairs = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Generating QA pairs")):
        qa_pairs = generate_qa_from_chunk(chunk, args.num_pairs, args.model)
        all_qa_pairs.extend(qa_pairs)
        time.sleep(0.1)  # Small delay to avoid overloading
    
    print(f"\nGenerated {len(all_qa_pairs)} QA pairs")
    
    # Convert to ChatML format
    chatml_data = convert_to_chatml(all_qa_pairs)
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chatml_data, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Total QA pairs: {len(chatml_data)}")
    print(f"Saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Review the dataset: less {output_path}")
    print(f"2. Extend dataset (optional): python scripts/extend_dataset_ollama.py --input {output_path}")
    print(f"3. Upload to Colab for training")


if __name__ == "__main__":
    main()
