#!/usr/bin/env python3
"""
PDF to QA Dataset Conversion Script
Uses Meta's Synthetic Data Kit to convert Bangladesh Labour Act PDF to QA pairs.

Usage:
    python scripts/pdf_to_qa.py --pdf data/input/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf
    python scripts/pdf_to_qa.py --pdf data/input/*.pdf --num-pairs 50
"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
from typing import Optional
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_command(cmd: list, description: str) -> bool:
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found. Make sure synthetic-data-kit is installed.")
        print("  Run: pip install synthetic-data-kit")
        return False


def ingest_pdf(pdf_path: str, output_dir: str = "data/parsed") -> Optional[str]:
    """
    Parse PDF and extract text using synthetic-data-kit.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save parsed text
        
    Returns:
        Path to the parsed text file, or None if failed
    """
    pdf_name = Path(pdf_path).stem
    
    cmd = [
        "synthetic-data-kit",
        "ingest",
        pdf_path,
    ]
    
    if run_command(cmd, f"Ingesting PDF: {pdf_name}"):
        # The output goes to data/parsed/ by default
        parsed_file = Path(output_dir) / f"{pdf_name}.lance"
        txt_file = Path(output_dir) / f"{pdf_name}.txt"
        
        # Check which format was created
        if parsed_file.exists():
            return str(parsed_file)
        elif txt_file.exists():
            return str(txt_file)
        else:
            # List files in output dir to help debug
            print(f"Looking for output in {output_dir}...")
            for f in Path(output_dir).glob("*"):
                print(f"  Found: {f}")
            return None
    return None


def generate_qa_pairs(
    text_path: str,
    num_pairs: int = 50,
    chunk_size: int = 4000,
    output_dir: str = "data/generated"
) -> Optional[str]:
    """
    Generate QA pairs from parsed text.
    
    Args:
        text_path: Path to the parsed text file
        num_pairs: Number of QA pairs to generate per chunk
        chunk_size: Size of text chunks for processing
        output_dir: Directory to save generated QA pairs
        
    Returns:
        Path to the generated QA pairs file, or None if failed
    """
    cmd = [
        "synthetic-data-kit",
        "create",
        text_path,
        "--type", "qa",
        "--num-pairs", str(num_pairs),
        "--chunk-size", str(chunk_size),
        "--verbose",
    ]
    
    if run_command(cmd, f"Generating QA pairs from {Path(text_path).name}"):
        # Find the output file
        text_name = Path(text_path).stem
        output_file = Path(output_dir) / f"{text_name}_qa_pairs.json"
        if output_file.exists():
            return str(output_file)
        # Try to find any JSON file that was created
        for f in Path(output_dir).glob(f"*{text_name}*.json"):
            return str(f)
    return None


def curate_qa_pairs(
    qa_path: str,
    threshold: float = 7.0,
    output_dir: str = "data/curated"
) -> Optional[str]:
    """
    Filter QA pairs by quality using LLM-as-judge.
    
    Args:
        qa_path: Path to the generated QA pairs file
        threshold: Minimum quality score (1-10)
        output_dir: Directory to save curated QA pairs
        
    Returns:
        Path to the curated QA pairs file, or None if failed
    """
    cmd = [
        "synthetic-data-kit",
        "curate",
        qa_path,
        "--threshold", str(threshold),
    ]
    
    if run_command(cmd, f"Curating QA pairs with threshold {threshold}"):
        qa_name = Path(qa_path).stem.replace("_qa_pairs", "")
        output_file = Path(output_dir) / f"{qa_name}_cleaned.json"
        if output_file.exists():
            return str(output_file)
        # Try to find any JSON file that was created
        for f in Path(output_dir).glob(f"*{qa_name}*.json"):
            return str(f)
    return None


def save_as_training_format(
    curated_path: str,
    format_type: str = "chatml",
    storage: str = "hf",
    output_dir: str = "data/final"
) -> Optional[str]:
    """
    Convert curated QA pairs to training format.
    
    Args:
        curated_path: Path to the curated QA pairs file
        format_type: Output format (chatml, alpaca, ft)
        storage: Storage format (json, hf for HuggingFace)
        output_dir: Directory to save final dataset
        
    Returns:
        Path to the final dataset file, or None if failed
    """
    cmd = [
        "synthetic-data-kit",
        "save-as",
        curated_path,
        "--format", format_type,
        "--storage", storage,
    ]
    
    if run_command(cmd, f"Saving as {format_type} format"):
        curated_name = Path(curated_path).stem.replace("_cleaned", "")
        # Check for output files
        for f in Path(output_dir).glob(f"*{curated_name}*"):
            return str(f)
    return None


def convert_to_sharegpt_format(qa_pairs: list) -> list:
    """
    Convert simple QA pairs to ShareGPT conversation format.
    
    Args:
        qa_pairs: List of {"question": ..., "answer": ...} dicts
        
    Returns:
        List of conversations in ShareGPT format
    """
    conversations = []
    for qa in qa_pairs:
        conv = {
            "conversations": [
                {"from": "human", "value": qa["question"]},
                {"from": "gpt", "value": qa["answer"]}
            ]
        }
        conversations.append(conv)
    return conversations


def convert_to_chatml_format(qa_pairs: list) -> list:
    """
    Convert simple QA pairs to ChatML format for Unsloth.
    
    Args:
        qa_pairs: List of {"question": ..., "answer": ...} dicts
        
    Returns:
        List of messages in ChatML format
    """
    conversations = []
    for qa in qa_pairs:
        conv = {
            "messages": [
                {"role": "user", "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]}
            ]
        }
        conversations.append(conv)
    return conversations


def manual_conversion(
    qa_path: str,
    output_dir: str = "data/final",
    output_format: str = "chatml"
) -> str:
    """
    Manually convert QA pairs to training format if synthetic-data-kit fails.
    
    Args:
        qa_path: Path to JSON file with QA pairs
        output_dir: Directory to save output
        output_format: Format type (sharegpt or chatml)
        
    Returns:
        Path to the converted file
    """
    print(f"\nManually converting {qa_path} to {output_format} format...")
    
    with open(qa_path, "r") as f:
        qa_pairs = json.load(f)
    
    if output_format == "sharegpt":
        converted = convert_to_sharegpt_format(qa_pairs)
    else:  # chatml
        converted = convert_to_chatml_format(qa_pairs)
    
    output_path = Path(output_dir) / f"bangladesh_labour_act_{output_format}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(converted)} conversations to {output_path}")
    return str(output_path)


def full_pipeline(
    pdf_path: str,
    num_pairs: int = 50,
    chunk_size: int = 4000,
    quality_threshold: float = 7.0,
    output_format: str = "chatml",
    config_path: str = "configs/config.yaml"
) -> Optional[str]:
    """
    Run the complete PDF to training dataset pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        num_pairs: Number of QA pairs per chunk
        chunk_size: Size of text chunks
        quality_threshold: Minimum quality score for curation
        output_format: Output format (chatml, alpaca, ft)
        config_path: Path to configuration file
        
    Returns:
        Path to the final training dataset, or None if failed
    """
    print("\n" + "="*60)
    print("HR Persona BD - PDF to QA Dataset Pipeline")
    print("="*60)
    print(f"PDF: {pdf_path}")
    print(f"QA pairs per chunk: {num_pairs}")
    print(f"Chunk size: {chunk_size}")
    print(f"Quality threshold: {quality_threshold}")
    print(f"Output format: {output_format}")
    
    # Load config
    config = load_config(config_path)
    
    # Step 1: Ingest PDF
    parsed_path = ingest_pdf(pdf_path, config["paths"]["parsed_dir"])
    if not parsed_path:
        print("✗ Failed to ingest PDF")
        return None
    
    # Step 2: Generate QA pairs
    qa_path = generate_qa_pairs(
        parsed_path,
        num_pairs=num_pairs,
        chunk_size=chunk_size,
        output_dir=config["paths"]["generated_dir"]
    )
    if not qa_path:
        print("✗ Failed to generate QA pairs")
        return None
    
    # Step 3: Curate QA pairs
    curated_path = curate_qa_pairs(
        qa_path,
        threshold=quality_threshold,
        output_dir=config["paths"]["curated_dir"]
    )
    if not curated_path:
        print("⚠ Curation step failed, using uncurated QA pairs")
        curated_path = qa_path
    
    # Step 4: Save as training format
    final_path = save_as_training_format(
        curated_path,
        format_type=output_format,
        output_dir=config["paths"]["final_dir"]
    )
    
    if not final_path:
        # Try manual conversion as fallback
        print("⚠ save-as step failed, attempting manual conversion...")
        final_path = manual_conversion(
            curated_path if curated_path else qa_path,
            output_dir=config["paths"]["final_dir"],
            output_format=output_format
        )
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"Final dataset saved to: {final_path}")
    
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to QA dataset for fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/pdf_to_qa.py --pdf data/input/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf
    
    # Generate more QA pairs
    python scripts/pdf_to_qa.py --pdf data/input/*.pdf --num-pairs 100
    
    # Lower quality threshold for more pairs
    python scripts/pdf_to_qa.py --pdf data/input/doc.pdf --threshold 5.0
    
    # Use ShareGPT format instead of ChatML
    python scripts/pdf_to_qa.py --pdf data/input/doc.pdf --format sharegpt
        """
    )
    
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        required=True,
        help="Path to PDF file to convert"
    )
    parser.add_argument(
        "--num-pairs", "-n",
        type=int,
        default=50,
        help="Number of QA pairs to generate per chunk (default: 50)"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=4000,
        help="Size of text chunks for processing (default: 4000)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=7.0,
        help="Quality threshold for curation (1-10, default: 7.0)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["chatml", "sharegpt", "alpaca", "ft"],
        default="chatml",
        help="Output format for training (default: chatml)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Check if PDF exists
    if not Path(args.pdf).exists():
        print(f"Error: PDF file not found: {args.pdf}")
        sys.exit(1)
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"Warning: Config file not found: {args.config}")
        print("Using default configuration...")
    
    # Run pipeline
    result = full_pipeline(
        pdf_path=args.pdf,
        num_pairs=args.num_pairs,
        chunk_size=args.chunk_size,
        quality_threshold=args.threshold,
        output_format=args.format,
        config_path=args.config
    )
    
    if result:
        print(f"\n✓ Success! Dataset saved to: {result}")
        sys.exit(0)
    else:
        print("\n✗ Pipeline failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
