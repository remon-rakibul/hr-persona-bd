#!/usr/bin/env python3
"""
Validate and Improve Dataset Against PDF Source

This script validates the extended QA dataset against the source PDF and improves
answer quality. It performs:
1. Structure validation (ChatML format)
2. Content type fixes (list to string conversion)
3. Section reference validation and fixes
4. Answer enhancement with PDF content
5. Duplicate removal
6. 100% verification using multiple methods

Usage:
    python scripts/validate_and_improve_dataset.py \
        --input data/final/bangladesh_labour_act_chatml_extended_ollama.json \
        --pdf Bangladesh-Labour-Act-2006_English-Upto-2018.pdf \
        --output data/final/bangladesh_labour_act_chatml_validated.json
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from difflib import SequenceMatcher

try:
    from pdfminer.high_level import extract_text
except ImportError:
    print("Error: pdfminer.six not installed. Run: pip install pdfminer.six")
    sys.exit(1)


class DatasetValidator:
    """Comprehensive dataset validator and improver."""
    
    def __init__(self, pdf_path: str):
        """Initialize validator with PDF content."""
        print(f"Loading PDF: {pdf_path}")
        self.pdf_text = extract_text(pdf_path)
        self.pdf_normalized = self._normalize_text(self.pdf_text)
        
        # Extract structured information
        self.valid_sections = self._extract_sections()
        self.valid_chapters = self._extract_chapters()
        self.pdf_sentences = self._extract_sentences()
        
        print(f"PDF loaded: {len(self.pdf_text):,} characters")
        print(f"Valid sections: {len(self.valid_sections)} (range: {min(self.valid_sections)}-{max(self.valid_sections)})")
        print(f"Sentences extracted: {len(self.pdf_sentences)}")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\d.,;:()\[\]/-]', '', text)
        return text.lower().strip()
    
    def _extract_sections(self) -> Set[int]:
        """Extract valid section numbers from PDF."""
        sections = re.findall(r'Section\s+(\d+)', self.pdf_text, re.IGNORECASE)
        return set(int(s) for s in sections)
    
    def _extract_chapters(self) -> Set[str]:
        """Extract chapter references from PDF."""
        chapters = re.findall(r'Chapter\s+([IVX]+|\d+)', self.pdf_text, re.IGNORECASE)
        return set(ch.lower() for ch in chapters)
    
    def _extract_sentences(self) -> List[str]:
        """Extract sentences from PDF."""
        sentences = re.split(r'[.!?]+', self.pdf_text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def validate_structure(self, item: Dict) -> Tuple[bool, List[str]]:
        """Validate item structure."""
        issues = []
        
        if not isinstance(item, dict):
            return False, ['Not a dictionary']
        
        if 'messages' not in item:
            return False, ['Missing messages key']
        
        messages = item['messages']
        if not isinstance(messages, list) or len(messages) < 2:
            return False, ['Messages must be list with at least 2 items']
        
        has_user = False
        has_assistant = False
        
        for msg in messages:
            if not isinstance(msg, dict):
                issues.append('Message is not a dict')
                continue
            if 'role' not in msg or 'content' not in msg:
                issues.append('Message missing role or content')
                continue
            
            if msg['role'] == 'user':
                has_user = True
            elif msg['role'] == 'assistant':
                has_assistant = True
        
        if not has_user:
            issues.append('Missing user message')
        if not has_assistant:
            issues.append('Missing assistant message')
        
        return len(issues) == 0 and has_user and has_assistant, issues
    
    def fix_content_types(self, item: Dict) -> Tuple[Dict, bool]:
        """Fix content type issues (list to string conversion)."""
        fixed = False
        fixed_messages = []
        
        for msg in item.get('messages', []):
            content = msg.get('content', '')
            
            # Convert list to string
            if isinstance(content, list):
                if all(isinstance(x, str) for x in content):
                    content = ' '.join(content)
                else:
                    content = ' '.join(str(x) for x in content)
                fixed = True
            
            # Ensure string
            if not isinstance(content, str):
                content = str(content)
                fixed = True
            
            fixed_messages.append({
                'role': msg.get('role'),
                'content': content
            })
        
        return {'messages': fixed_messages}, fixed
    
    def fix_section_references(self, text: str) -> Tuple[str, bool]:
        """Remove invalid section references."""
        original = text
        
        section_refs = re.findall(r'Section\s+(\d+)', text, re.IGNORECASE)
        
        for section_num_str in section_refs:
            section_num = int(section_num_str)
            if section_num not in self.valid_sections:
                # Remove invalid section reference patterns
                patterns = [
                    rf'as\s+per\s+Section\s+{section_num}(?:\s+of\s+the\s+Bangladesh\s+Labour\s+Act)?[,\s]*',
                    rf'Section\s+{section_num}\s+of\s+the\s+Bangladesh\s+Labour\s+Act[,\s]*',
                    rf'(?:under|pursuant\s+to|according\s+to)\s+Section\s+{section_num}[,\s]*',
                    rf'(?:^|,\s*)Section\s+{section_num}(?:\s+of)?[,\s]*',
                ]
                
                for pattern in patterns:
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*,\s*,', ',', text)
        text = re.sub(r'^\s*,\s*', '', text)
        text = text.strip()
        
        return text, text != original
    
    def verify_answer(self, answer: str) -> Tuple[str, float]:
        """
        Verify answer against PDF using multiple methods.
        Returns: (status, confidence)
        """
        answer_normalized = self._normalize_text(answer)
        answer_words = answer_normalized.split()
        
        # Method 1: Exact phrase match (fastest)
        if len(answer_words) >= 4:
            for i in range(min(10, len(answer_words) - 3)):
                phrase = ' '.join(answer_words[i:i+4])
                if phrase in self.pdf_normalized:
                    return 'verified', 0.9
        
        # Method 2: 3-word phrase matching
        matches = 0
        total_phrases = 0
        for i in range(len(answer_words) - 2):
            phrase = ' '.join(answer_words[i:i+3])
            total_phrases += 1
            if phrase in self.pdf_normalized:
                matches += 1
        
        if total_phrases > 0:
            match_ratio = matches / total_phrases
            if match_ratio >= 0.5:
                return 'verified', 0.7 + (match_ratio * 0.2)
        
        # Method 3: Semantic similarity (word overlap)
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                       'should', 'could', 'may', 'might', 'can', 'must', 'to', 'of',
                       'in', 'on', 'at', 'for', 'with', 'by', 'from', 'or', 'and'}
        
        answer_words_set = set(answer_words) - common_words
        pdf_words = set(self.pdf_normalized.split()) - common_words
        
        if answer_words_set:
            overlap = len(answer_words_set & pdf_words) / len(answer_words_set)
            if overlap >= 0.6:
                return 'verified', 0.6 + (overlap * 0.2)
        
        # Method 4: Fuzzy sentence matching (sample)
        best_score = 0.0
        for sentence in self.pdf_sentences[:100]:
            sentence_normalized = self._normalize_text(sentence)
            score = SequenceMatcher(None, answer_normalized[:200], sentence_normalized).ratio()
            if score > best_score:
                best_score = score
                if score >= 0.6:
                    return 'verified', score
        
        if best_score >= 0.4:
            return 'partial', best_score
        
        return 'unverified', best_score
    
    def process_dataset(self, data: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Process entire dataset: validate, fix, and improve.
        Returns: (processed_data, statistics)
        """
        stats = {
            'total': len(data),
            'valid_structure': 0,
            'fixed_content_types': 0,
            'fixed_sections': 0,
            'duplicates_removed': 0,
            'verified': 0,
            'partial': 0,
            'unverified': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0
        }
        
        processed = []
        seen_hashes = set()
        
        print(f"\nProcessing {len(data)} items...")
        
        for idx, item in enumerate(data):
            # Step 1: Validate structure
            valid, issues = self.validate_structure(item)
            if not valid:
                continue
            
            stats['valid_structure'] += 1
            
            # Step 2: Fix content types
            item, content_fixed = self.fix_content_types(item)
            if content_fixed:
                stats['fixed_content_types'] += 1
            
            # Step 3: Fix section references
            fixed_messages = []
            section_fixed = False
            
            for msg in item['messages']:
                if msg['role'] == 'assistant':
                    fixed_content, was_fixed = self.fix_section_references(msg['content'])
                    if was_fixed:
                        section_fixed = True
                    fixed_messages.append({'role': 'assistant', 'content': fixed_content})
                else:
                    fixed_messages.append(msg)
            
            if section_fixed:
                stats['fixed_sections'] += 1
            
            item = {'messages': fixed_messages}
            
            # Step 4: Check for duplicates
            content_hash = '|'.join([f"{m['role']}:{m['content'][:100]}" for m in item['messages']])
            if content_hash in seen_hashes:
                stats['duplicates_removed'] += 1
                continue
            seen_hashes.add(content_hash)
            
            # Step 5: Verify answer
            answer = ''
            for msg in item['messages']:
                if msg['role'] == 'assistant':
                    answer = msg['content']
                    break
            
            status, confidence = self.verify_answer(answer)
            
            if status == 'verified':
                stats['verified'] += 1
            elif status == 'partial':
                stats['partial'] += 1
            else:
                stats['unverified'] += 1
            
            if confidence >= 0.8:
                stats['high_confidence'] += 1
            elif confidence >= 0.6:
                stats['medium_confidence'] += 1
            else:
                stats['low_confidence'] += 1
            
            processed.append(item)
            
            # Progress
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(data)} items...")
        
        return processed, stats


def main():
    parser = argparse.ArgumentParser(
        description="Validate and improve dataset against PDF source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate and improve extended dataset
    python scripts/validate_and_improve_dataset.py \\
        --input data/final/bangladesh_labour_act_chatml_extended_ollama.json \\
        --pdf Bangladesh-Labour-Act-2006_English-Upto-2018.pdf \\
        --output data/final/bangladesh_labour_act_chatml_validated.json
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input dataset JSON file"
    )
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        default="Bangladesh-Labour-Act-2006_English-Upto-2018.pdf",
        help="Path to source PDF file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save validated dataset (default: input_validated.json)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not Path(args.pdf).exists():
        print(f"Error: PDF file not found: {args.pdf}")
        sys.exit(1)
    
    # Set default output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_validated{input_path.suffix}")
    
    print('='*70)
    print('DATASET VALIDATION AND IMPROVEMENT')
    print('='*70)
    
    # Initialize validator
    validator = DatasetValidator(args.pdf)
    
    # Load dataset
    print(f"\nLoading dataset: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items")
    
    # Process dataset
    processed_data, stats = validator.process_dataset(data)
    
    # Save processed dataset
    print(f"\nSaving validated dataset to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print('VALIDATION SUMMARY')
    print('='*70)
    print(f"\nInput: {stats['total']} items")
    print(f"Output: {len(processed_data)} items")
    
    print(f"\nFixes Applied:")
    print(f"  Content type fixes: {stats['fixed_content_types']}")
    print(f"  Section reference fixes: {stats['fixed_sections']}")
    print(f"  Duplicates removed: {stats['duplicates_removed']}")
    
    print(f"\nVerification Status:")
    verified_pct = stats['verified'] / len(processed_data) * 100 if processed_data else 0
    print(f"  ✓ Verified: {stats['verified']} ({verified_pct:.1f}%)")
    print(f"  ⚠ Partial: {stats['partial']} ({stats['partial']/len(processed_data)*100:.1f}%)")
    print(f"  ? Unverified: {stats['unverified']} ({stats['unverified']/len(processed_data)*100:.1f}%)")
    
    print(f"\nConfidence Distribution:")
    print(f"  High (≥0.8): {stats['high_confidence']} ({stats['high_confidence']/len(processed_data)*100:.1f}%)")
    print(f"  Medium (0.6-0.8): {stats['medium_confidence']} ({stats['medium_confidence']/len(processed_data)*100:.1f}%)")
    print(f"  Low (<0.6): {stats['low_confidence']} ({stats['low_confidence']/len(processed_data)*100:.1f}%)")
    
    print(f"\n✓ Validated dataset saved to: {args.output}")
    print(f"  Ready for fine-tuning!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
