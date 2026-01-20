#!/usr/bin/env python3
"""
Inference script for dialogue summarization.

This script provides command-line interface for generating summaries
from dialogues using a trained model.

Usage:
    # Summarize a single dialogue from command line
    python scripts/inference.py --checkpoint ./checkpoints/best_model \
        --dialogue "Alice: Hey, how are you?\nBob: I'm good, thanks!"
    
    # Summarize dialogues from a file
    python scripts/inference.py --checkpoint ./checkpoints/best_model \
        --input_file dialogues.txt --output_file summaries.txt
    
    # Interactive mode
    python scripts/inference.py --checkpoint ./checkpoints/best_model --interactive
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.summarizer import DialogueSummarizer, SummarizationPipeline


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def load_dialogues_from_file(file_path: str) -> List[str]:
    """
    Load dialogues from a file.
    
    Supports:
    - Plain text (one dialogue per line, or separated by blank lines)
    - JSON (list of strings or list of objects with 'dialogue' key)
    - JSONL (one JSON object per line)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    content = file_path.read_text(encoding='utf-8')
    
    # Try JSON format first
    if file_path.suffix == '.json':
        data = json.loads(content)
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return data
            elif all(isinstance(item, dict) for item in data):
                return [item.get('dialogue', item.get('text', '')) for item in data]
        raise ValueError("JSON file must contain a list of strings or objects")
    
    # Try JSONL format
    if file_path.suffix == '.jsonl':
        dialogues = []
        for line in content.strip().split('\n'):
            if line.strip():
                item = json.loads(line)
                if isinstance(item, str):
                    dialogues.append(item)
                elif isinstance(item, dict):
                    dialogues.append(item.get('dialogue', item.get('text', '')))
        return dialogues
    
    # Plain text format
    # First try splitting by double newlines (paragraph-style)
    paragraphs = content.strip().split('\n\n')
    if len(paragraphs) > 1:
        return [p.strip() for p in paragraphs if p.strip()]
    
    # Otherwise, treat each line as a dialogue
    lines = content.strip().split('\n')
    return [line.strip() for line in lines if line.strip()]


def save_summaries(summaries: List[str], output_path: str, format: str = 'txt'):
    """Save summaries to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
    elif format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for summary in summaries:
                f.write(json.dumps({'summary': summary}, ensure_ascii=False) + '\n')
    else:  # txt
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(summaries))


def run_interactive_mode(summarizer: DialogueSummarizer, logger: logging.Logger):
    """Run interactive summarization mode."""
    print("\n" + "="*60)
    print("Interactive Dialogue Summarization")
    print("="*60)
    print("\nEnter a dialogue (multi-line input supported).")
    print("Press Enter twice to submit, or type 'quit' to exit.\n")
    
    while True:
        print("-" * 40)
        print("Enter dialogue:")
        
        lines = []
        empty_line_count = 0
        
        while True:
            try:
                line = input()
            except EOFError:
                break
            
            if line.lower() == 'quit':
                print("\nGoodbye!")
                return
            
            if line == '':
                empty_line_count += 1
                if empty_line_count >= 2:
                    break
                lines.append('')
            else:
                empty_line_count = 0
                lines.append(line)
        
        dialogue = '\n'.join(lines).strip()
        
        if not dialogue:
            print("No dialogue entered. Please try again.")
            continue
        
        print("\nGenerating summary...")
        start_time = time.time()
        
        try:
            summary = summarizer.summarize(dialogue)
            elapsed = time.time() - start_time
            
            print(f"\n{'='*40}")
            print("SUMMARY:")
            print(f"{'='*40}")
            print(summary)
            print(f"\n[Generated in {elapsed:.2f}s]")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate summaries for dialogues using a trained model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dialogue from command line
  python scripts/inference.py --checkpoint ./checkpoints/best \\
      --dialogue "Alice: Hi!\\nBob: Hello!"
  
  # Batch processing from file
  python scripts/inference.py --checkpoint ./checkpoints/best \\
      --input_file dialogues.txt --output_file summaries.txt
  
  # Interactive mode
  python scripts/inference.py --checkpoint ./checkpoints/best --interactive
  
  # Use HuggingFace model directly (no fine-tuning)
  python scripts/inference.py --model google/pegasus-cnn_dailymail \\
      --dialogue "..."
        """
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--checkpoint',
        type=str,
        help='Path to trained model checkpoint directory'
    )
    model_group.add_argument(
        '--model',
        type=str,
        default='google/pegasus-cnn_dailymail',
        help='HuggingFace model name (used if no checkpoint provided)'
    )
    model_group.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    # Input arguments
    input_group = parser.add_argument_group('Input')
    input_group.add_argument(
        '--dialogue',
        type=str,
        help='Single dialogue to summarize (use \\n for newlines)'
    )
    input_group.add_argument(
        '--input_file',
        type=str,
        help='Path to file containing dialogues'
    )
    input_group.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output_file',
        type=str,
        help='Path to save generated summaries'
    )
    output_group.add_argument(
        '--output_format',
        type=str,
        choices=['txt', 'json', 'jsonl'],
        default='txt',
        help='Output file format (default: txt)'
    )
    
    # Generation arguments
    gen_group = parser.add_argument_group('Generation')
    gen_group.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Maximum summary length in tokens (default: 128)'
    )
    gen_group.add_argument(
        '--min_length',
        type=int,
        default=10,
        help='Minimum summary length in tokens (default: 10)'
    )
    gen_group.add_argument(
        '--num_beams',
        type=int,
        default=4,
        help='Number of beams for beam search (default: 4)'
    )
    gen_group.add_argument(
        '--length_penalty',
        type=float,
        default=2.0,
        help='Length penalty for beam search (default: 2.0)'
    )
    gen_group.add_argument(
        '--no_repeat_ngram_size',
        type=int,
        default=3,
        help='N-gram size for no repeat constraint (default: 3)'
    )
    gen_group.add_argument(
        '--do_sample',
        action='store_true',
        help='Use sampling instead of beam search'
    )
    gen_group.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (default: 1.0)'
    )
    gen_group.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k sampling parameter (default: 50)'
    )
    gen_group.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help='Top-p (nucleus) sampling parameter (default: 0.95)'
    )
    
    # Other arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for batch processing (default: 8)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Validate arguments
    if not any([args.dialogue, args.input_file, args.interactive]):
        parser.error("Must specify --dialogue, --input_file, or --interactive")
    
    # Load config if provided
    config = load_config(args.config)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Build generation config
    generation_config = {
        'max_length': args.max_length,
        'min_length': args.min_length,
        'num_beams': args.num_beams,
        'length_penalty': args.length_penalty,
        'no_repeat_ngram_size': args.no_repeat_ngram_size,
        'do_sample': args.do_sample,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'early_stopping': True,
    }
    
    # Initialize summarizer
    logger.info("Loading model...")
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        
        summarizer = DialogueSummarizer.from_pretrained(
            str(checkpoint_path),
            device=device,
            generation_config=generation_config
        )
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        summarizer = DialogueSummarizer(
            model_name=args.model,
            device=device,
            generation_config=generation_config
        )
        logger.info(f"Loaded base model: {args.model}")
    
    # Run appropriate mode
    if args.interactive:
        run_interactive_mode(summarizer, logger)
    
    elif args.dialogue:
        # Single dialogue mode
        dialogue = args.dialogue.replace('\\n', '\n')
        
        logger.info("Generating summary...")
        start_time = time.time()
        summary = summarizer.summarize(dialogue)
        elapsed = time.time() - start_time
        
        print("\n" + "="*50)
        print("DIALOGUE:")
        print("="*50)
        print(dialogue)
        print("\n" + "="*50)
        print("SUMMARY:")
        print("="*50)
        print(summary)
        print(f"\n[Generated in {elapsed:.2f}s]")
        
        if args.output_file:
            save_summaries([summary], args.output_file, args.output_format)
            logger.info(f"Summary saved to: {args.output_file}")
    
    elif args.input_file:
        # Batch processing mode
        logger.info(f"Loading dialogues from: {args.input_file}")
        dialogues = load_dialogues_from_file(args.input_file)
        logger.info(f"Loaded {len(dialogues)} dialogues")
        
        logger.info("Generating summaries...")
        start_time = time.time()
        
        summaries = summarizer.summarize_batch(
            dialogues,
            batch_size=args.batch_size,
            show_progress=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Generated {len(summaries)} summaries in {elapsed:.2f}s")
        logger.info(f"Average time per dialogue: {elapsed/len(dialogues):.2f}s")
        
        # Save or print results
        if args.output_file:
            save_summaries(summaries, args.output_file, args.output_format)
            logger.info(f"Summaries saved to: {args.output_file}")
        else:
            # Print first few summaries as sample
            print("\n" + "="*50)
            print("SAMPLE SUMMARIES (first 5):")
            print("="*50)
            for i, (dialogue, summary) in enumerate(zip(dialogues[:5], summaries[:5])):
                print(f"\n--- Example {i+1} ---")
                print(f"Dialogue: {dialogue[:200]}..." if len(dialogue) > 200 else f"Dialogue: {dialogue}")
                print(f"Summary: {summary}")


if __name__ == '__main__':
    main()
