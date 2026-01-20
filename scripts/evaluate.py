#!/usr/bin/env python3
"""
Evaluation Script for Dialogue Summarization
=============================================

Evaluates a trained model on the SAMSum test set and generates detailed reports.

Usage:
    python scripts/evaluate.py --checkpoint ./checkpoints/best_model
    python scripts/evaluate.py --checkpoint ./checkpoints/best_model --detailed
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import SAMSumDataModule, DataConfig
from src.models.pegasus import PegasusForDialogueSummarization, ModelConfig
from src.models.tokenizer import load_tokenizer
from src.evaluation.metrics import (
    RougeEvaluator,
    compute_all_metrics,
    format_metrics_report,
    LengthStatistics,
)
from src.inference.summarizer import DialogueSummarizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate dialogue summarization model"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for generation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed evaluation report"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    
    return parser.parse_args()


def evaluate_model(
    model: PegasusForDialogueSummarization,
    tokenizer,
    dataloader,
    device: torch.device,
    num_beams: int = 4,
    max_length: int = 128,
) -> Dict[str, Any]:
    """
    Run evaluation on a dataloader.
    
    Returns dictionary with predictions, references, and metrics.
    """
    model.eval()
    
    all_predictions = []
    all_references = []
    all_sources = []
    
    logger.info("Generating predictions...")
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Generate predictions
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num_beams=num_beams,
                max_length=max_length,
            )
        
        # Decode predictions
        predictions = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        
        # Decode references (replace -100 with pad_token_id)
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        references = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        
        # Decode sources
        sources = tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        
        all_predictions.extend(predictions)
        all_references.extend(references)
        all_sources.extend(sources)
    
    return {
        "predictions": all_predictions,
        "references": all_references,
        "sources": all_sources,
    }


def generate_detailed_report(
    predictions: List[str],
    references: List[str],
    sources: List[str],
    metrics: Dict[str, Any]
) -> str:
    """Generate a detailed evaluation report."""
    lines = []
    
    lines.append("=" * 70)
    lines.append("DETAILED EVALUATION REPORT")
    lines.append("=" * 70)
    
    # Summary statistics
    lines.append("\n📊 SUMMARY STATISTICS")
    lines.append("-" * 50)
    lines.append(f"Total samples evaluated: {len(predictions)}")
    
    # ROUGE scores
    lines.append("\n📈 ROUGE SCORES")
    lines.append("-" * 50)
    rouge = metrics.get("rouge", {})
    for metric, value in rouge.items():
        lines.append(f"  {metric.upper():12}: {value:.4f}")
    
    # Length statistics
    lines.append("\n📏 LENGTH STATISTICS")
    lines.append("-" * 50)
    length = metrics.get("length", {})
    
    if "prediction_length" in length:
        pred_len = length["prediction_length"]
        lines.append(f"  Prediction length (words):")
        lines.append(f"    Mean: {pred_len['mean']:.1f} | Std: {pred_len['std']:.1f}")
        lines.append(f"    Min: {pred_len['min']} | Max: {pred_len['max']}")
    
    if "reference_length" in length:
        ref_len = length["reference_length"]
        lines.append(f"  Reference length (words):")
        lines.append(f"    Mean: {ref_len['mean']:.1f} | Std: {ref_len['std']:.1f}")
        lines.append(f"    Min: {ref_len['min']} | Max: {ref_len['max']}")
    
    if "compression_ratio" in length:
        lines.append(f"  Compression ratio: {length['compression_ratio']['mean']:.2%}")
    
    # Sample predictions
    lines.append("\n📝 SAMPLE PREDICTIONS")
    lines.append("-" * 50)
    
    num_samples = min(5, len(predictions))
    for i in range(num_samples):
        lines.append(f"\n--- Sample {i+1} ---")
        lines.append(f"Source: {sources[i][:200]}...")
        lines.append(f"Reference: {references[i]}")
        lines.append(f"Prediction: {predictions[i]}")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info(f"Loading model from {args.checkpoint}")
    tokenizer = load_tokenizer(args.checkpoint)
    
    model_config = ModelConfig(
        max_length=args.max_length,
        num_beams=args.num_beams,
    )
    model = PegasusForDialogueSummarization(model_config)
    model.load_pretrained(checkpoint_path=args.checkpoint)
    model.to(device)
    
    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")
    
    # Load dataset
    logger.info("Loading dataset...")
    data_config = DataConfig(
        max_input_length=1024,
        max_target_length=args.max_length,
    )
    data_module = SAMSumDataModule(tokenizer, config=data_config)
    data_module.setup(stage="test")
    
    # Get dataloader
    if args.split == "test":
        dataloader = data_module.test_dataloader(batch_size=args.batch_size)
    else:
        dataloader = data_module.val_dataloader(batch_size=args.batch_size)
    
    # Limit samples if specified
    if args.num_samples:
        from torch.utils.data import Subset, DataLoader
        
        if args.split == "test":
            dataset = data_module._test_processed
        else:
            dataset = data_module._val_processed
        
        subset = Subset(dataset, range(min(args.num_samples, len(dataset))))
        dataloader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_module._collate_fn
        )
    
    logger.info(f"Evaluating on {len(dataloader.dataset)} samples")
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        device=device,
        num_beams=args.num_beams,
        max_length=args.max_length,
    )
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(
        predictions=results["predictions"],
        references=results["references"],
        sources=results["sources"],
    )
    
    # Print results
    print("\n" + format_metrics_report(metrics))
    
    # Save results
    results_file = os.path.join(args.output_dir, f"metrics_{args.split}.json")
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    logger.info(f"Metrics saved to {results_file}")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_file = os.path.join(
            args.output_dir, f"predictions_{args.split}.json"
        )
        predictions_data = [
            {
                "source": src,
                "reference": ref,
                "prediction": pred,
            }
            for src, ref, pred in zip(
                results["sources"],
                results["references"],
                results["predictions"]
            )
        ]
        with open(predictions_file, "w") as f:
            json.dump(predictions_data, f, indent=2)
        logger.info(f"Predictions saved to {predictions_file}")
    
    # Generate detailed report if requested
    if args.detailed:
        report = generate_detailed_report(
            predictions=results["predictions"],
            references=results["references"],
            sources=results["sources"],
            metrics=metrics,
        )
        
        report_file = os.path.join(args.output_dir, f"report_{args.split}.txt")
        with open(report_file, "w") as f:
            f.write(report)
        
        print(report)
        logger.info(f"Detailed report saved to {report_file}")
    
    logger.info("Evaluation complete!")
    
    return metrics


if __name__ == "__main__":
    main()
