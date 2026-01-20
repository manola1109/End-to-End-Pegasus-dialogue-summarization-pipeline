#!/usr/bin/env python3
"""
Training Script for Dialogue Summarization
==========================================

Fine-tunes a Pegasus model on the SAMSum dataset for dialogue summarization.

Usage:
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --epochs 5 --batch_size 4 --output_dir ./checkpoints
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import SAMSumDataModule, DataConfig
from src.models.pegasus import PegasusForDialogueSummarization, ModelConfig
from src.models.tokenizer import load_tokenizer
from src.training.trainer import DialogueSummarizationTrainer, TrainingConfig
from src.training.callbacks import (
    MetricsLoggingCallback,
    ProgressCallback,
    GradientMonitorCallback,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train dialogue summarization model"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name or path (overrides config)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="Directory for logs"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (small dataset)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    
    # Set seed
    seed = args.seed or config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    
    # Determine device
    if args.no_cuda or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    logger.info(f"Using device: {device}")
    
    # Model configuration
    model_name = args.model_name or config.get("model", {}).get(
        "name", "google/pegasus-cnn_dailymail"
    )
    
    # Training configuration
    training_config = TrainingConfig(
        output_dir=args.output_dir or config.get("training", {}).get(
            "output_dir", "./checkpoints"
        ),
        logging_dir=args.logging_dir or config.get("training", {}).get(
            "logging_dir", "./logs"
        ),
        epochs=args.epochs or config.get("training", {}).get("epochs", 5),
        batch_size=args.batch_size or config.get("training", {}).get("batch_size", 4),
        learning_rate=args.learning_rate or config.get("training", {}).get(
            "learning_rate", 5e-5
        ),
        gradient_accumulation_steps=args.gradient_accumulation_steps or config.get(
            "training", {}
        ).get("gradient_accumulation_steps", 4),
        fp16=args.fp16 or config.get("training", {}).get("fp16", True),
        warmup_steps=config.get("training", {}).get("scheduler", {}).get(
            "warmup_steps", 500
        ),
        weight_decay=config.get("training", {}).get("weight_decay", 0.01),
        max_grad_norm=config.get("training", {}).get("max_grad_norm", 1.0),
        early_stopping_patience=config.get("training", {}).get(
            "early_stopping", {}
        ).get("patience", 3),
        save_total_limit=config.get("training", {}).get("save_total_limit", 3),
        num_beams=config.get("evaluation", {}).get("num_beams", 4),
        max_length=config.get("model", {}).get("max_target_length", 128),
        seed=seed,
    )
    
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Epochs: {training_config.epochs}")
    logger.info(f"Batch size: {training_config.batch_size}")
    logger.info(f"Learning rate: {training_config.learning_rate}")
    logger.info(f"Output directory: {training_config.output_dir}")
    logger.info("=" * 60)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(model_name)
    
    # Load and prepare dataset
    logger.info("Loading dataset...")
    data_config = DataConfig(
        max_input_length=config.get("model", {}).get("max_input_length", 1024),
        max_target_length=config.get("model", {}).get("max_target_length", 128),
        cache_dir=config.get("data", {}).get("cache_dir", "./cache"),
    )
    
    data_module = SAMSumDataModule(tokenizer, config=data_config)
    data_module.setup()
    
    # Print sample
    logger.info("\nSample from training set:")
    data_module.print_sample("train", 0)
    
    # Create dataloaders
    train_dataloader = data_module.train_dataloader(
        batch_size=training_config.batch_size,
        shuffle=True,
    )
    eval_dataloader = data_module.val_dataloader(
        batch_size=training_config.batch_size * 2,
    )
    
    if args.debug:
        # Use small subset for debugging
        logger.info("Debug mode: using small dataset subset")
        from torch.utils.data import Subset, DataLoader
        
        train_subset = Subset(data_module._train_processed, range(100))
        eval_subset = Subset(data_module._val_processed, range(50))
        
        train_dataloader = DataLoader(
            train_subset,
            batch_size=training_config.batch_size,
            shuffle=True,
            collate_fn=data_module._collate_fn
        )
        eval_dataloader = DataLoader(
            eval_subset,
            batch_size=training_config.batch_size,
            shuffle=False,
            collate_fn=data_module._collate_fn
        )
    
    logger.info(f"Training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Validation samples: {len(eval_dataloader.dataset)}")
    
    # Load model
    logger.info("Loading model...")
    model_config = ModelConfig(
        model_name=model_name,
        max_length=config.get("model", {}).get("max_target_length", 128),
        num_beams=config.get("evaluation", {}).get("num_beams", 4),
        gradient_checkpointing=config.get("training", {}).get(
            "gradient_checkpointing", False
        ),
    )
    
    model = PegasusForDialogueSummarization(model_config)
    
    if args.checkpoint:
        model.load_pretrained(checkpoint_path=args.checkpoint)
    else:
        model.load_pretrained()
    
    logger.info(f"Model parameters: {model.num_parameters():,}")
    logger.info(f"Trainable parameters: {model.num_parameters(trainable_only=True):,}")
    
    # Setup callbacks
    callbacks = [
        ProgressCallback(),
        MetricsLoggingCallback(
            log_dir=training_config.logging_dir,
            use_tensorboard=True,
        ),
    ]
    
    if args.debug:
        callbacks.append(GradientMonitorCallback(log_freq=10))
    
    # Initialize trainer
    trainer = DialogueSummarizationTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=training_config,
        callbacks=callbacks,
    )
    
    # Train
    logger.info("\nStarting training...")
    state = trainer.train()
    
    # Print final results
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best metric (ROUGE-1): {state.best_metric:.4f}")
    logger.info(f"Best model saved at: {state.best_model_checkpoint}")
    logger.info(f"Training history saved at: {training_config.output_dir}/training_state.json")
    
    # Save final model
    final_model_path = os.path.join(training_config.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Final model saved at: {final_model_path}")
    
    return state


if __name__ == "__main__":
    main()
