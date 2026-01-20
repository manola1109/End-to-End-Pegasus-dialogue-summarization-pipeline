"""
Custom Trainer Module
=====================

Provides a custom trainer class and training utilities for
fine-tuning the dialogue summarization model.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.models.pegasus import PegasusForDialogueSummarization
from src.evaluation.metrics import RougeEvaluator
from src.training.callbacks import (
    Callback,
    CallbackHandler,
    EarlyStoppingCallback,
    CheckpointCallback,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Output paths
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    
    # Training hyperparameters
    epochs: int = 5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate scheduler
    scheduler_type: str = "linear"  # linear, cosine
    warmup_steps: int = 500
    warmup_ratio: float = 0.1
    
    # Optimizer settings
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # Mixed precision
    fp16: bool = True
    
    # Checkpointing
    save_strategy: str = "epoch"  # epoch, steps
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Evaluation
    evaluation_strategy: str = "epoch"  # epoch, steps
    eval_steps: int = 500
    
    # Logging
    logging_steps: int = 100
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Reproducibility
    seed: int = 42
    
    # Generation settings for evaluation
    num_beams: int = 4
    max_length: int = 128


class TrainingState:
    """Tracks training state and metrics."""
    
    def __init__(self):
        self.global_step: int = 0
        self.epoch: int = 0
        self.best_metric: float = float("-inf")
        self.best_model_checkpoint: Optional[str] = None
        
        self.train_loss: float = 0.0
        self.eval_loss: float = 0.0
        self.learning_rate: float = 0.0
        
        self.train_history: List[Dict[str, float]] = []
        self.eval_history: List[Dict[str, float]] = []
        
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "best_model_checkpoint": self.best_model_checkpoint,
            "train_history": self.train_history,
            "eval_history": self.eval_history,
        }
    
    def save(self, path: str) -> None:
        """Save state to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class DialogueSummarizationTrainer:
    """
    Custom trainer for dialogue summarization.
    
    Provides full training pipeline with support for:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpoint management
    - Evaluation during training
    
    Args:
        model: The model to train
        tokenizer: Tokenizer for decoding predictions
        train_dataloader: Training data loader
        eval_dataloader: Validation data loader
        config: Training configuration
        callbacks: List of training callbacks
        
    Example:
        >>> trainer = DialogueSummarizationTrainer(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     train_dataloader=train_loader,
        ...     eval_dataloader=val_loader,
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: PegasusForDialogueSummarization,
        tokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        callbacks: Optional[List[Callback]] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()
        self.compute_metrics = compute_metrics
        
        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        # Initialize state
        self.state = TrainingState()
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.fp16 else None
        
        # Setup callbacks
        self.callback_handler = CallbackHandler(callbacks or [])
        self._setup_default_callbacks()
        
        # Setup evaluator
        self.evaluator = RougeEvaluator()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logging_dir, exist_ok=True)
        
        logger.info(f"Trainer initialized. Device: {self.device}")
        logger.info(f"Training config: {self.config}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        num_training_steps = (
            len(self.train_dataloader) 
            * self.config.epochs 
            // self.config.gradient_accumulation_steps
        )
        
        # Use warmup ratio if warmup_steps not specified
        warmup_steps = self.config.warmup_steps
        if warmup_steps == 0:
            warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        logger.info(f"Created scheduler with {warmup_steps} warmup steps")
        logger.info(f"Total training steps: {num_training_steps}")
        
        return scheduler
    
    def _setup_default_callbacks(self) -> None:
        """Setup default training callbacks."""
        # Add early stopping if enabled
        if self.config.early_stopping_patience > 0:
            early_stopping = EarlyStoppingCallback(
                patience=self.config.early_stopping_patience,
                threshold=self.config.early_stopping_threshold,
            )
            self.callback_handler.add_callback(early_stopping)
        
        # Add checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_dir=self.config.output_dir,
            save_total_limit=self.config.save_total_limit,
        )
        self.callback_handler.add_callback(checkpoint_callback)
    
    def train(self) -> TrainingState:
        """
        Run the full training loop.
        
        Returns:
            TrainingState with training history and metrics
        """
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)
        
        self.state.start_time = datetime.now()
        self.model.train()
        
        # Notify callbacks
        self.callback_handler.on_train_begin(self)
        
        try:
            for epoch in range(self.config.epochs):
                self.state.epoch = epoch
                logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
                logger.info("-" * 40)
                
                # Training epoch
                train_metrics = self._train_epoch()
                self.state.train_history.append(train_metrics)
                
                # Evaluation
                if self.eval_dataloader is not None:
                    eval_metrics = self.evaluate()
                    self.state.eval_history.append(eval_metrics)
                    
                    # Check for best model
                    current_metric = eval_metrics.get("rouge1", 0)
                    if current_metric > self.state.best_metric:
                        self.state.best_metric = current_metric
                        self._save_best_model()
                    
                    # Early stopping check
                    self.callback_handler.on_evaluate(self, eval_metrics)
                    if self.callback_handler.should_stop:
                        logger.info("Early stopping triggered")
                        break
                
                # Epoch end callbacks
                self.callback_handler.on_epoch_end(self)
                
                # Save checkpoint
                if self.config.save_strategy == "epoch":
                    self._save_checkpoint(f"checkpoint-epoch-{epoch + 1}")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            self.state.end_time = datetime.now()
            self.callback_handler.on_train_end(self)
            
            # Load best model if available
            if self.config.load_best_model_at_end and self.state.best_model_checkpoint:
                logger.info(f"Loading best model from {self.state.best_model_checkpoint}")
                self.model.load_pretrained(
                    checkpoint_path=self.state.best_model_checkpoint
                )
        
        # Save final state
        self.state.save(os.path.join(self.config.output_dir, "training_state.json"))
        
        return self.state
    
    def _train_epoch(self) -> Dict[str, float]:
        """Run a single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training",
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.config.fp16 and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs["loss"] / self.config.gradient_accumulation_steps
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.fp16 and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if self.state.global_step % self.config.logging_steps == 0:
                self.callback_handler.on_log(self, {
                    "loss": loss.item() * self.config.gradient_accumulation_steps,
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "global_step": self.state.global_step,
                })
        
        avg_loss = total_loss / num_batches
        self.state.train_loss = avg_loss
        
        logger.info(f"Training loss: {avg_loss:.4f}")
        
        return {"train_loss": avg_loss}
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the validation set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        logger.info("Running evaluation...")
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_references = []
        
        progress_bar = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            leave=False
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Compute loss
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_loss += outputs["loss"].item()
                num_batches += 1
                
                # Generate predictions
                generated_ids = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    num_beams=self.config.num_beams,
                    max_length=self.config.max_length,
                )
                
                # Decode predictions and references
                predictions = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                # Decode labels (replace -100 with pad_token_id)
                labels = batch["labels"].clone()
                labels[labels == -100] = self.tokenizer.pad_token_id
                references = self.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                
                all_predictions.extend(predictions)
                all_references.extend(references)
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        rouge_scores = self.evaluator.compute(all_predictions, all_references)
        
        metrics = {
            "eval_loss": avg_loss,
            **rouge_scores
        }
        
        self.state.eval_loss = avg_loss
        
        logger.info(f"Evaluation loss: {avg_loss:.4f}")
        logger.info(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        logger.info(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        logger.info(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        self.model.train()
        
        return metrics
    
    def _save_checkpoint(self, name: str) -> str:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save training state
        self.state.save(os.path.join(checkpoint_path, "training_state.json"))
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        return checkpoint_path
    
    def _save_best_model(self) -> None:
        """Save the best model checkpoint."""
        best_path = os.path.join(self.config.output_dir, "best_model")
        self.model.save_pretrained(best_path)
        self.tokenizer.save_pretrained(best_path)
        self.state.best_model_checkpoint = best_path
        logger.info(f"New best model saved to {best_path}")


def train_model(
    model: PegasusForDialogueSummarization,
    tokenizer,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None,
    callbacks: Optional[List[Callback]] = None,
) -> Tuple[PegasusForDialogueSummarization, TrainingState]:
    """
    Convenience function to train a model.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataloader: Training data
        eval_dataloader: Validation data
        config: Training configuration
        callbacks: Training callbacks
        
    Returns:
        Tuple of (trained model, training state)
        
    Example:
        >>> model, state = train_model(model, tokenizer, train_loader, val_loader)
    """
    trainer = DialogueSummarizationTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        callbacks=callbacks,
    )
    
    state = trainer.train()
    
    return trainer.model, state
