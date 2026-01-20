"""
Training Callbacks Module
=========================

Provides callback classes for customizing training behavior,
including early stopping, checkpointing, and logging.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.trainer import DialogueSummarizationTrainer

logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Base class for training callbacks.
    
    Callbacks can be used to customize training behavior at various points
    in the training loop.
    """
    
    def on_train_begin(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_step_begin(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Called at the beginning of each training step."""
        pass
    
    def on_step_end(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Called at the end of each training step."""
        pass
    
    def on_evaluate(
        self,
        trainer: "DialogueSummarizationTrainer",
        metrics: Dict[str, float]
    ) -> None:
        """Called after evaluation."""
        pass
    
    def on_log(
        self,
        trainer: "DialogueSummarizationTrainer",
        logs: Dict[str, Any]
    ) -> None:
        """Called when logging metrics."""
        pass
    
    def should_stop_training(self) -> bool:
        """Return True to stop training early."""
        return False


class CallbackHandler:
    """
    Manages a collection of callbacks.
    
    Dispatches events to all registered callbacks.
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
        self.should_stop = False
    
    def add_callback(self, callback: Callback) -> None:
        """Add a callback to the handler."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callback) -> None:
        """Remove a callback from the handler."""
        self.callbacks.remove(callback)
    
    def on_train_begin(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Dispatch train begin event."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Dispatch train end event."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Dispatch epoch begin event."""
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer)
    
    def on_epoch_end(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Dispatch epoch end event."""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer)
            if callback.should_stop_training():
                self.should_stop = True
    
    def on_step_begin(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Dispatch step begin event."""
        for callback in self.callbacks:
            callback.on_step_begin(trainer)
    
    def on_step_end(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Dispatch step end event."""
        for callback in self.callbacks:
            callback.on_step_end(trainer)
    
    def on_evaluate(
        self,
        trainer: "DialogueSummarizationTrainer",
        metrics: Dict[str, float]
    ) -> None:
        """Dispatch evaluate event."""
        for callback in self.callbacks:
            callback.on_evaluate(trainer, metrics)
            if callback.should_stop_training():
                self.should_stop = True
    
    def on_log(
        self,
        trainer: "DialogueSummarizationTrainer",
        logs: Dict[str, Any]
    ) -> None:
        """Dispatch log event."""
        for callback in self.callbacks:
            callback.on_log(trainer, logs)


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback based on validation metrics.
    
    Stops training if the monitored metric doesn't improve for
    a specified number of evaluations.
    
    Args:
        patience: Number of evaluations with no improvement to wait
        threshold: Minimum change to qualify as an improvement
        metric_name: Name of the metric to monitor
        mode: 'max' for metrics where higher is better, 'min' for lower
        
    Example:
        >>> callback = EarlyStoppingCallback(patience=3, metric_name="rouge1")
        >>> trainer = DialogueSummarizationTrainer(..., callbacks=[callback])
    """
    
    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.001,
        metric_name: str = "rouge1",
        mode: str = "max"
    ):
        self.patience = patience
        self.threshold = threshold
        self.metric_name = metric_name
        self.mode = mode
        
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self._should_stop = False
    
    def on_evaluate(
        self,
        trainer: "DialogueSummarizationTrainer",
        metrics: Dict[str, float]
    ) -> None:
        """Check if training should stop."""
        if self.metric_name not in metrics:
            logger.warning(f"Metric '{self.metric_name}' not found in metrics")
            return
        
        current_metric = metrics[self.metric_name]
        
        # Check for improvement
        if self.mode == "max":
            improved = current_metric > self.best_metric + self.threshold
        else:
            improved = current_metric < self.best_metric - self.threshold
        
        if improved:
            self.best_metric = current_metric
            self.counter = 0
            logger.info(
                f"Early stopping: New best {self.metric_name} = {current_metric:.4f}"
            )
        else:
            self.counter += 1
            logger.info(
                f"Early stopping: No improvement for {self.counter}/{self.patience} evals"
            )
            
            if self.counter >= self.patience:
                logger.info("Early stopping triggered!")
                self._should_stop = True
    
    def should_stop_training(self) -> bool:
        """Return whether training should stop."""
        return self._should_stop


class CheckpointCallback(Callback):
    """
    Checkpoint management callback.
    
    Manages saving and cleanup of model checkpoints during training.
    
    Args:
        save_dir: Directory to save checkpoints
        save_total_limit: Maximum number of checkpoints to keep
        save_on_train_end: Whether to save final checkpoint
    """
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_total_limit: int = 3,
        save_on_train_end: bool = True
    ):
        self.save_dir = Path(save_dir)
        self.save_total_limit = save_total_limit
        self.save_on_train_end = save_on_train_end
        self.checkpoints: List[Path] = []
    
    def on_train_begin(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Create checkpoint directory."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Manage checkpoint cleanup."""
        self._cleanup_checkpoints()
    
    def on_train_end(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Save final checkpoint if configured."""
        if self.save_on_train_end:
            final_path = self.save_dir / "final_model"
            trainer.model.save_pretrained(str(final_path))
            trainer.tokenizer.save_pretrained(str(final_path))
            logger.info(f"Final model saved to {final_path}")
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints if limit exceeded."""
        if len(self.checkpoints) <= self.save_total_limit:
            return
        
        # Sort by modification time
        self.checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        # Remove oldest checkpoints
        while len(self.checkpoints) > self.save_total_limit:
            oldest = self.checkpoints.pop(0)
            if oldest.exists():
                import shutil
                shutil.rmtree(oldest)
                logger.info(f"Removed old checkpoint: {oldest}")


class MetricsLoggingCallback(Callback):
    """
    Callback for logging metrics to various backends.
    
    Supports logging to TensorBoard, Weights & Biases, or custom loggers.
    
    Args:
        log_dir: Directory for log files
        use_tensorboard: Whether to log to TensorBoard
        use_wandb: Whether to log to Weights & Biases
        project_name: Project name for W&B
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        project_name: str = "dialogue-summarization"
    ):
        self.log_dir = Path(log_dir)
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.project_name = project_name
        
        self.writer = None
        self.wandb_run = None
    
    def on_train_begin(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Initialize logging backends."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                logger.info(f"TensorBoard logging enabled at {self.log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available")
        
        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=self.project_name)
                logger.info("Weights & Biases logging enabled")
            except ImportError:
                logger.warning("Weights & Biases not available")
    
    def on_log(
        self,
        trainer: "DialogueSummarizationTrainer",
        logs: Dict[str, Any]
    ) -> None:
        """Log metrics to backends."""
        step = logs.get("global_step", trainer.state.global_step)
        
        if self.writer:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", value, step)
        
        if self.wandb_run:
            import wandb
            wandb.log(logs, step=step)
    
    def on_evaluate(
        self,
        trainer: "DialogueSummarizationTrainer",
        metrics: Dict[str, float]
    ) -> None:
        """Log evaluation metrics."""
        step = trainer.state.global_step
        
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"eval/{key}", value, step)
        
        if self.wandb_run:
            import wandb
            eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            wandb.log(eval_metrics, step=step)
    
    def on_train_end(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Close logging backends."""
        if self.writer:
            self.writer.close()
        
        if self.wandb_run:
            import wandb
            wandb.finish()


class ProgressCallback(Callback):
    """
    Progress reporting callback with rich formatting.
    
    Provides detailed progress information during training.
    """
    
    def __init__(self, print_freq: int = 10):
        self.print_freq = print_freq
        self.epoch_start_time = None
        self.train_start_time = None
    
    def on_train_begin(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Log training start."""
        self.train_start_time = datetime.now()
        
        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info("=" * 60)
        logger.info(f"Start time: {self.train_start_time}")
        logger.info(f"Device: {trainer.device}")
        logger.info(f"Epochs: {trainer.config.epochs}")
        logger.info(f"Batch size: {trainer.config.batch_size}")
        logger.info(f"Learning rate: {trainer.config.learning_rate}")
        logger.info("=" * 60)
    
    def on_epoch_begin(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Log epoch start."""
        self.epoch_start_time = datetime.now()
    
    def on_epoch_end(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Log epoch summary."""
        epoch_duration = datetime.now() - self.epoch_start_time
        
        logger.info(f"\nEpoch {trainer.state.epoch + 1} Summary:")
        logger.info(f"  Duration: {epoch_duration}")
        logger.info(f"  Train loss: {trainer.state.train_loss:.4f}")
        if trainer.state.eval_loss > 0:
            logger.info(f"  Eval loss: {trainer.state.eval_loss:.4f}")
    
    def on_train_end(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Log training summary."""
        total_duration = datetime.now() - self.train_start_time
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total duration: {total_duration}")
        logger.info(f"Best metric: {trainer.state.best_metric:.4f}")
        logger.info(f"Final train loss: {trainer.state.train_loss:.4f}")
        logger.info("=" * 60)


class GradientMonitorCallback(Callback):
    """
    Callback for monitoring gradient statistics.
    
    Useful for debugging training issues like vanishing/exploding gradients.
    """
    
    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq
        self.step = 0
    
    def on_step_end(self, trainer: "DialogueSummarizationTrainer") -> None:
        """Log gradient statistics."""
        self.step += 1
        
        if self.step % self.log_freq != 0:
            return
        
        total_norm = 0.0
        param_count = 0
        
        for param in trainer.model.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
                param_count += 1
        
        total_norm = total_norm ** 0.5
        
        logger.debug(
            f"Step {self.step}: Gradient norm = {total_norm:.4f} "
            f"(from {param_count} params)"
        )
