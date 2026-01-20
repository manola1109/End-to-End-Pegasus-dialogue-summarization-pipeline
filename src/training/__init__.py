"""
Training Module
===============

Provides training pipeline, custom trainer, and callbacks for
fine-tuning the dialogue summarization model.
"""

from src.training.trainer import (
    DialogueSummarizationTrainer,
    TrainingConfig,
    train_model,
)
from src.training.callbacks import (
    EarlyStoppingCallback,
    MetricsLoggingCallback,
    CheckpointCallback,
    ProgressCallback,
)

__all__ = [
    "DialogueSummarizationTrainer",
    "TrainingConfig",
    "train_model",
    "EarlyStoppingCallback",
    "MetricsLoggingCallback",
    "CheckpointCallback",
    "ProgressCallback",
]
