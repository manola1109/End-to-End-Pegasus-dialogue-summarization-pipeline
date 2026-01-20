"""
Dialogue Summarization System
=============================

A production-ready text summarization system for dialogue summarization
using the SAMSum dataset and Google Pegasus model.

Modules:
    - data: Dataset loading and preprocessing
    - models: Model definitions and utilities
    - training: Training pipeline and callbacks
    - evaluation: Metrics computation
    - inference: Inference pipeline
"""

__version__ = "1.0.0"
__author__ = "Dialogue Summarization Team"

from src.inference.summarizer import DialogueSummarizer
from src.data.dataset import SAMSumDataModule
from src.models.pegasus import PegasusForDialogueSummarization
from src.evaluation.metrics import RougeEvaluator

__all__ = [
    "DialogueSummarizer",
    "SAMSumDataModule",
    "PegasusForDialogueSummarization",
    "RougeEvaluator",
]
