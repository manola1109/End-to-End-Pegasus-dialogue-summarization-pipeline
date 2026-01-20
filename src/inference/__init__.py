"""
Inference Module
================

Provides inference pipeline for dialogue summarization,
including batch processing and API-like interfaces.
"""

from src.inference.summarizer import (
    DialogueSummarizer,
    SummarizationResult,
    batch_summarize,
)

__all__ = [
    "DialogueSummarizer",
    "SummarizationResult",
    "batch_summarize",
]
