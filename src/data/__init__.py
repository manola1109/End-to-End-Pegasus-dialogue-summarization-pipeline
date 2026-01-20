"""
Data Module
===========

Handles dataset loading, preprocessing, and data utilities for the
dialogue summarization pipeline.
"""

from src.data.dataset import SAMSumDataModule, load_samsum_dataset
from src.data.preprocessing import DialoguePreprocessor, clean_dialogue

__all__ = [
    "SAMSumDataModule",
    "load_samsum_dataset",
    "DialoguePreprocessor",
    "clean_dialogue",
]
