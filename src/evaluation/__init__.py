"""
Evaluation Module
=================

Provides metrics computation and evaluation utilities for
assessing summarization quality.
"""

from src.evaluation.metrics import (
    RougeEvaluator,
    compute_rouge_scores,
    compute_all_metrics,
)

__all__ = [
    "RougeEvaluator",
    "compute_rouge_scores",
    "compute_all_metrics",
]
