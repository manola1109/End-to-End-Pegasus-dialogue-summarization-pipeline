"""
Metrics Module
==============

Provides evaluation metrics for assessing summarization quality,
including ROUGE scores and other text similarity measures.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RougeScores:
    """Container for ROUGE scores."""
    
    rouge1: float
    rouge2: float
    rougeL: float
    rougeLsum: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rougeL": self.rougeL,
            "rougeLsum": self.rougeLsum,
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"ROUGE-1: {self.rouge1:.4f} | "
            f"ROUGE-2: {self.rouge2:.4f} | "
            f"ROUGE-L: {self.rougeL:.4f} | "
            f"ROUGE-Lsum: {self.rougeLsum:.4f}"
        )


class RougeEvaluator:
    """
    ROUGE score evaluator for summarization.
    
    Computes ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores
    for comparing generated summaries against references.
    
    Args:
        use_stemmer: Whether to use stemming for comparison
        
    Example:
        >>> evaluator = RougeEvaluator()
        >>> predictions = ["The meeting was productive."]
        >>> references = ["The meeting went well and was productive."]
        >>> scores = evaluator.compute(predictions, references)
        >>> print(scores)
        {'rouge1': 0.75, 'rouge2': 0.5, 'rougeL': 0.75, 'rougeLsum': 0.75}
    """
    
    def __init__(self, use_stemmer: bool = True):
        self.use_stemmer = use_stemmer
        self._scorer = None
    
    @property
    def scorer(self):
        """Lazy initialization of ROUGE scorer."""
        if self._scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL", "rougeLsum"],
                    use_stemmer=self.use_stemmer
                )
            except ImportError:
                logger.error("rouge_score package not installed")
                raise ImportError(
                    "Please install rouge_score: pip install rouge-score"
                )
        return self._scorer
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        aggregate: bool = True
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Compute ROUGE scores for predictions against references.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            aggregate: Whether to return aggregated scores
            
        Returns:
            Dictionary with ROUGE scores (aggregated) or list of score dicts
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) doesn't match "
                f"number of references ({len(references)})"
            )
        
        all_scores = []
        
        for pred, ref in zip(predictions, references):
            # Handle empty strings
            if not pred.strip():
                pred = " "
            if not ref.strip():
                ref = " "
            
            scores = self.scorer.score(ref, pred)
            
            score_dict = {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
                "rougeLsum": scores["rougeLsum"].fmeasure,
            }
            all_scores.append(score_dict)
        
        if aggregate:
            return self._aggregate_scores(all_scores)
        
        return all_scores
    
    def _aggregate_scores(
        self,
        scores: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate scores by computing mean."""
        if not scores:
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "rougeLsum": 0.0,
            }
        
        aggregated = {}
        for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
            values = [s[key] for s in scores]
            aggregated[key] = np.mean(values)
        
        return aggregated
    
    def compute_with_confidence(
        self,
        predictions: List[str],
        references: List[str],
        n_bootstrap: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE scores with bootstrap confidence intervals.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with mean, std, and confidence intervals
        """
        # Get individual scores
        individual_scores = self.compute(predictions, references, aggregate=False)
        
        # Bootstrap
        n_samples = len(individual_scores)
        bootstrap_scores = {key: [] for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_scores = [individual_scores[i] for i in indices]
            agg = self._aggregate_scores(sample_scores)
            
            for key in bootstrap_scores:
                bootstrap_scores[key].append(agg[key])
        
        # Compute statistics
        results = {}
        for key in bootstrap_scores:
            values = bootstrap_scores[key]
            results[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "ci_lower": np.percentile(values, 2.5),
                "ci_upper": np.percentile(values, 97.5),
            }
        
        return results


def compute_rouge_scores(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = True
) -> Dict[str, float]:
    """
    Convenience function to compute ROUGE scores.
    
    Args:
        predictions: Generated summaries
        references: Reference summaries
        use_stemmer: Whether to use stemming
        
    Returns:
        Dictionary with ROUGE scores
        
    Example:
        >>> scores = compute_rouge_scores(
        ...     ["Short summary."],
        ...     ["Reference summary text."]
        ... )
    """
    evaluator = RougeEvaluator(use_stemmer=use_stemmer)
    return evaluator.compute(predictions, references)


class BLEUEvaluator:
    """
    BLEU score evaluator (optional metric).
    
    BLEU is less common for summarization but can provide
    additional perspective on n-gram overlap.
    """
    
    def __init__(self):
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of NLTK."""
        if not self._initialized:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            self._initialized = True
    
    def compute(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute BLEU scores."""
        self._initialize()
        
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        
        smoother = SmoothingFunction()
        scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = [word_tokenize(ref.lower())]
            
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                smoothing_function=smoother.method1
            )
            scores.append(score)
        
        return {
            "bleu": np.mean(scores),
            "bleu_std": np.std(scores)
        }


class LengthStatistics:
    """
    Compute length-based statistics for summaries.
    
    Useful for analyzing summary compression ratios and lengths.
    """
    
    @staticmethod
    def compute(
        predictions: List[str],
        references: List[str],
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute length statistics.
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            sources: Optional source documents
            
        Returns:
            Dictionary with length statistics
        """
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        stats = {
            "prediction_length": {
                "mean": np.mean(pred_lengths),
                "std": np.std(pred_lengths),
                "min": min(pred_lengths),
                "max": max(pred_lengths),
            },
            "reference_length": {
                "mean": np.mean(ref_lengths),
                "std": np.std(ref_lengths),
                "min": min(ref_lengths),
                "max": max(ref_lengths),
            },
            "length_ratio": {
                "mean": np.mean([p/r if r > 0 else 0 for p, r in zip(pred_lengths, ref_lengths)]),
            }
        }
        
        if sources:
            source_lengths = [len(s.split()) for s in sources]
            stats["source_length"] = {
                "mean": np.mean(source_lengths),
                "std": np.std(source_lengths),
            }
            stats["compression_ratio"] = {
                "mean": np.mean([p/s if s > 0 else 0 for p, s in zip(pred_lengths, source_lengths)]),
            }
        
        return stats


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None,
    include_bleu: bool = False
) -> Dict[str, Any]:
    """
    Compute all available metrics.
    
    Args:
        predictions: Generated summaries
        references: Reference summaries
        sources: Optional source documents
        include_bleu: Whether to include BLEU scores
        
    Returns:
        Comprehensive metrics dictionary
    """
    metrics = {}
    
    # ROUGE scores
    rouge_eval = RougeEvaluator()
    metrics["rouge"] = rouge_eval.compute(predictions, references)
    
    # Length statistics
    metrics["length"] = LengthStatistics.compute(predictions, references, sources)
    
    # BLEU (optional)
    if include_bleu:
        bleu_eval = BLEUEvaluator()
        metrics["bleu"] = bleu_eval.compute(predictions, references)
    
    return metrics


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """
    Format metrics as a readable report.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 50)
    
    # ROUGE scores
    if "rouge" in metrics:
        lines.append("\n📊 ROUGE Scores:")
        lines.append("-" * 30)
        for key, value in metrics["rouge"].items():
            lines.append(f"  {key.upper()}: {value:.4f}")
    
    # Length statistics
    if "length" in metrics:
        lines.append("\n📏 Length Statistics:")
        lines.append("-" * 30)
        length = metrics["length"]
        lines.append(f"  Prediction avg length: {length['prediction_length']['mean']:.1f} words")
        lines.append(f"  Reference avg length: {length['reference_length']['mean']:.1f} words")
        lines.append(f"  Length ratio: {length['length_ratio']['mean']:.2f}")
    
    # BLEU (if available)
    if "bleu" in metrics:
        lines.append("\n📈 BLEU Score:")
        lines.append("-" * 30)
        lines.append(f"  BLEU: {metrics['bleu']['bleu']:.4f}")
    
    lines.append("\n" + "=" * 50)
    
    return "\n".join(lines)
