"""
Dialogue Summarizer Module
==========================

Provides the main inference interface for dialogue summarization,
with support for single and batch processing.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from src.models.pegasus import PegasusForDialogueSummarization, ModelConfig
from src.models.tokenizer import load_tokenizer, TokenizerWrapper
from src.data.preprocessing import DialoguePreprocessor

logger = logging.getLogger(__name__)


@dataclass
class SummarizationResult:
    """Container for summarization results."""
    
    dialogue: str
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"Summary: {self.summary}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dialogue": self.dialogue,
            "summary": self.summary,
            "metadata": self.metadata,
        }


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    max_length: int = 128
    min_length: int = 10
    num_beams: int = 4
    length_penalty: float = 2.0
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95


class DialogueSummarizer:
    """
    Main inference class for dialogue summarization.
    
    Provides a clean API for generating summaries from dialogues,
    with support for both single inputs and batch processing.
    
    Args:
        model_path: Path to trained model checkpoint
        model_name: HuggingFace model name (alternative to model_path)
        device: Device to run inference on
        generation_config: Configuration for text generation
        
    Example:
        >>> summarizer = DialogueSummarizer("./checkpoints/best_model")
        >>> summary = summarizer.summarize("Alice: Hi! Bob: Hello!")
        >>> print(summary)
        "Alice and Bob greet each other."
        
        >>> # Batch processing
        >>> summaries = summarizer.summarize_batch(dialogues)
        
        >>> # Using from_pretrained class method
        >>> summarizer = DialogueSummarizer.from_pretrained("./checkpoint")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        device: str = "auto",
        generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.device = self._resolve_device(device)
        
        # Handle generation_config as dict or GenerationConfig
        if isinstance(generation_config, dict):
            self.generation_config = GenerationConfig(**{
                k: v for k, v in generation_config.items()
                if k in GenerationConfig.__dataclass_fields__
            })
        else:
            self.generation_config = generation_config or GenerationConfig()
        
        self.model: Optional[PegasusForDialogueSummarization] = None
        self.tokenizer = None
        self.tokenizer_wrapper: Optional[TokenizerWrapper] = None
        self.preprocessor = DialoguePreprocessor()
        
        self._is_loaded = False
        
        # Auto-load if path or model name provided
        if model_path:
            self.load_model(model_path)
        elif model_name:
            self.load_from_hub(model_name)
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "auto",
        generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
        **kwargs
    ) -> "DialogueSummarizer":
        """
        Load a DialogueSummarizer from a pretrained checkpoint.
        
        This is a convenience class method for loading trained models.
        
        Args:
            model_path: Path to model checkpoint directory
            device: Device to use ('auto', 'cuda', or 'cpu')
            generation_config: Generation configuration
            **kwargs: Additional arguments passed to load_model
            
        Returns:
            Initialized DialogueSummarizer instance
            
        Example:
            >>> summarizer = DialogueSummarizer.from_pretrained(
            ...     "./checkpoints/best_model",
            ...     device="cuda"
            ... )
        """
        return cls(
            model_path=model_path,
            device=device,
            generation_config=generation_config,
        )
    
    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def load_model(
        self,
        model_path: str,
        **kwargs
    ) -> "DialogueSummarizer":
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            **kwargs: Additional arguments for model loading
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(model_path)
        self.tokenizer_wrapper = TokenizerWrapper(
            self.tokenizer,
            max_input_length=1024,
            max_target_length=self.generation_config.max_length,
        )
        
        # Load model
        model_config = ModelConfig(
            max_length=self.generation_config.max_length,
            min_length=self.generation_config.min_length,
            num_beams=self.generation_config.num_beams,
            length_penalty=self.generation_config.length_penalty,
            early_stopping=self.generation_config.early_stopping,
            no_repeat_ngram_size=self.generation_config.no_repeat_ngram_size,
        )
        
        self.model = PegasusForDialogueSummarization(model_config)
        self.model.load_pretrained(checkpoint_path=model_path, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        
        self._is_loaded = True
        logger.info(f"Model loaded successfully on {self.device}")
        
        return self
    
    def load_from_hub(
        self,
        model_name: str = "google/pegasus-cnn_dailymail",
        **kwargs
    ) -> "DialogueSummarizer":
        """
        Load model from Hugging Face Hub.
        
        Args:
            model_name: Model name on Hugging Face Hub
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Loading model from Hugging Face Hub: {model_name}")
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(model_name)
        self.tokenizer_wrapper = TokenizerWrapper(
            self.tokenizer,
            max_input_length=1024,
            max_target_length=self.generation_config.max_length,
        )
        
        # Load model
        model_config = ModelConfig(
            model_name=model_name,
            max_length=self.generation_config.max_length,
            min_length=self.generation_config.min_length,
            num_beams=self.generation_config.num_beams,
            length_penalty=self.generation_config.length_penalty,
            early_stopping=self.generation_config.early_stopping,
            no_repeat_ngram_size=self.generation_config.no_repeat_ngram_size,
        )
        
        self.model = PegasusForDialogueSummarization(model_config)
        self.model.load_pretrained(model_name=model_name, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        
        self._is_loaded = True
        logger.info(f"Model loaded successfully on {self.device}")
        
        return self
    
    def summarize(
        self,
        dialogue: str,
        clean_input: bool = True,
        return_result: bool = False,
        return_metadata: bool = False,
        **generation_kwargs
    ) -> Union[str, SummarizationResult]:
        """
        Generate summary for a single dialogue.
        
        Args:
            dialogue: Input dialogue text
            clean_input: Whether to preprocess the input
            return_result: If True, return SummarizationResult object
            return_metadata: Whether to include metadata (implies return_result=True)
            **generation_kwargs: Override generation parameters
            
        Returns:
            Generated summary string, or SummarizationResult if return_result=True
            
        Example:
            >>> summary = summarizer.summarize(
            ...     "Alice: Let's meet tomorrow\\nBob: Sure, what time?"
            ... )
            >>> print(summary)
            
            >>> # Get full result object
            >>> result = summarizer.summarize(dialogue, return_result=True)
            >>> print(result.summary, result.metadata)
        """
        self._check_loaded()
        
        # Preprocess if requested
        if clean_input:
            dialogue = self.preprocessor.clean(dialogue)
        
        # Encode input
        encoded = self.tokenizer_wrapper.encode_dialogue(
            dialogue,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Merge generation config with kwargs
        gen_kwargs = self._get_generation_kwargs(**generation_kwargs)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
        
        # Decode
        summary = self.tokenizer_wrapper.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        # Return string by default, or full result if requested
        if return_result or return_metadata:
            metadata = {}
            if return_metadata:
                metadata = {
                    "input_length": input_ids.shape[1],
                    "output_length": generated_ids.shape[1],
                    "device": str(self.device),
                }
            
            return SummarizationResult(
                dialogue=dialogue,
                summary=summary,
                metadata=metadata
            )
        
        return summary
    
    def summarize_batch(
        self,
        dialogues: List[str],
        clean_input: bool = True,
        batch_size: int = 8,
        show_progress: bool = True,
        return_results: bool = False,
        **generation_kwargs
    ) -> Union[List[str], List[SummarizationResult]]:
        """
        Generate summaries for multiple dialogues.
        
        Args:
            dialogues: List of input dialogues
            clean_input: Whether to preprocess inputs
            batch_size: Batch size for inference
            show_progress: Whether to show progress bar
            return_results: If True, return SummarizationResult objects
            **generation_kwargs: Override generation parameters
            
        Returns:
            List of summary strings, or List of SummarizationResult if return_results=True
            
        Example:
            >>> dialogues = ["dialogue1...", "dialogue2..."]
            >>> summaries = summarizer.summarize_batch(dialogues)
            >>> # summaries is a list of strings
            
            >>> # Get full result objects
            >>> results = summarizer.summarize_batch(dialogues, return_results=True)
        """
        self._check_loaded()
        
        from tqdm import tqdm
        
        # Preprocess if requested
        original_dialogues = dialogues
        if clean_input:
            dialogues = [self.preprocessor.clean(d) for d in dialogues]
        
        results = []
        summaries = []
        
        # Process in batches
        iterator = range(0, len(dialogues), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Summarizing")
        
        for i in iterator:
            batch_dialogues = dialogues[i:i + batch_size]
            
            # Encode batch
            encoded = self.tokenizer(
                batch_dialogues,
                max_length=1024,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Generate
            gen_kwargs = self._get_generation_kwargs(**generation_kwargs)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
            
            # Decode
            batch_summaries = self.tokenizer_wrapper.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            summaries.extend(batch_summaries)
            
            # Build results if needed
            if return_results:
                for dialogue, summary in zip(batch_dialogues, batch_summaries):
                    results.append(SummarizationResult(
                        dialogue=dialogue,
                        summary=summary
                    ))
        
        if return_results:
            return results
        return summaries
    
    def _get_generation_kwargs(self, **overrides) -> Dict[str, Any]:
        """Get generation kwargs with config defaults."""
        kwargs = {
            "max_length": self.generation_config.max_length,
            "min_length": self.generation_config.min_length,
            "num_beams": self.generation_config.num_beams,
            "length_penalty": self.generation_config.length_penalty,
            "early_stopping": self.generation_config.early_stopping,
            "no_repeat_ngram_size": self.generation_config.no_repeat_ngram_size,
        }
        
        if self.generation_config.do_sample:
            kwargs.update({
                "do_sample": True,
                "temperature": self.generation_config.temperature,
                "top_k": self.generation_config.top_k,
                "top_p": self.generation_config.top_p,
            })
        
        kwargs.update(overrides)
        return kwargs
    
    def _check_loaded(self) -> None:
        """Check if model is loaded."""
        if not self._is_loaded:
            raise RuntimeError(
                "Model not loaded. Call load_model() or load_from_hub() first."
            )
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "device": str(self.device),
            "model_path": self.model_path,
            "vocab_size": self.tokenizer.vocab_size,
            "generation_config": {
                "max_length": self.generation_config.max_length,
                "num_beams": self.generation_config.num_beams,
            }
        }


def batch_summarize(
    dialogues: List[str],
    model_path: str,
    device: str = "auto",
    batch_size: int = 8,
    **kwargs
) -> List[str]:
    """
    Convenience function for batch summarization.
    
    Args:
        dialogues: List of dialogues to summarize
        model_path: Path to model checkpoint
        device: Device to use
        batch_size: Batch size
        **kwargs: Additional arguments
        
    Returns:
        List of generated summaries
        
    Example:
        >>> summaries = batch_summarize(
        ...     dialogues=["..."],
        ...     model_path="./checkpoints/best_model"
        ... )
    """
    summarizer = DialogueSummarizer(model_path=model_path, device=device)
    results = summarizer.summarize_batch(
        dialogues,
        batch_size=batch_size,
        **kwargs
    )
    return [r.summary for r in results]


class SummarizationPipeline:
    """
    High-level pipeline for dialogue summarization.
    
    Provides a simple API similar to Hugging Face pipelines.
    """
    
    def __init__(
        self,
        model: str = "google/pegasus-cnn_dailymail",
        device: str = "auto",
    ):
        self.summarizer = DialogueSummarizer(device=device)
        
        if Path(model).exists():
            self.summarizer.load_model(model)
        else:
            self.summarizer.load_from_hub(model)
    
    def __call__(
        self,
        dialogues: Union[str, List[str]],
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate summaries.
        
        Args:
            dialogues: Single dialogue or list of dialogues
            **kwargs: Generation arguments
            
        Returns:
            Summary or list of summaries
        """
        if isinstance(dialogues, str):
            result = self.summarizer.summarize(dialogues, **kwargs)
            return result.summary
        else:
            results = self.summarizer.summarize_batch(dialogues, **kwargs)
            return [r.summary for r in results]
