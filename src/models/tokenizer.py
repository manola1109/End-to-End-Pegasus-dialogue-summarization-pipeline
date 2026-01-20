"""
Tokenizer Module
================

Provides tokenization utilities and wrappers for text encoding/decoding
in the dialogue summarization pipeline.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    PegasusTokenizer,
    PegasusTokenizerFast,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


def load_tokenizer(
    model_name: str = "google/pegasus-cnn_dailymail",
    use_fast: bool = True,
    cache_dir: Optional[str] = None,
    **kwargs
) -> PreTrainedTokenizerBase:
    """
    Load tokenizer from Hugging Face Hub.
    
    Args:
        model_name: Model name or path
        use_fast: Whether to use fast tokenizer
        cache_dir: Directory for caching
        **kwargs: Additional tokenizer arguments
        
    Returns:
        Loaded tokenizer
        
    Example:
        >>> tokenizer = load_tokenizer()
        >>> tokens = tokenizer("Hello, world!")
    """
    logger.info(f"Loading tokenizer: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            cache_dir=cache_dir,
            **kwargs
        )
        logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


class TokenizerWrapper:
    """
    Wrapper class providing additional tokenization utilities.
    
    Provides convenient methods for encoding dialogues, decoding summaries,
    and handling batch processing.
    
    Args:
        tokenizer: Base tokenizer to wrap
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
        
    Example:
        >>> tokenizer = load_tokenizer()
        >>> wrapper = TokenizerWrapper(tokenizer)
        >>> encoded = wrapper.encode_dialogue("Alice: Hi! Bob: Hello!")
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_input_length: int = 1024,
        max_target_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def encode_dialogue(
        self,
        dialogue: str,
        return_tensors: Optional[str] = "pt",
        padding: Union[bool, str] = True,
        truncation: bool = True,
    ) -> Dict[str, Any]:
        """
        Encode a dialogue for model input.
        
        Args:
            dialogue: Dialogue text to encode
            return_tensors: Return type ("pt", "tf", "np", None)
            padding: Padding strategy
            truncation: Whether to truncate
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoded = self.tokenizer(
            dialogue,
            max_length=self.max_input_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded
    
    def encode_summary(
        self,
        summary: str,
        return_tensors: Optional[str] = "pt",
        padding: Union[bool, str] = True,
        truncation: bool = True,
    ) -> Dict[str, Any]:
        """
        Encode a summary for training labels.
        
        Args:
            summary: Summary text to encode
            return_tensors: Return type
            padding: Padding strategy
            truncation: Whether to truncate
            
        Returns:
            Dictionary with input_ids
        """
        encoded = self.tokenizer(
            text_target=summary,
            max_length=self.max_target_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded
    
    def encode_batch(
        self,
        dialogues: List[str],
        summaries: Optional[List[str]] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of dialogues and optionally summaries.
        
        Args:
            dialogues: List of dialogue texts
            summaries: Optional list of summary texts
            return_tensors: Return type
            
        Returns:
            Dictionary with encoded inputs and optionally labels
        """
        # Encode dialogues
        encoded = self.tokenizer(
            dialogues,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_tensors=return_tensors,
        )
        
        # Encode summaries if provided
        if summaries is not None:
            labels = self.tokenizer(
                text_target=summaries,
                max_length=self.max_target_length,
                padding=True,
                truncation=True,
                return_tensors=return_tensors,
            )
            
            # Replace padding token id with -100 for loss computation
            label_ids = labels["input_ids"]
            label_ids[label_ids == self.tokenizer.pad_token_id] = -100
            encoded["labels"] = label_ids
        
        return encoded
    
    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
    
    def batch_decode(
        self,
        token_ids: Union[torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token IDs to texts.
        
        Args:
            token_ids: Batch of token IDs [batch_size, seq_len]
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            List of decoded text strings
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
    
    def get_token_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get tokenization statistics for a text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with token statistics
        """
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)
        
        return {
            "num_tokens": len(tokens),
            "num_token_ids": len(token_ids),
            "tokens": tokens[:20],  # First 20 tokens for preview
            "exceeds_max_input": len(token_ids) > self.max_input_length,
            "truncation_needed": len(token_ids) > self.max_input_length,
        }
    
    def analyze_dialogue(self, dialogue: str) -> Dict[str, Any]:
        """
        Analyze a dialogue's tokenization characteristics.
        
        Args:
            dialogue: Dialogue text
            
        Returns:
            Dictionary with analysis results
        """
        stats = self.get_token_statistics(dialogue)
        
        # Count speaker turns
        lines = dialogue.strip().split("\n")
        num_turns = len([l for l in lines if ":" in l])
        
        return {
            **stats,
            "num_turns": num_turns,
            "num_characters": len(dialogue),
            "tokens_per_turn": stats["num_tokens"] / max(num_turns, 1),
            "within_limit": not stats["exceeds_max_input"],
        }
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.tokenizer.eos_token_id
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size
    
    def save_pretrained(self, save_path: str) -> None:
        """Save tokenizer to disk."""
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Tokenizer saved to {save_path}")
