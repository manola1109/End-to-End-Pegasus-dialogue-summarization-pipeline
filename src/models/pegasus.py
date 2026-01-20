"""
Pegasus Model Module
====================

Provides wrapper classes and utilities for Google Pegasus model
used in dialogue summarization.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    PegasusForConditionalGeneration,
    PegasusConfig,
    PreTrainedModel,
    GenerationConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the Pegasus model."""
    
    model_name: str = "google/pegasus-cnn_dailymail"
    
    # Generation parameters
    max_length: int = 128
    min_length: int = 10
    num_beams: int = 4
    length_penalty: float = 2.0
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    
    # Model loading options
    load_in_8bit: bool = False
    torch_dtype: Optional[str] = None  # "float16", "bfloat16", "float32"
    device_map: Optional[str] = None  # "auto", "cuda", "cpu"
    
    # Fine-tuning options
    freeze_encoder: bool = False
    freeze_embeddings: bool = False
    gradient_checkpointing: bool = False


class PegasusForDialogueSummarization(nn.Module):
    """
    Pegasus model wrapper optimized for dialogue summarization.
    
    Provides additional functionality for fine-tuning, generation,
    and model management.
    
    Args:
        config: ModelConfig with model parameters
        
    Example:
        >>> model = PegasusForDialogueSummarization()
        >>> model.load_pretrained()
        >>> summary_ids = model.generate(input_ids, attention_mask)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.model: Optional[PegasusForConditionalGeneration] = None
        self._device = None
        
    def load_pretrained(
        self,
        model_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> "PegasusForDialogueSummarization":
        """
        Load pretrained or fine-tuned model.
        
        Args:
            model_name: Hugging Face model name (overrides config)
            checkpoint_path: Path to local checkpoint
            **kwargs: Additional arguments for from_pretrained
            
        Returns:
            Self for method chaining
        """
        model_name = model_name or self.config.model_name
        
        logger.info(f"Loading model: {checkpoint_path or model_name}")
        
        # Prepare loading arguments
        load_kwargs = self._prepare_load_kwargs(**kwargs)
        
        try:
            if checkpoint_path:
                self.model = PegasusForConditionalGeneration.from_pretrained(
                    checkpoint_path,
                    **load_kwargs
                )
            else:
                self.model = PegasusForConditionalGeneration.from_pretrained(
                    model_name,
                    **load_kwargs
                )
            
            # Apply fine-tuning configurations
            self._apply_finetuning_config()
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model parameters: {self.num_parameters():,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        return self
    
    def _prepare_load_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Prepare keyword arguments for model loading."""
        load_kwargs = {}
        
        # Handle dtype
        if self.config.torch_dtype:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            load_kwargs["torch_dtype"] = dtype_map.get(
                self.config.torch_dtype, torch.float32
            )
        
        # Handle device mapping
        if self.config.device_map:
            load_kwargs["device_map"] = self.config.device_map
        
        # Handle 8-bit loading
        if self.config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        
        # Override with any additional kwargs
        load_kwargs.update(kwargs)
        
        return load_kwargs
    
    def _apply_finetuning_config(self) -> None:
        """Apply fine-tuning specific configurations."""
        if self.model is None:
            return
        
        # Freeze encoder if specified
        if self.config.freeze_encoder:
            logger.info("Freezing encoder parameters")
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
        
        # Freeze embeddings if specified
        if self.config.freeze_embeddings:
            logger.info("Freezing embedding parameters")
            for param in self.model.model.shared.parameters():
                param.requires_grad = False
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing")
            self.model.gradient_checkpointing_enable()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target token IDs for training [batch_size, target_len]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with loss and logits
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Generate summaries for input sequences.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            **generation_kwargs: Override generation parameters
            
        Returns:
            Generated token IDs [batch_size, output_len]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")
        
        # Merge default config with provided kwargs
        gen_config = {
            "max_length": self.config.max_length,
            "min_length": self.config.min_length,
            "num_beams": self.config.num_beams,
            "length_penalty": self.config.length_penalty,
            "early_stopping": self.config.early_stopping,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
        }
        gen_config.update(generation_kwargs)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_config
            )
        
        return generated_ids
    
    def save_pretrained(self, save_path: str) -> None:
        """
        Save model to disk.
        
        Args:
            save_path: Directory path to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
    
    def to(self, device: Union[str, torch.device]) -> "PegasusForDialogueSummarization":
        """Move model to specified device."""
        if self.model is not None:
            self.model = self.model.to(device)
            self._device = device
        return self
    
    @property
    def device(self) -> torch.device:
        """Get the current device of the model."""
        if self.model is not None:
            return next(self.model.parameters()).device
        return torch.device("cpu")
    
    def num_parameters(self, trainable_only: bool = False) -> int:
        """
        Count model parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if self.model is None:
            return 0
        
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())
    
    def get_encoder(self) -> nn.Module:
        """Get the encoder module."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        return self.model.model.encoder
    
    def get_decoder(self) -> nn.Module:
        """Get the decoder module."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        return self.model.model.decoder
    
    def train(self, mode: bool = True) -> "PegasusForDialogueSummarization":
        """Set training mode."""
        if self.model is not None:
            self.model.train(mode)
        return self
    
    def eval(self) -> "PegasusForDialogueSummarization":
        """Set evaluation mode."""
        return self.train(False)


def load_pegasus_model(
    model_name: str = "google/pegasus-cnn_dailymail",
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    **kwargs
) -> PegasusForDialogueSummarization:
    """
    Convenience function to load a Pegasus model.
    
    Args:
        model_name: Hugging Face model name
        checkpoint_path: Path to local checkpoint
        device: Device to load model on ("auto", "cuda", "cpu")
        **kwargs: Additional model configuration
        
    Returns:
        Loaded PegasusForDialogueSummarization model
        
    Example:
        >>> model = load_pegasus_model(device="cuda")
        >>> model = load_pegasus_model(checkpoint_path="./checkpoints/best_model")
    """
    config = ModelConfig(
        model_name=model_name,
        device_map=device if device != "auto" else None,
        **kwargs
    )
    
    model = PegasusForDialogueSummarization(config)
    model.load_pretrained(checkpoint_path=checkpoint_path)
    
    # Handle device placement
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if config.device_map is None:
        model.to(device)
    
    return model


def get_model_info(model: PegasusForDialogueSummarization) -> Dict[str, Any]:
    """
    Get detailed information about the model.
    
    Args:
        model: The model to inspect
        
    Returns:
        Dictionary with model information
    """
    if model.model is None:
        return {"status": "not_loaded"}
    
    config = model.model.config
    
    return {
        "model_type": config.model_type,
        "vocab_size": config.vocab_size,
        "encoder_layers": config.encoder_layers,
        "decoder_layers": config.decoder_layers,
        "d_model": config.d_model,
        "encoder_attention_heads": config.encoder_attention_heads,
        "decoder_attention_heads": config.decoder_attention_heads,
        "encoder_ffn_dim": config.encoder_ffn_dim,
        "decoder_ffn_dim": config.decoder_ffn_dim,
        "max_position_embeddings": config.max_position_embeddings,
        "total_parameters": model.num_parameters(),
        "trainable_parameters": model.num_parameters(trainable_only=True),
        "device": str(model.device),
    }
