
"""
SAMSum Dataset Module
=====================

Handles loading and processing of the SAMSum dialogue summarization dataset.
Provides a DataModule class for easy integration with training pipelines.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.data.preprocessing import DialoguePreprocessor

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset processing."""
    
    dataset_name: str ="knkarthick/samsum"
    dialogue_column: str = "dialogue"
    summary_column: str = "summary"
    max_input_length: int = 1024
    max_target_length: int = 128
    padding: str = "max_length"
    truncation: bool = True
    cache_dir: Optional[str] = None
    preprocessing_num_workers: int = 4
    remove_columns: bool = True
    
    # Preprocessing options
    clean_dialogues: bool = True
    lowercase: bool = False


def load_samsum_dataset(
    cache_dir: Optional[str] = None,
    split: Optional[str] = None
) -> Union[DatasetDict, Dataset]:
    """
    Load the SAMSum dataset from Hugging Face Hub.
    
    Args:
        cache_dir: Directory to cache the dataset
        split: Specific split to load ('train', 'validation', 'test')
               If None, returns all splits as DatasetDict
    
    Returns:
        Dataset or DatasetDict containing the SAMSum data
    
    Example:
        >>> dataset = load_samsum_dataset()
        >>> print(dataset)
        DatasetDict({
            train: Dataset({features: ['id', 'dialogue', 'summary'], num_rows: 14732})
            validation: Dataset({features: ['id', 'dialogue', 'summary'], num_rows: 818})
            test: Dataset({features: ['id', 'dialogue', 'summary'], num_rows: 819})
        })
    """
    logger.info(f"Loading SAMSum dataset (split: {split or 'all'})")
    
    try:
        dataset = load_dataset(
            "knkarthick/samsum",
            cache_dir=cache_dir,
            split=split
        )
        logger.info(f"Successfully loaded SAMSum dataset")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load SAMSum dataset: {e}")
        raise


class SAMSumDataModule:
    """
    Data module for SAMSum dataset processing and loading.
    
    Handles tokenization, preprocessing, and DataLoader creation
    for training, validation, and test splits.
    
    Args:
        tokenizer: Hugging Face tokenizer for text encoding
        config: DataConfig object with processing parameters
        
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
        >>> data_module = SAMSumDataModule(tokenizer)
        >>> data_module.setup()
        >>> train_loader = data_module.train_dataloader(batch_size=4)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[DataConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or DataConfig()
        self.preprocessor = DialoguePreprocessor()
        
        # Dataset splits
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        
        # Processed datasets (tokenized)
        self._train_processed: Optional[Dataset] = None
        self._val_processed: Optional[Dataset] = None
        self._test_processed: Optional[Dataset] = None
        
        self._is_setup = False
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load and preprocess the dataset.
        
        Args:
            stage: Optional stage identifier ('fit', 'validate', 'test', 'predict')
        """
        if self._is_setup:
            logger.info("Data module already set up, skipping...")
            return
        
        logger.info("Setting up SAMSum data module...")
        
        # Load raw dataset
        dataset = load_samsum_dataset(cache_dir=self.config.cache_dir)
        
        self._train_dataset = dataset["train"]
        self._val_dataset = dataset["validation"]
        self._test_dataset = dataset["test"]
        
        # Log dataset statistics
        logger.info(f"Train samples: {len(self._train_dataset)}")
        logger.info(f"Validation samples: {len(self._val_dataset)}")
        logger.info(f"Test samples: {len(self._test_dataset)}")
        
        # Preprocess and tokenize
        if stage in (None, "fit"):
            self._train_processed = self._preprocess_dataset(
                self._train_dataset, "train"
            )
            self._val_processed = self._preprocess_dataset(
                self._val_dataset, "validation"
            )
        
        if stage in (None, "test", "predict"):
            self._test_processed = self._preprocess_dataset(
                self._test_dataset, "test"
            )
        
        self._is_setup = True
        logger.info("Data module setup complete")
    
    def _preprocess_dataset(
        self,
        dataset: Dataset,
        split_name: str
    ) -> Dataset:
        """
        Apply preprocessing and tokenization to a dataset split.
        
        Args:
            dataset: Raw dataset to process
            split_name: Name of the split (for logging)
            
        Returns:
            Processed and tokenized dataset
        """
        logger.info(f"Preprocessing {split_name} split...")
        
        # Apply text cleaning if enabled
        if self.config.clean_dialogues:
            dataset = dataset.map(
                self._clean_example,
                num_proc=self.config.preprocessing_num_workers,
                desc=f"Cleaning {split_name}"
            )
        
        # Tokenize
        processed = dataset.map(
            self._tokenize_example,
            batched=True,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=dataset.column_names if self.config.remove_columns else None,
            desc=f"Tokenizing {split_name}"
        )
        
        # Set format for PyTorch
        processed.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        return processed
    
    def _clean_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a single example's dialogue and summary."""
        example[self.config.dialogue_column] = self.preprocessor.clean(
            example[self.config.dialogue_column]
        )
        return example
    
    def _tokenize_example(
        self,
        examples: Dict[str, List[Any]]
    ) -> Dict[str, List[Any]]:
        """
        Tokenize a batch of examples.
        
        Args:
            examples: Batch of examples with dialogues and summaries
            
        Returns:
            Tokenized batch with input_ids, attention_mask, and labels
        """
        dialogues = examples[self.config.dialogue_column]
        summaries = examples[self.config.summary_column]
        
        # Tokenize inputs (dialogues)
        model_inputs = self.tokenizer(
            dialogues,
            max_length=self.config.max_input_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors=None  # Return lists for dataset mapping
        )
        
        # Tokenize targets (summaries) with text_target for proper handling
        labels = self.tokenizer(
            text_target=summaries,
            max_length=self.config.max_target_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors=None
        )
        
        # Replace padding token id with -100 for loss computation
        labels_ids = labels["input_ids"]
        labels_ids = [
            [(token if token != self.tokenizer.pad_token_id else -100) for token in label]
            for label in labels_ids
        ]
        
        model_inputs["labels"] = labels_ids
        
        return model_inputs
    
    def train_dataloader(
        self,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> DataLoader:
        """Create training DataLoader."""
        if self._train_processed is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        
        return DataLoader(
            self._train_processed,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> DataLoader:
        """Create validation DataLoader."""
        if self._val_processed is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        
        return DataLoader(
            self._val_processed,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> DataLoader:
        """Create test DataLoader."""
        if self._test_processed is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        
        return DataLoader(
            self._test_processed,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.
        
        Handles dynamic batching and padding if needed.
        """
        import torch
        
        return {
            key: torch.stack([example[key] for example in batch])
            for key in batch[0].keys()
        }
    
    @property
    def train_dataset(self) -> Optional[Dataset]:
        """Access raw training dataset."""
        return self._train_dataset
    
    @property
    def val_dataset(self) -> Optional[Dataset]:
        """Access raw validation dataset."""
        return self._val_dataset
    
    @property
    def test_dataset(self) -> Optional[Dataset]:
        """Access raw test dataset."""
        return self._test_dataset
    
    def get_sample(self, split: str = "train", idx: int = 0) -> Dict[str, Any]:
        """
        Get a sample from the specified split.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            idx: Sample index
            
        Returns:
            Dictionary with dialogue and summary
        """
        dataset_map = {
            "train": self._train_dataset,
            "validation": self._val_dataset,
            "test": self._test_dataset
        }
        
        dataset = dataset_map.get(split)
        if dataset is None:
            raise ValueError(f"Invalid split: {split}")
        
        return dataset[idx]
    
    def print_sample(self, split: str = "train", idx: int = 0) -> None:
        """Print a formatted sample from the dataset."""
        sample = self.get_sample(split, idx)
        
        print("=" * 60)
        print(f"Sample from {split} split (index: {idx})")
        print("=" * 60)
        print("\n📝 DIALOGUE:")
        print("-" * 40)
        print(sample[self.config.dialogue_column])
        print("\n📋 SUMMARY:")
        print("-" * 40)
        print(sample[self.config.summary_column])
        print("=" * 60)
