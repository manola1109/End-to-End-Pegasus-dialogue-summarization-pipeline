"""
Text Preprocessing Module
=========================

Provides utilities for cleaning and preprocessing dialogue text
for the summarization pipeline.
"""

import re
import logging
from typing import List, Optional, Pattern
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    
    # Character handling
    remove_extra_whitespace: bool = True
    normalize_unicode: bool = True
    
    # Dialogue-specific
    preserve_speaker_labels: bool = True
    normalize_speaker_format: bool = True
    
    # Special tokens
    handle_urls: bool = True
    handle_emojis: bool = True
    handle_special_tokens: bool = True  # <file_photo>, <file_gif>, etc.
    
    # Case handling
    lowercase: bool = False


class DialoguePreprocessor:
    """
    Preprocessor for dialogue text.
    
    Handles cleaning and normalization of messenger-style dialogues
    while preserving important structural elements.
    
    Args:
        config: PreprocessingConfig with processing options
        
    Example:
        >>> preprocessor = DialoguePreprocessor()
        >>> cleaned = preprocessor.clean("John:   Hello!   How are you?")
        >>> print(cleaned)
        "John: Hello! How are you?"
    """
    
    # Compiled regex patterns for efficiency
    PATTERNS = {
        "extra_whitespace": re.compile(r"\s+"),
        "speaker_label": re.compile(r"^([A-Za-z][A-Za-z0-9_\s]*)\s*:\s*", re.MULTILINE),
        "url": re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|"
            r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        ),
        "special_token": re.compile(r"<file_[a-z]+>|<[a-z_]+>"),
        "emoji": re.compile(
            r"[\U0001F600-\U0001F64F"  # emoticons
            r"\U0001F300-\U0001F5FF"   # symbols & pictographs
            r"\U0001F680-\U0001F6FF"   # transport & map symbols
            r"\U0001F1E0-\U0001F1FF"   # flags
            r"\U00002702-\U000027B0"
            r"\U000024C2-\U0001F251]+"
        ),
        "line_breaks": re.compile(r"\n\s*\n+"),
        "trailing_punctuation": re.compile(r"([.!?])\1+"),
    }
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        logger.debug(f"Initialized DialoguePreprocessor with config: {self.config}")
    
    def clean(self, text: str) -> str:
        """
        Clean and normalize dialogue text.
        
        Args:
            text: Raw dialogue text
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not text.strip():
            return ""
        
        # Apply preprocessing steps in order
        text = self._normalize_unicode(text)
        text = self._handle_special_tokens(text)
        text = self._handle_urls(text)
        text = self._normalize_speaker_labels(text)
        text = self._clean_whitespace(text)
        text = self._clean_punctuation(text)
        
        if self.config.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        if not self.config.normalize_unicode:
            return text
        
        import unicodedata
        # Normalize to NFKC form
        text = unicodedata.normalize("NFKC", text)
        return text
    
    def _handle_special_tokens(self, text: str) -> str:
        """Handle SAMSum special tokens like <file_photo>, <file_gif>."""
        if not self.config.handle_special_tokens:
            return text
        
        # Replace special tokens with descriptive text
        replacements = {
            "<file_photo>": "[photo]",
            "<file_gif>": "[gif]",
            "<file_video>": "[video]",
            "<file_audio>": "[audio]",
            "<file_other>": "[file]",
            "<link>": "[link]",
        }
        
        for token, replacement in replacements.items():
            text = text.replace(token, replacement)
        
        # Handle any remaining special tokens
        text = self.PATTERNS["special_token"].sub("[media]", text)
        
        return text
    
    def _handle_urls(self, text: str) -> str:
        """Replace URLs with placeholder."""
        if not self.config.handle_urls:
            return text
        
        return self.PATTERNS["url"].sub("[URL]", text)
    
    def _normalize_speaker_labels(self, text: str) -> str:
        """Normalize speaker label formatting."""
        if not self.config.normalize_speaker_format:
            return text
        
        # Ensure consistent "Speaker: " format
        lines = text.split("\n")
        normalized_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with a speaker label
            match = self.PATTERNS["speaker_label"].match(line)
            if match:
                speaker = match.group(1).strip()
                content = line[match.end():].strip()
                line = f"{speaker}: {content}"
            
            normalized_lines.append(line)
        
        return "\n".join(normalized_lines)
    
    def _clean_whitespace(self, text: str) -> str:
        """Remove extra whitespace while preserving structure."""
        if not self.config.remove_extra_whitespace:
            return text
        
        # Replace multiple spaces with single space (but preserve newlines)
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            # Replace multiple spaces with single space
            line = self.PATTERNS["extra_whitespace"].sub(" ", line)
            cleaned_lines.append(line.strip())
        
        # Remove multiple consecutive empty lines
        text = "\n".join(cleaned_lines)
        text = self.PATTERNS["line_breaks"].sub("\n", text)
        
        return text
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean up punctuation issues."""
        # Replace repeated punctuation (e.g., "!!!" -> "!")
        text = self.PATTERNS["trailing_punctuation"].sub(r"\1", text)
        return text
    
    def batch_clean(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]


def clean_dialogue(
    text: str,
    remove_special_tokens: bool = True,
    normalize_speakers: bool = True,
    lowercase: bool = False
) -> str:
    """
    Convenience function for dialogue cleaning.
    
    Args:
        text: Raw dialogue text
        remove_special_tokens: Whether to handle special tokens
        normalize_speakers: Whether to normalize speaker labels
        lowercase: Whether to lowercase the text
        
    Returns:
        Cleaned dialogue text
        
    Example:
        >>> cleaned = clean_dialogue("Bob:   hey!!  <file_gif>")
        >>> print(cleaned)
        "Bob: hey! [gif]"
    """
    config = PreprocessingConfig(
        handle_special_tokens=remove_special_tokens,
        normalize_speaker_format=normalize_speakers,
        lowercase=lowercase
    )
    preprocessor = DialoguePreprocessor(config)
    return preprocessor.clean(text)


class SummaryPreprocessor:
    """
    Preprocessor for summary text.
    
    Handles cleaning of summary text while maintaining readability.
    """
    
    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase
    
    def clean(self, text: str) -> str:
        """Clean summary text."""
        if not text or not text.strip():
            return ""
        
        # Basic cleaning
        text = " ".join(text.split())  # Normalize whitespace
        
        # Ensure proper sentence ending
        if text and text[-1] not in ".!?":
            text += "."
        
        if self.lowercase:
            text = text.lower()
        
        return text.strip()


def compute_dialogue_statistics(dialogues: List[str]) -> dict:
    """
    Compute statistics about a collection of dialogues.
    
    Args:
        dialogues: List of dialogue texts
        
    Returns:
        Dictionary with statistics
    """
    import statistics
    
    word_counts = []
    char_counts = []
    turn_counts = []
    speaker_counts = []
    
    speaker_pattern = re.compile(r"^([A-Za-z][A-Za-z0-9_\s]*)\s*:", re.MULTILINE)
    
    for dialogue in dialogues:
        # Word and character counts
        words = dialogue.split()
        word_counts.append(len(words))
        char_counts.append(len(dialogue))
        
        # Turn counts (number of lines with speaker labels)
        turns = len(speaker_pattern.findall(dialogue))
        turn_counts.append(turns)
        
        # Unique speakers
        speakers = set(speaker_pattern.findall(dialogue))
        speaker_counts.append(len(speakers))
    
    return {
        "num_dialogues": len(dialogues),
        "words": {
            "mean": statistics.mean(word_counts),
            "median": statistics.median(word_counts),
            "min": min(word_counts),
            "max": max(word_counts),
            "std": statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        },
        "characters": {
            "mean": statistics.mean(char_counts),
            "median": statistics.median(char_counts),
            "min": min(char_counts),
            "max": max(char_counts)
        },
        "turns": {
            "mean": statistics.mean(turn_counts),
            "median": statistics.median(turn_counts),
            "min": min(turn_counts),
            "max": max(turn_counts)
        },
        "speakers": {
            "mean": statistics.mean(speaker_counts),
            "median": statistics.median(speaker_counts),
            "min": min(speaker_counts),
            "max": max(speaker_counts)
        }
    }
