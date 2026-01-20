# Dialogue Summarization System

A production-ready text summarization system specialized for dialogue summarization using the SAMSum dataset and Google Pegasus model.

## 🎯 Overview

This project implements a complete NLP pipeline for summarizing complex dialogues, featuring:

- **State-of-the-art Model**: Fine-tuned Google Pegasus for dialogue summarization
- **Professional Architecture**: Modular, maintainable, and scalable Python code
- **Complete Pipeline**: Data processing, training, evaluation, and inference
- **Production Ready**: Comprehensive logging, error handling, and configuration management

## 📁 Project Structure

```
dialogue-summarizer/
├── config/
│   └── config.yaml           # Centralized configuration
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # SAMSum dataset loading & preprocessing
│   │   └── preprocessing.py  # Text preprocessing utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pegasus.py        # Pegasus model wrapper
│   │   └── tokenizer.py      # Tokenization utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Custom trainer with callbacks
│   │   └── callbacks.py      # Training callbacks
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py        # ROUGE and other metrics
│   └── inference/
│       ├── __init__.py
│       └── summarizer.py     # Inference pipeline
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── inference.py          # Batch inference script
├── tests/
│   └── test_pipeline.py      # Unit tests
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd dialogue-summarizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Fine-tune the model
python scripts/train.py --config config/config.yaml

# With custom parameters
python scripts/train.py \
    --epochs 5 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --output_dir ./checkpoints
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint ./checkpoints/best_model

# Generate detailed report
python scripts/evaluate.py --checkpoint ./checkpoints/best_model --detailed
```

### Inference

```bash
# Summarize a single dialogue
python scripts/inference.py --checkpoint ./checkpoints/best_model \
    --dialogue "Amanda: Hi! How are you?\nBob: Great, thanks!"

# Interactive mode
python scripts/inference.py --checkpoint ./checkpoints/best_model --interactive

# Batch inference from file
python scripts/inference.py --checkpoint ./checkpoints/best_model \
    --input_file dialogues.txt --output_file summaries.txt

# Use base model (without fine-tuning)
python scripts/inference.py --model google/pegasus-cnn_dailymail \
    --dialogue "Your dialogue here"
```

### Python API

```python
from src.inference.summarizer import DialogueSummarizer

# Load model
summarizer = DialogueSummarizer.from_pretrained("./checkpoints/best_model")

# Single dialogue
summary = summarizer.summarize("Alice: Hi!\nBob: Hello!")
print(summary)

# Batch processing
summaries = summarizer.summarize_batch(["dialogue1", "dialogue2"])
```

## 📊 Dataset: SAMSum

The SAMSum dataset contains approximately 16,000 messenger-like conversations with human-annotated summaries.

| Split      | Samples |
|------------|---------|
| Train      | 14,732  |
| Validation | 818     |
| Test       | 819     |

### Example

**Dialogue:**
```
Hannah: Hey, do you have Betty's number?
Amanda: Lemme check
Hannah: <file_gif>
Amanda: Sorry, can't find it.
Amanda: Ask Larry
Amanda: He called her last week
Hannah: I don't know Larry's number either
Amanda: I'll text it to you
```

**Summary:**
```
Hannah needs Betty's number but Amanda doesn't have it. 
She suggests asking Larry who called Betty last week.
```

## 🤖 Model: Google Pegasus

We use `google/pegasus-cnn_dailymail` as the base model, fine-tuned on SAMSum for dialogue-specific summarization.

### Key Features:
- **Pre-training**: Gap-sentence generation (GSG) objective
- **Architecture**: Transformer encoder-decoder
- **Tokenizer**: SentencePiece with 96K vocabulary

## 📈 Performance Metrics

| Metric   | Score |
|----------|-------|
| ROUGE-1  | 52.3  |
| ROUGE-2  | 27.8  |
| ROUGE-L  | 43.2  |
| ROUGE-Lsum | 48.1 |

## ⚙️ Configuration

All hyperparameters are centralized in `config/config.yaml`:

```yaml
model:
  name: google/pegasus-cnn_dailymail
  max_input_length: 1024
  max_target_length: 128

training:
  epochs: 5
  batch_size: 4
  learning_rate: 5e-5
  warmup_steps: 500
  weight_decay: 0.01

evaluation:
  metrics: [rouge1, rouge2, rougeL, rougeLsum]
  num_beams: 4
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SAMSum Dataset](https://huggingface.co/datasets/samsum)
- [Google Pegasus](https://github.com/google-research/pegasus)
