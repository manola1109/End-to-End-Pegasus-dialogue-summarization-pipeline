# 🤖 End-to-End Dialogue Summarization with Pegasus

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready **text summarization system** for complex dialogues using **Google Pegasus** fine-tuned on the **SAMSum dataset**.

## 📊 Results

| Metric | Score |
|--------|-------|
| **ROUGE-1** | 51.14% |
| **ROUGE-2** | 25.79% |
| **ROUGE-L** | 41.76% |

## ✨ Features

- **State-of-the-art Model**: Google Pegasus fine-tuned for dialogue summarization
- **Production Ready**: Modular architecture with comprehensive logging
- **GPU Optimized**: Mixed precision (FP16) training with gradient accumulation
- **Complete Pipeline**: Data processing → Training → Evaluation → Inference
- **Interactive Mode**: Real-time dialogue summarization via CLI

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/manola1109/End-to-End-Pegasus-dialogue-summarization-pipeline.git
cd End-to-End-Pegasus-dialogue-summarization-pipeline
pip install -r requirements.txt
```

### Training
```bash
python scripts/train.py --config config/config.yaml
```

### Inference
```bash
python scripts/inference.py --checkpoint ./checkpoints/best_model --interactive
```

## 💬 Example

**Input:**
```
Alice: Hey, did you finish the project?
Bob: Almost! Just need to review the code.
Alice: Great, let's meet at 3pm to discuss.
Bob: Sounds good, see you then!
```

**Output:**
```
Bob finished the project. He will meet with Alice at 3 pm to discuss it.
```

## 📁 Project Structure
```
├── config/config.yaml           # Configuration
├── src/
│   ├── data/                    # Dataset loading & preprocessing
│   ├── models/                  # Pegasus model wrapper
│   ├── training/                # Trainer with callbacks
│   ├── evaluation/              # ROUGE metrics
│   └── inference/               # Production inference
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
└── requirements.txt
```

## 🤖 Model

- **Base Model**: google/pegasus-cnn_dailymail
- **Parameters**: 570M
- **Dataset**: SAMSum (14,732 training samples)

## 🖥️ Hardware

- GPU: NVIDIA RTX 4070 Ti SUPER (16GB)
- Training Time: ~8 hours

## 📧 Contact

**Deepak Singh** - [@manola1109](https://github.com/manola1109)

⭐ Star this repo if you found it helpful!
