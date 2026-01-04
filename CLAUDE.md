# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a BERT-based Chinese sentiment analysis system that can accurately identify sentiment polarity (positive/negative) in Chinese text reviews. The project uses PyTorch and the Chinese BERT model (bert-base-chinese) to implement an efficient, high-accuracy text sentiment classifier.

## Repository Structure

```
text_sentiment_classification/
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
├── .gitignore              # Git ignore patterns
├── data/                   # Data directory
│   ├── Data.csv            # Main dataset
│   ├── train_data.csv      # Training data
│   ├── test_data.csv       # Test data
│   └── dev_data.csv        # Validation data
├── src/                    # Source code
│   ├── data/               # Data processing module
│   │   ├── __init__.py
│   │   └── preprocess.py   # Data preprocessing
│   ├── models/             # Model definitions
│   │   ├── __init__.py
│   │   └── bert_classifier.py  # BERT classifier
│   ├── utils/              # Utility modules
│   │   ├── log_utils.py    # Logging utilities
│   │   └── model_loader.py # Model loading utilities
│   ├── train.py            # Basic training script
│   ├── train_advanced.py   # Advanced training with cross-validation and hyperparameter optimization
│   ├── fine_tune_lora.py   # LoRA fine-tuning implementation
│   └── api.py              # API for model serving
├── run.sh                  # Run script for training
└── outputs/                # Output directory
    ├── best_model.pth      # Best model weights
    ├── final_model.pth     # Final model weights
    ├── loss_curve.png      # Loss curve plot
    └── accuracy_curve.png  # Accuracy curve plot
```

## Key Architecture Components

1. **BertClassifier (src/models/bert_classifier.py)**: The main model class that extends PyTorch's nn.Module, using a pre-trained BERT model with a dropout layer and classification head.

2. **ModelLoader (src/utils/model_loader.py)**: Handles loading pre-trained models from HuggingFace or ModelScope with caching capabilities.

3. **TextDataset (src/data/preprocess.py)**: PyTorch Dataset implementation for processing Chinese text with BERT tokenizer.

4. **Training Scripts**: Two main training approaches - basic training (train.py) and advanced training with cross-validation and hyperparameter optimization (train_advanced.py).

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Training Commands
```bash
# Basic training
python src/train.py

# Advanced training with cross-validation and hyperparameter optimization
python src/train_advanced.py --epochs 10 --batch_size 16 --learning_rate 2e-5

# Using the run script (uses train_advanced.py)
bash run.sh
```

### Model Serving
```bash
# Start API server
python src/api.py
```

### Fine-tuning with LoRA
```bash
# Fine-tune using LoRA
python src/fine_tune_lora.py
```

## Model Architecture

The system uses a pre-trained Chinese BERT model (bert-base-chinese) with these components:
1. **BERT Encoder**: Converts Chinese text to 768-dimensional vector representations
2. **Dropout Layer**: Prevents overfitting (default dropout rate 0.3)
3. **Classification Layer**: Linear layer mapping 768-dim vectors to 2 classes (positive/negative)

## Key Features

1. **Advanced Training Techniques**:
   - K-fold cross-validation
   - Hyperparameter optimization using Optuna
   - Ensemble models
   - Early stopping
   - Learning rate scheduling
   - Gradient clipping

2. **Data Processing**:
   - Chinese text tokenization using BERT tokenizer
   - Max sequence length of 128 tokens
   - Data augmentation (partially implemented)

3. **Performance Optimizations**:
   - Supports MPS acceleration on Apple Silicon Macs
   - GPU support for CUDA-enabled devices
   - Model caching to speed up loading
   - Batch processing for efficient training

4. **Monitoring & Evaluation**:
   - Training and validation loss curves
   - Accuracy tracking
   - Detailed classification reports
   - SwanLab (similar to Weights & Biases) integration for experiment tracking

## Prediction Usage

To use the trained model for sentiment prediction:

```python
from transformers import BertTokenizer
import torch
from src.models.bert_classifier import BertClassifier

# Load model
model = BertClassifier('bert-base-chinese')
model.load_state_dict(torch.load('outputs/best_model.pth'))
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def predict_sentiment(text):
    # Preprocess text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Predict
    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])
        _, predicted = torch.max(outputs, 1)

    return "正面评价" if predicted.item() == 1 else "负面评价"

# Example
text = "服务态度很好，送餐很快，味道也不错！"
result = predict_sentiment(text)
print(f"评论: {text}")
print(f"情感分析结果: {result}")
```

## Common Issues and Notes

1. The `train_advanced.py` file now properly handles command-line arguments and logs metrics to SwanLab.

2. The project supports both HuggingFace and ModelScope models through the ModelLoader utility.

3. The system is optimized for Chinese text processing and uses the `bert-base-chinese` model by default.

4. SwanLab metrics logged during training include train_loss_batch, train_loss_epoch, val_loss, val_f1, and best_val_f1_so_far.

## Dependencies

The project requires PyTorch, Transformers, HuggingFace tokenizers, pandas, numpy, scikit-learn, and other common ML libraries. See requirements.txt for the complete list.