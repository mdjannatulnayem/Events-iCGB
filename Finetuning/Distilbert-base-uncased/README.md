
# DistilBERT Text Classification

This repository contains implementations and fine-tuned models for text classification using DistilBERT. It includes scripts for both fine-tuning the model and performing inference.

## Table of Contents

- [Getting Started](#getting-started)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)

**Dependencies**: 
- `transformers`
- `torch`
- `datasets`
- `scikit-learn`
- `pandas`
- `numpy`

## Getting Started

1. **Fine-Tuning the Model**:
   To fine-tune the DistilBERT model on your text classification task, run the following command:

   ```bash
   python3 llm_finetuning.py
   ```

2. **Performing Inference**:
   After fine-tuning, you can perform inference using the fine-tuned model:

   - For the fine-tuned model:

     ```bash
     python3 finetuned_inf.py
     ```

   - For the non-fine-tuned model:

     ```bash
     python3 not_finetuned_inf.py
     ```

## File Descriptions

- **`finetuned_inf.py`**: Script for inference with the fine-tuned DistilBERT model.
- **`not_finetuned_inf.py`**: Script for inference with the base DistilBERT model.
- **`llm_finetuning.py`**: Script for fine-tuning the DistilBERT model on a text classification task.
- **`distilbert-base-uncased-finetuned`**: Directory containing the fine-tuned DistilBERT model files.

## Usage

### Fine-Tuning

You can customize the training process by modifying the parameters in `llm_finetuning.py`, such as learning rate, batch size, and number of epochs.

### Inference

Use the inference scripts ending with `_inf` notation.
