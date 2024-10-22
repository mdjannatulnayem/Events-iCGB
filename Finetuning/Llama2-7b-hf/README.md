---
library_name: transformers
tags:
- peft
- trl
- torch
- wandb
- ipex
license: apache-2.0
language:
- en
base_model:
- NousResearch/Llama-2-7b-hf
datasets:
- mlabonne/mini-platypus
pipeline_tag: text-generation
---


# Model Card for Fine-Tuned Llama-2-7b Model

## Model Details

### Model Description

This model is a fine-tuned version of the Llama-2-7b model, specifically adapted for causal language modeling tasks. The fine-tuning utilizes the PEFT (Parameter-Efficient Fine-Tuning) technique with LoRA (Low-Rank Adaptation) to optimize performance while reducing computational costs. The training was conducted using the `mlabonne/mini-platypus` dataset and incorporates features such as integration with W&B for experiment tracking and Intel's Extension for PyTorch (IPEX) for enhanced performance.

- **Developed by:** Md. Jannatul Nayem
- **Model type:** Causal Language Model
- **Language(s) (NLP):** Engish
- **License:** Apache 2.0
- **Finetuned from model :** NousResearch/Llama-2-7b-hf

## Uses

### Direct Use

The model can be utilized for text generation tasks where the generation of coherent and contextually relevant text is required. This includes applications like chatbots, content creation, and interactive storytelling.

### Downstream Use

When fine-tuned, this model can serve in larger ecosystems for tasks like personalized dialogue systems, question answering, and other natural language understanding applications.

### Out-of-Scope Use

The model is not intended for use in generating harmful or misleading content, and users should exercise caution to prevent misuse in sensitive areas such as misinformation or hate speech.

### Recommendations

Users should consider implementing bias mitigation strategies and ensure thorough evaluation of the model's outputs, especially in sensitive applications.

## How to Get Started with the Model

Use the following code snippet to get started with loading and using the model:

```python
# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import intel_extension_for_pytorch as ipex  # Optional for Intel optimization

# Specify your Hugging Face model repository
hf_model = "nayem-ng/mdjannatulnayem_llama2_7b_finetuned_casuallm_lora"

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained(hf_model)
tokenizer = AutoTokenizer.from_pretrained(hf_model)

# Move the model to the desired device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Set the model to evaluation mode
model.eval()

# Optional: Optimize with Intel extensions for PyTorch
# Uncomment the next line if you want to use Intel optimizations
# model = ipex.optimize(model)

# Function to generate text
def generate_text(prompt, max_length=50):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)

    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    prompt = "Once upon a time"
    generated_text = generate_text(prompt)
    print("Generated Text:", generated_text)
```

## Training Details

### Training Data

The model was fine-tuned using the mlabonne/mini-platypus dataset, which consists of diverse text inputs designed to enhance the model's capabilities in conversational settings.

[mlabonne/mini-platypus](https://huggingface.co/datasets/mlabonne/mini-platypus)

### Training Procedure

The training utilized a supervised fine-tuning procedure with the following hyperparameters:

#### Training Hyperparameters

The model was trained using bfloat16 (bf16) mixed precision, which allows for faster training times and reduced memory usage compared to traditional fp32 (float32). This precision format is particularly beneficial when working with large models, as it helps to maintain numerical stability while optimizing performance on compatible hardware.

- Training regime: bf16 mixed precision
- Number of epochs: 1
- Batch size: 10
- Warmup steps: 10
- Gradient accumulation steps: 1
- Learning rate: 2e-4
- Warmup steps: 10
- Evaluation strategy: Evaluations are performed every 1000 steps to monitor the model's performance during training.


## Model Examination

Further interpretability studies can be conducted to understand decision-making processes within the model's responses.

### Model Architecture and Objective

The model is based on the Transformer architecture, specifically designed for Causal Language Modeling (CLM).

### Compute Infrastructure

Intel® Tiber™ AI Cloud

#### Hardware

Intel(R) Xeon(R) Platinum 8480+

#### Software

PyTorch, Transformers Library (from Hugging Face),PEFT, TRL, WandB, Intel Extension for PyTorch (IPEX)

## Model Card Contact

🤖 Md. Jannatul nayem | [Mail](nayemalimran106@gmail.com) | [LinkedIn](https://www.linkedin.com/in/md-jannatul-nayem) 
