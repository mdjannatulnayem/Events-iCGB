from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import torch
import os
import wandb
import intel_extension_for_pytorch as ipex

# Model
base_model = "NousResearch/Llama-2-7b-hf"
new_model = "Llama2_7B_Finetuned_CasualLM"
hf_model = "mdjannatulnayem_llama2_7b_finetuned_casuallm_lora"
hf_token = os.getenv('HUGGINGFACE_TOKEN')

# Dataset
dataset = load_dataset("mlabonne/mini-platypus", split="train")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

# Hardware
device = "xpu:0" if torch.xpu.is_available() else "cpu"
print(f"Using device: {device}")

# Load base moodel
model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
model.config.use_cache = False 

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj',
                     'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Set training arguments
training_arguments = TrainingArguments(
    use_ipex=True,
    num_train_epochs=1,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=1,
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=1,
    bf16=True,
    learning_rate=2e-4,
    warmup_steps=10,
    output_dir="./results",
    report_to="wandb",
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="instruction",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

def print_training_summary(results):
    print(f"Time: {results.metrics['train_runtime']: .2f}")
    print(f"Samples/second: {results.metrics['train_samples_per_second']: .2f}")

# Train model
results = trainer.train()
wandb.finish()
print("Finetuning complete!")
print_training_summary(results)

# Save trained model
trainer.model.save_pretrained(new_model)

# Reload model and merge it with LoRA weights
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()
model.save_pretrained(hf_model)

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Push to HuggingFace
model.push_to_hub(hf_model, use_temp_dir=False, token=hf_token)
tokenizer.push_to_hub(hf_model, use_temp_dir=False, token=hf_token)

# Optimize with ipex
# model = ipex.optimize_transformers(model)

print("Done")
