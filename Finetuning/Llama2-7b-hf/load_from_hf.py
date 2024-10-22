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
