from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the pre-trained distilbert-base-uncased model and tokenizer
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# Function to predict sentiment for a single user input text (before fine-tuning)
def predict_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class
    predictions = torch.argmax(outputs.logits, dim=1)

    return "Positive" if predictions.item() == 1 else "Negative"

# Get text input from user
user_input = input("Enter a sentence or paragraph for sentiment analysis: ")

# Predict sentiment using the pre-trained (non-finetuned) DistilBERT model
sentiment = predict_sentiment(user_input)
print(f"Sentiment: {sentiment}")
