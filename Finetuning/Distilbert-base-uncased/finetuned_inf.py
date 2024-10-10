from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_name = "distilbert-base-uncased-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to predict sentiment for a single user input text
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

# Predict sentiment
sentiment = predict_sentiment(user_input)
print(f"Sentiment: {sentiment}")
