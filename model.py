from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

# Download model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save them locally
model.save_pretrained("amazon_model")
tokenizer.save_pretrained("amazon_model")

print("Model downloaded successfully!")
