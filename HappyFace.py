# import requests

# # ashish-001/Amazon_Review_Sentiment
# url = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
# headers = {"Authorization": "Bearer hf_defftxDAEkgnadfakeupwfEXUuSYpKOQzGvjFOPiMWAThisisfake"}

# data = {"inputs": "The product seems to havee been used a lot. I wanted new one not refurbished"}
# response = requests.post(url, headers=headers, json=data)

# print(response.json())

# hassanmashmoom/Amazon_Review_Sentiment_cognitive
import requests

# Replace with your actual API URL
api_url = "https://hassanmashmoom-amazon-review-sentiment-cognitive.hf.space/analyze"

# Sample review text
review_text = "Customers find the mouse useful for productive workflows. They appreciate its sleek design and good connectivity. The build quality is praised as high-quality, with an impressive finish and appearance. The mouse's smooth scrolling and clicks are also appreciated."

# Prepare payload
payload = {"text": review_text}

# Send POST request
response = requests.post(api_url, json=payload)

# Print response
print(response.json())
