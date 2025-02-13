########################working code 

# from agno.agent import Agent
# from groq import Groq
# from textblob import TextBlob  # Sentiment Analysis Library
# import os

# # Set API Key (Optional, if not using environment variable)
# # os.environ["GROQ_API_KEY"] = "your-groq-api-key"  # Replace with your actual key

# # Initialize Groq Model
# groq_model =Groq(api_key="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")

# # Define a sentiment analysis tool
# class SentimentAnalysisTool:
#     def analyze(self, review_text):
#         """Performs sentiment analysis on the given review."""
#         analysis = TextBlob(review_text)
#         sentiment_score = analysis.sentiment.polarity  # -1 to 1 scale
#         sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
#         return {"review": review_text, "sentiment": sentiment, "score": sentiment_score}

# # Initialize Sentiment Analysis Tool
# sentiment_tool = SentimentAnalysisTool()

# # Create Agnos Agent
# agent = Agent(
#     name="Amazon Review Sentiment Analysis",
#     model=groq_model,  
#     tools=[sentiment_tool],  # Using sentiment analysis instead of FeedbackFive()
#     instructions=["Analyze Amazon reviews and determine sentiment"],
#     markdown=True
# )

# # Example: Sending a review to the agent
# review_text = "This product is not amazing! The quality is top-notch, and I love using it every day." 
# review_text = "This product is worst! The quality is bad."

# # Get sentiment analysis result
# result = agent.print_response(review_text)
# # result = sentiment_tool.analyze(review_text)

# # Print response
# print(f"Review: {result['review']}\nSentiment: {result['sentiment']} (Score: {result['score']})")

##################################################33


import groq
import json
import re
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq  # Correct import for Groq models

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key="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",  # Use the correct Groq API key
    # model_name="llama3-70b-8192"  # Adjust model name as per availability
    # model_name="Mixtral-8x7b"  # Adjust model name as per availability
    model_name="llama-3.3-70b-versatile"  # Adjust model name as per availability
)

def extract_json_from_prompt(prompt):
    """Extracts JSON data from the prompt using regex"""
    match = re.search(r'\{.*\}', prompt, re.DOTALL)
    if match:
        json_data = match.group(0)
        try:
            return json.loads(json_data)
        except json.JSONDecodeError:
            return json.loads("{}")
    else:
        return json.loads("{}")

def analyze_sentiment(reviews):
    client = groq.Client(api_key="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")
    
    prompt = f"""
    You are an AI that analyzes Amazon product reviews. Based on the reviews provided, generate the following:
    
    1. Overall sentiment (Positive/Negative/Neutral)
    2. Key positive aspects
    3. Key negative aspects
    4. Recurring issues
    5. Recommendation (Use/Do Not Use)
    6. Suggestions to improve the product
    7. Also give trend analysis
    
    Reviews:
    {json.dumps(reviews, indent=2)}
    
    Provide a structured JSON response with keys: sentiment, positive_aspects, negative_aspects, recurring_issues, and recommendation.
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    result = response.choices[0].message.content.strip()
    # print(result)
    # return json.loads(result)
    
    
    # Extract JSON content using regex
    return extract_json_from_prompt(result)

# Function to Analyze Sales Trends
def analyze_trends(sales_data):
    client = groq.Client(api_key="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")

    """
    Analyzes product sales trends and provides insights.
    """
    prompt = f"""
    You are an expert in trend analysis. Based on the given sales data, analyze the following:

    1. Best-selling products.
    2. Sales trends (rising, stable, declining).
    3. Seasonal patterns (if any).
    4. Market insights and recommendations.

    Sales Data (JSON format):
    {json.dumps(sales_data, indent=2)}

    IMPORTANT: Return ONLY a structured JSON response, without any extra text.
    Example format:
    {{
      "best_selling_products": ["Product A", "Product B"],
      "trends": {{
        "rising": ["Product C"],
        "stable": ["Product D"],
        "declining": ["Product E"]
      }},
      "seasonal_patterns": "High demand for winter-related products",
      "market_insights": "Increase in demand for eco-friendly products."
    }}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    result = response.choices[0].message.content.strip()

    # Extract JSON using regex
    return extract_json_from_prompt(result)
    
def get_summary_in_json(data):
    client = groq.Client(api_key="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")

    """
    Analyzes the summary and get it in json format.
    """
    prompt = f"""
    You are an expert in getting json from a summary. Based on the given summary, give me data in following JSON format accordingly:

    Format 1:
    ```
        {{
            "sentiment": "something",
            "positive_aspects": ["something"],
            "negative_aspects": ["something"],
            "recurring_issues": ["something"],
            "recommendation": "something",
            "suggestions_to_improve": ["something"]
        }}
    ```
    Format 2:
    ```
        {{
            "best_selling_products": ["something"],
            "trends": {{
                "stable": ["something"],
                "declining": ["something"]
            }},
            "seasonal_patterns": "something",
            "market_insights": "something"
        }}
    ```
    1. Best-selling products.
    2. Sales trends (rising, stable, declining).
    3. Seasonal patterns (if any).
    4. Market insights and recommendations.

    Summary:
    {data}

    IMPORTANT: Return ONLY a structured JSON response, without any extra text.
    """

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    result = response.choices[0].message.content.strip()

    # Extract JSON using regex
    match = re.search(r'\{.*\}', result, re.DOTALL)
    if match:
        json_data = match.group(0)
        return json.loads(json_data)
    else:
        raise ValueError("No valid JSON found in the response.")

# Define Trend Analysis Tool
trend_tool = Tool(
    name="Sales Trend Analysis Tool",
    func=analyze_trends,
    description="Analyzes product sales data to identify best-sellers, trends, and market insights."
)

sentiment_tool = Tool(
    name="Sentiment Analysis Tool",
    func=analyze_sentiment,
    description="Analyzes Amazon reviews to determine sentiment, pros, cons, recurring issues, and recommendations."
)

agent = initialize_agent(
    tools=[sentiment_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

trendAgent = initialize_agent(
    tools=[trend_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

class MasterAgent:
    def __init__(self, sentiment_agent, trend_agent):
        self.sentiment_agent = sentiment_agent
        self.trend_agent = trend_agent

    def analyze_reviews(self, reviews):
        sentiments = self.sentiment_agent.run(reviews)
        trends = self.trend_agent.run(reviews)
        return sentiments, trends

# Example Usage
if __name__ == "__main__":
    
    # analysis = analyze_sentiment(sample_reviews)
    # print(json.dumps(analysis, indent=2))

    reviews =  [
        "This product is amazing! It works perfectly and exceeded my expectations.",
        "The quality is good, but the battery life is shorter than advertised.",
        "Terrible experience. The item arrived broken and customer service was unhelpful.",
        "Good value for money. Would recommend!",
        {"product": "Wireless Earbuds pro", "sales": "5K+ bought in past month"},
        {"product": "Wireless Earbuds", "sales": "3K+ bought in past month"},
        {"product": "Wired Earbuds 10m", "sales": "2.5K+ bought in past month"},
        {"product": "Wired Earbuds 3m", "sales": "4K+ bought in past month"},
        {"product": "Speaker", "sales": "1.2K+ bought in past month"}
    ]
    master_agent = MasterAgent(agent, trendAgent)
    sentiments, trends = master_agent.analyze_reviews(reviews)
    print("Sentiments:", sentiments)
    print("Trends:", trends)

    # response1 = agent1.print_response(f"Analyze the sentiment of these reviews: {json.dumps(sample_reviews)}")
    # print(response1)
    analysis = get_summary_in_json(sentiments)
    print("sentiments")
    print(json.dumps(analysis, indent=2))
    print("---------")
    analysis = get_summary_in_json(trends)
    print("trends")
    print(json.dumps(analysis, indent=2))
    print("---------")

    # Define the task
    # message = f"""Analyze the sentiment of these reviews: {json.dumps(reviews)}. 
    #     Give the details as a summary and also in JSON format.
    #     Example JSON:
    #     ```
    #     {{
    #         "sentiment": "something",
    #         "positive_aspects": ["something"],
    #         "negative_aspects": ["something"],
    #         "recurring_issues": ["something"],
    #         "recommendation": "something",
    #         "suggestions_to_improve": ["something"]
    #     }}
    #     ```
    # """
