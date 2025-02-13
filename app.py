import http.client
import streamlit as st
import os
import json
import streamlit as st
import matplotlib.pyplot as plt

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab') 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from agno.agent import Agent
from agno.models.groq import Groq
import groq
import json
import re
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq  # Correct import for Groq models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import base64
# from langchain.globals import set_llm_cache
# from langchain.cache import InMemoryCache

# set_llm_cache(InMemoryCache())  # 🚀 Prevents redundant API calls

import streamlit as st
import matplotlib.pyplot as plt

conn = http.client.HTTPSConnection("real-time-amazon-data.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "414e80138cmsh335f3feba6eb6b1p154dbejsn4d19da621060",
    'x-rapidapi-host': "real-time-amazon-data.p.rapidapi.com"
}

st.set_page_config(page_title="Amazon Product Reviews Analysis")
import urllib.parse

def convert_to_url_friendly(text):
    # Convert spaces to hyphens
    url_friendly_text = text.replace(' ', '-')
    
    # Encode the URL
    url_friendly_text = urllib.parse.quote(url_friendly_text)
    
    return url_friendly_text

def get_product_list_by_name(product_name):
    conn.request("GET", f"/search?query={product_name}&page=1&country=IN&sort_by=RELEVANCE&product_condition=ALL&is_prime=false&deals_and_discounts=NONE", headers=headers)
    res = conn.getresponse()
    scrapped_data = res.read()

    print(f'{scrapped_data}')
    # Check if scrapped_data is in bytes and decode it if necessary
    if isinstance(scrapped_data, bytes):
        scrapped_data = scrapped_data.decode('utf-8')
    
    # Parse the JSON data
    scrapped_data = json.loads(scrapped_data)
    # Process the products
    scrapped_data_formatted = []
    for item in scrapped_data["data"]["products"]:
        # Your logic to process each product
        scrapped_data_formatted.append(item)

    matching_products = get_product_list(scrapped_data_formatted)
    return matching_products

# Extracting specified properties
def get_product_list(scrapped_data_formatted):
    product_list = [
        {
            "asin": item["asin"],
            "title": item["product_title"],
            "price": item["product_price"],
            "image": item["product_photo"],
            "product_url": item["product_url"]
        }
        for item in scrapped_data_formatted
    ]
    return product_list

def get_reviews_list_by_asin(asin):
    conn.request("GET", f"/product-reviews?asin={asin}&country=IN&sort_by=TOP_REVIEWS&star_rating=ALL&verified_purchases_only=false&images_or_videos_only=false&current_format_only=false&page=1", headers=headers)
    res = conn.getresponse()
    scrapped_reviewed_data = res.read()
    
    # Check if scrapped_reviewed_data is in bytes and decode it if necessary
    if isinstance(scrapped_reviewed_data, bytes):
        scrapped_reviewed_data = scrapped_reviewed_data.decode('utf-8')
    
    # Parse the JSON data
    scrapped_data_json = json.loads(scrapped_reviewed_data)

    # Process the reviews
    scrapped_reviewed_data_formatted = []
    for item in scrapped_data_json["data"]["reviews"][:10]:  # Get only the first 10 reviews
        # Your logic to process each review
        scrapped_reviewed_data_formatted.append(item)

    product_reviews = extract_review_comments(scrapped_reviewed_data_formatted)
    return product_reviews

def extract_review_comments(scrapped_reviewed_data):
    review_comments = [review["review_comment"] for review in scrapped_reviewed_data]
    return review_comments


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
def preprocess_reviews(reviews):
    preprocessed_reviews = []
    for review in reviews:
        # Check if review is a string
        if isinstance(review, str):
            review = re.sub(r'[^a-zA-Z\s]', '', review)
            preprocessed_reviews.append(review)
        else:
            print(f"Skipping non-string review: {review} (type: {type(review)})")
    return preprocessed_reviews



def display_product_card(product, small=False):
    size = "small" if small else "normal"
    st.markdown(
        f"""
        <div style='display: flex; flex-direction: column; align-items: center; 
                     border: 1px solid lightgray; border-radius: 10px; padding: {5 if small else 10}px; 
                     margin: 10px; box-sizing: border-box; background-color: #f9f9f9;'>
            <img src='{product["image"]}' style='max-width: 100%; height: {75 if small else 150}px;'>
            <h4 style='text-align: center; font-size: {12 if small else 16}px;'>{product["title"]}</h4>
            <p style='text-align: center; font-size: {12 if small else 14}px;'><strong>Price:</strong> {product["price"]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if "selected_product" not in st.session_state:
        st.session_state.selected_product = None
    if not small:
        if st.button("View Review Sentiment", key=f"button_{product['asin']}"):
            st.session_state.selected_product = product
            st.rerun()  # Forces Streamlit to refresh the UI immediately

import matplotlib.pyplot as plt
import matplotlib.text as mtext

def show_final_analysis(data):
    st.title("Product Review Analysis")

    # Display Nuanced Emotions
    st.header("Nuanced Emotion")
    st.write(data['sentiment'])
    # data['sentiment']
    # Display Pros and Cons
    st.subheader("Positive Aspects")
    st.write(", ".join(data["positive_aspects"]))

    st.subheader("Negative Aspects")
    st.write(", ".join(data["negative_aspects"]))

    # Display Recurring Issues
    st.header("Area of Improvements")
    st.write(", ".join(data["recurring_issues"]))

    # Pie chart for sentiment analysis
    st.header("Sentiment Analysis")
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [len(data["positive_aspects"]), len(data["negative_aspects"]), 1]  # Assuming 1 neutral aspect for demo
    colors = ['green', 'red', 'blue']
    explode = (0.1, 0, 0)  # explode the 1st slice (Positive)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)

    # Display Recommendation in a separate card
    st.markdown(
        f"""
        <div style='display: flex; flex-direction: column; align-items: center; 
                     border: 1px solid lightgray; border-radius: 10px; padding: 15px; 
                     margin: 15px; box-sizing: border-box; background-color: #f0f0f0; 
                     box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);'>
            <h4 style='text-align: center; margin-bottom: 10px;'>Overall product recommendation</h4>
            <p>{data['recommendation']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def extract_text_part(string_with_json):
    # Split the input string to separate the text part from the JSON part
    text_part = string_with_json.split("```")[0]
    return text_part
    
def display_selected_product(product):
    st.header(f"Sentiment Analysis for {product['title']}")
    reviews_for_product = get_reviews_list_by_asin(product['asin'])
    preprocessed_reviews = preprocess_reviews(reviews_for_product)
    
    # Un-comment and analyze reviews
    review_analysis = analyze_reviews(preprocessed_reviews)
    display_product_card(product, small=True)
    
 
    reveiw_summary_hassan = getsentientAnalysis(preprocessed_reviews)
    print(f'ffff{reveiw_summary_hassan}')
    review_json = extract_json_from_prompt(reveiw_summary_hassan)
    # Display review summary
    st.header("Review Summary")  
    st.write(extract_text_part(reveiw_summary_hassan))

    # st.write(review_json)
    show_final_analysis(review_json)
    return preprocessed_reviews

# Summarize reviews and analyze sentiments
def analyze_reviews(reviews):
    summary = "Review Summary: \n"
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    emotions = []
    
    for review_text in reviews:
        # Analyze sentiment (dummy sentiment analysis for illustration)
        if "good" in review_text or "excellent" in review_text or "phenomenal" in review_text:
            positive_count += 1
            emotions.append("😃 (Positive)")
        elif "poor" in review_text or "bad" in review_text:
            negative_count += 1
            emotions.append("😔 (Negative)")
        else:
            neutral_count += 1
            emotions.append("😐 (Neutral)")
    
    total_reviews = positive_count + negative_count + neutral_count
    
    summary += f"Total Reviews: {total_reviews}\n"
    summary += f"Positive: {positive_count}\n"
    summary += f"Negative: {negative_count}\n"
    summary += f"Neutral: {neutral_count}\n"
    
    return {
        "summary": summary,
        "emotions": emotions,
        "sentiments": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count
        }
    }




groq_api_key1 = "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"
groq_api_key2 = ""

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=f"{groq_api_key1}",  # Use the correct Groq API key
    model_name="gemma2-9b-it"  # Adjust model name as per availability
)

def analyze_sentiment(reviews):
    client = groq.Client(api_key=f"{groq_api_key1}")
    #     2. Key positive aspects
    # 3. Key negative aspects
    # 4. Recurring issues
    # 5. Recommendation (Use/Do Not Use)
    # 6. Suggestions to improve the product
    # 7. Also give trend analysis
    
    prompt = f"""
    You are an AI that analyzes Amazon product reviews. Based on the reviews provided, generate the following:
    
    1. Overall sentiment (Positive/Negative/Neutral)

    
    Reviews:
    {json.dumps(reviews, indent=2)}
    
    Provide a structured JSON response with keys: sentiment, positive_aspects, negative_aspects, recurring_issues, and recommendation.
    """
    
    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    result = response.choices[0].message.content.strip()
    # print(result)
    # return json.loads(result)
    
    
    # Extract JSON content using regex
    return extract_json_from_prompt(result)

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
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    result = response.choices[0].message.content.strip()

    # Extract JSON using regex
    return extract_json_from_prompt(result)


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


def main():
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = None

    if 'product_list' not in st.session_state:
        st.session_state.product_list = []

    sample_review = None  # Initialize sample_review

    if st.session_state.selected_product:
        sample_review = display_selected_product(st.session_state.selected_product)
    else:
        product_name = st.text_input("Enter product name:", autocomplete = "off")
        if st.button('Search'):
            urlProductName = convert_to_url_friendly(product_name)
            product_list = get_product_list_by_name(urlProductName)
            st.session_state.product_list = product_list

        if st.session_state.product_list:
            st.header("List of Products")
            num_columns = 3  # Number of columns in each row
            for i in range(0, len(st.session_state.product_list), num_columns):
                cols = st.columns(num_columns)
                for j, col in enumerate(cols):
                    if i + j < len(st.session_state.product_list):
                        with col:
                            product = st.session_state.product_list[i + j]
                            display_product_card(product)
                            if product == st.session_state.selected_product:
                                sample_review = display_selected_product(product)
    return sample_review

def getsentientAnalysis(list_reviews):
    sample_reviews = list_reviews

    sample_sales_data = [
        {"product": "Wireless Earbuds", "sales": "5K+ bought in past month"},
        {"product": "Gaming Laptop", "sales": "3K+ bought in past month"},
        {"product": "Smartwatch", "sales": "2.5K+ bought in past month"},
        {"product": "Air Fryer", "sales": "4K+ bought in past month"},
        {"product": "Electric Scooter", "sales": "1.2K+ bought in past month"}
    ]
    
    # analysis = analyze_sentiment(sample_reviews)
    # print(json.dumps(analysis, indent=2))

    sentimentResponse = agent.run(f"""Analyze the sentiment of these reviews: {json.dumps(sample_reviews)}. 
                                Give the details as a summary and also in JSON format.
                                Example JSON:
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
                                """)

    # trendResponse = trendAgent.run(f"""Analyze the sales trend of these products: {json.dumps(sample_sales_data)}. 
    #                             Give the details as a summary and also in JSON format.
    #                             Example JSON:
    #                             ```
    #                             {{
    #                                 "best_selling_products": ["something"],
    #                                 "trends": {{
    #                                     "stable": ["something"],
    #                                     "declining": ["something"]
    #                                 }},
    #                                 "seasonal_patterns": "something",
    #                                 "market_insights": "something"
    #                             }}
    #                             ```
    #                             """)

    
    # print(trendResponse)

    # response1 = agent1.print_response(f"Analyze the sentiment of these reviews: {json.dumps(sample_reviews)}")
    # print(response1)

    # Define the task
    # message = f"""Analyze the sentiment of these reviews: {json.dumps(sample_reviews)}. 
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
    return sentimentResponse


# Example Usage
if __name__ == "__main__":
    main()
   
