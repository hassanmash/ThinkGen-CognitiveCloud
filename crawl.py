
import http.client
# scrapped_data = data.decode("utf-8")
import json
import streamlit as st


product_name = st.text_input("Enter some text:")



# Display the user inputs
st.write("You entered:")
st.write(f"Text: {product_name}")

conn = http.client.HTTPSConnection("real-time-amazon-data.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "9f575f6728msh6cc537587896654p10c882jsnb51889add3b6",
    'x-rapidapi-host': "real-time-amazon-data.p.rapidapi.com"
}

# conn.request("GET", "/product-reviews?asin=B0CB1FYVJZ&country=IN&sort_by=TOP_REVIEWS&star_rating=ALL&verified_purchases_only=false&images_or_videos_only=false&current_format_only=false&page=1", headers=headers)

# conn.request("GET", f"/search?query={product_name}&page=1&country=IN&sort_by=RELEVANCE&product_condition=ALL&is_prime=false&deals_and_discounts=NONE", headers=headers)

# res = conn.getresponse()
# data = res.read()
scrapped_data = {
    "status": "OK",
    "request_id": "9d6fbfb7-93fa-4429-a2c1-f8bce988ed3c",
    "parameters": {
        "query": "phone",
        "country": "US",
        "sort_by": "RELEVANCE",
        "page": 1,
    },
    "data": {
        "total_products": 148376,
        "country": "US",
        "domain": "www.amazon.com",
        "products": [
            {
                "asin": "B0CTW8TXGH",
                "product_title": "Tracfone | Motorola Moto g Play 2024 | Locked | 64GB | 5000mAh Battery | 50MP Quad Pixel Camera | 6.5-in. HD+ 90Hz Display | Sapphire Blue",
                "product_price": "$39.88",
                "product_original_price": "$49.99",
                "currency": "USD",
                "product_star_rating": "4.4",
                "product_num_ratings": 308,
                "product_url": "https://www.amazon.com/dp/B0CTW8TXGH",
                "product_photo": "https://m.media-amazon.com/images/I/71CxUvG46rL._AC_UY654_FMwebp_QL65_.jpg",
                "product_num_offers": 1,
                "product_minimum_offer_price": "$39.88",
                "sales_volume": "3K+ bought in past month",
                "delivery": "FREE delivery Mon, Feb 17 Or fastest delivery Tomorrow, Feb 13",
            },
            {
                "asin": "B0CRWT7WF1",
                "product_title": "Samsung Galaxy A05 A055M 64GB Dual-SIM GSM Unlocked Android Smartphone (Latin America Version) - Black",
                "product_price": "$87.99",
                "product_original_price": "$94.49",
                "currency": "USD",
                "product_star_rating": "4.2",
                "product_num_ratings": 403,
                "product_url": "https://www.amazon.com/dp/B0CRWT7WF1",
                "product_photo": "https://m.media-amazon.com/images/I/61UjBLFlH2L._AC_UY654_FMwebp_QL65_.jpg",
                "product_num_offers": 15,
                "product_minimum_offer_price": "$75.00",
                "sales_volume": "1K+ bought in past month",
                "delivery": "FREE delivery Mon, Feb 17 Or fastest delivery Fri, Feb 14",
            }
        ]
    }
}

scrapped_reviewed_data = {
    "status": "OK",
    "request_id": "21196b86-eda0-4cec-af28-36d52796933d",
    "parameters": {
        "asin": "B0CB1FYVJZ",
        "country": "IN",
        "sort_by": "TOP_REVIEWS",
        "star_rating": "ALL",
        "page": 1
    },
    "data": {
        "asin": "B0CB1FYVJZ",
        "total_reviews": 282,
        "total_ratings": 572,
        "country": "IN",
        "domain": "www.amazon.in",
        "reviews": [
            {
                "review_id": "R43GHYYHKU5KQ",
                "review_title": "Best for office users",
                "review_comment": "I recently purchased the **Offbeat Atom Dual Bluetooth Mouse** from Amazon, and I'm delighted with my choice. Here's why:\n\n1. **Performance**: This mouse is lightweight, smooth, and silent. The Bluetooth connection is rapid, and I've experienced no lag. It's perfect for both work and leisure.\n\n2. **Battery Life**: The built-in rechargeable battery lasts impressively long. After an initial full charge, it lasted around 10 days with 8-9 hours of daily use. No more frequent battery changes!\n\n3. **Dual Device Connectivity**: The Atom Dual Mouse can seamlessly connect to two devices simultaneously. I paired it with my iPad and laptop via Bluetooth, and switching between them is a breeze.\n\n4. **Sleek Design**: The superior finish and appearance make it stand out. Plus, the Type-C charging port adds convenience.\n\nOverall, the Offbeat Atom Dual Bluetooth Mouse combines functionality, style, and value. Highly recommended! ¹",
                "review_star_rating": "5",
                "review_link": "https://www.amazon.in/gp/customer-reviews/R43GHYYHKU5KQ",
                "review_author": "Param Kumawat",
                "review_author_avatar": "https://images-eu.ssl-images-amazon.com/images/S/amazon-avatars-global/default._CR0,0,1024,1024_SX48_.png",
                "review_images": [
                    "https://m.media-amazon.com/images/I/51qt3hQddbL._SY88.jpg",
                    "https://m.media-amazon.com/images/I/615FHtfNzuL._SY88.jpg"
                ],
                "review_date": "Reviewed in India on 17 July 2024",
                "is_verified_purchase": True,
                "helpful_vote_statement": "5 people found this helpful",
                "reviewed_product_asin": "B0CQJSW9DG"
            },
            {
                "review_id": "R1WG4010NAYWYL",
                "review_title": "Pretty good alternative to the Magic Mouse!",
                "review_comment": "Just got it yesterday. Have been using it for only a day so can’t comment on longevity and battery. First impressions:-\nPros:\n-The clicks are really muffled and silent. Good design job.\n-It’s very light in weight.\n-Good fit and finish. Mat rubber finish on the top helps with good grip.\n-The scroll wheel has nurled finish for better grip.\n-Pretty easy to use. Just click on the mode switch to switch between usb and Bluetooth.\n-Charging was quick and got fully charged in less than 2 hours, out of the box that is. Charging post use, I’ll update later.\n-Works best without a mouse pad/mat (this will be a Con if you have a desk mat) the pointer movement becomes slow and glitchy on a mouse pad.\n-Auto off/sleep feature is pretty nice. It wakes up on a click.\n-Paired easily and quickly without any glitch’s to my MacBook.\nCons:\n-Although the form factor is the same as the Magic Mouse, it’s not ergonomic and doesn’t snuggly fit into your palms just like the Magic Mouse.\n-Works best without a mousepad. One star\nLess for this. I sometimes have to use it outside the desk mat due to this issue.\n- The “mode switch” should have been provided at the bottom instead of top as it is now. You end up clicking the mouse unintentionally when you try to switch between the modes due to this.\n- Felt the length to be slightly longer than my palm. That could again be an ergonomic issue.\n-Can’t comment on battery life yet.\n-Yet to try multiple Bluetooth device connections.",
                "review_star_rating": "4",
                "review_link": "https://www.amazon.in/gp/customer-reviews/R1WG4010NAYWYL",
                "review_author": "Placeholder",
                "review_author_avatar": "https://images-eu.ssl-images-amazon.com/images/S/amazon-avatars-global/default._CR0,0,1024,1024_SX48_.png",
                "review_images": [
                    "https://m.media-amazon.com/images/I/619JNOBN-TL._SY88.jpg",
                    "https://m.media-amazon.com/images/I/61Z3FB78roL._SY88.jpg",
                    "https://m.media-amazon.com/images/I/61lCT9BSDgL._SY88.jpg",
                    "https://m.media-amazon.com/images/I/61lS5IXYLNL._SY88.jpg"
                ],
                "review_date": "Reviewed in India on 24 January 2024",
                "is_verified_purchase": True,
                "helpful_vote_statement": "9 people found this helpful",
                "reviewed_product_asin": "B0CB1FYVJZ"
            }]
    }
}

def extract_review_comments(scrapped_reviewed_data):
    reviews = scrapped_reviewed_data["data"]["reviews"]
    review_comments = [review["review_comment"] for review in reviews]
    return review_comments

# Extracting specified properties
product_list = [
    {
        "asin": item["asin"],
        "title": item["product_title"],
        "price": item["product_price"],
        "image": item["product_photo"],
        "product_url": item["product_url"]
    }
    for item in scrapped_data["data"]["products"]
]

# Print the extracted data
# print(json.dumps(product_list, indent=2))
# print(product_list)

def navigate_to_product(asin):
    print(asin)
    response = extract_review_comments(scrapped_reviewed_data)
    print(response)

#conn.request("GET", f"/product-reviews?asin={asin}&country=IN&sort_by=TOP_REVIEWS&star_rating=ALL&verified_purchases_only=false&images_or_videos_only=false&current_format_only=false&page=1", headers=headers)

# conn.request("GET", f"/search?query={product_name}&page=1&country=IN&sort_by=RELEVANCE&product_condition=ALL&is_prime=false&deals_and_discounts=NONE", headers=headers)

# res = conn.getresponse()
# data = res.read()
# Streamlit UI
# Initialize session state to keep track of selected product
if "selected_product" not in st.session_state:
    st.session_state["selected_product"] = None

# Function to reset the selected product
def reset_selected_product():
    st.session_state["selected_product"] = None

placeholder = st.empty()

if st.session_state["selected_product"] is None:
    with placeholder.container():
        st.title("🛍️ Product List")
        for product in product_list:
            col1, col2, col3 = st.columns([2, 5, 3])  # Layout columns
            with col2:  # Center column for card
                with st.container():
                    st.image(product["image"], use_column_width=True)  # Display product image
                    st.markdown(f"### {product['title']}")  # Title
                    st.write(f"💰 **Price:** {product['price']}")  # Price
                    if st.button(f"🔗 View Product", key=product["asin"]):
                        st.session_state["selected_product"] = product
else:
    product = st.session_state["selected_product"]
    with placeholder.container():
        st.image(product["image"], use_column_width=True)
        st.markdown(f"### {product['title']}")
        st.write(f"💰 **Price:** {product['price']}")
        st.write(f"**Description:** Detailed description of {product['title']}")
        if st.button("🔙 Back to Product List"):
            reset_selected_product()
