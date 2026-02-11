## Not allowing me to query historical data

import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MARKET_AUX_API_KEY")

symbol = "AAPL"
from_date = "2016-02-10"  # start date
to_date = "2016-02-10"    # end date (same for single day)

url = (
    f"https://api.marketaux.com/v1/news/all?"
    f"symbols={symbol}&"
    f"from={from_date}&"
    f"to={to_date}&"
    f"api_token={API_KEY}"
)

response = requests.get(url)
data = response.json()

for article in data.get("data", []):
    print(f"Title: {article['title']}")
    print(f"Sentiment: {article.get('sentiment', 'N/A')}")
    print(f"Published: {article['published_at']}")
    print("---")