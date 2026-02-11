# Library Imports
import yfinance as yf
import pandas as pd

# List of stock tickers
tickers = ["AAPL", "NVDA", "TSLA", "META", "GOOG"]

# Container for all articles
all_articles = []

# Iterate through each stock ticker
for symbol in tickers:
    # Fetch Ticker Object
    ticker_obj = yf.Ticker(symbol)
    
    # Fetch News for that stock
    try:
        news_items = ticker_obj.news
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        continue
    
    # For each news article, fetch title + url
    for item in news_items:
        content = item.get("content", {})
        title = content.get("title")
        url = content.get("canonicalUrl", {}).get("url")
        pubDate = content.get("pubDate")

        if title and url:  # skip empty entries
            all_articles.append({
                "ticker": symbol,
                "title": title,
                "url": url,
                "pubDate": pubDate
            })

# Final DataFrame
df_final = pd.DataFrame(all_articles)

# Display
print(df_final.head(30))
print(f"Total articles collected: {len(df_final)}")