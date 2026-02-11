# Import yfinance
import yfinance as yf

# List of stock tickers
tickers = ["AAPL", "NVDA", "TSLA", "META", "GOOG"]

# Fetch Historical Data for each ticker
for ticker in tickers:
    yf_ticker = yf.Ticker(ticker)
    historical_asset_data = yf_ticker.history(period="50y")
    historical_asset_data.to_csv(f"Historical_Data_{ticker}.csv")