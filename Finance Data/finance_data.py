# Import yfinance
import yfinance as yf

# Fetch Sample Stock Ticker
ticker = yf.Ticker("TSLA")

historical_asset_data = ticker.history(period="20y")
print(historical_asset_data)