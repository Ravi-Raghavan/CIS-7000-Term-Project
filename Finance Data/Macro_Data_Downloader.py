import yfinance as yf
import pandas as pd
import os

def get_macro_csv_path():
    """Helper to resolve the path for the macro dataset based on your VS Code workspace."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    csv_path = os.path.join(project_root, 'Finance Data', 'Historical_Data_MACRO.csv')
    
    if not os.path.exists(os.path.dirname(csv_path)):
        csv_path = "Finance Data/Historical_Data_MACRO.csv"
    return csv_path

def download_compound_market_data(start_date="2010-01-01"):
    csv_path = get_macro_csv_path()
    print("Downloading diverse index data (SPY, QQQ) from Yahoo Finance...")
    
    try:
        # Download full OHLCV data for SPY and QQQ
        spy = yf.Ticker("SPY").history(start=start_date)[['Open', 'High', 'Low', 'Close', 'Volume']]
        spy = spy.rename(columns=lambda x: f"SPY_{x}")
        
        qqq = yf.Ticker("QQQ").history(start=start_date)[['Open', 'High', 'Low', 'Close', 'Volume']]
        qqq = qqq.rename(columns=lambda x: f"QQQ_{x}")
        
        # Merge and clean
        df = pd.concat([spy, qqq], axis=1).dropna().reset_index()
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        
        # Validation Check: Ensure data isn't empty before saving
        if len(df) < 100:
            print(f"Error: yfinance only returned {len(df)} rows. Download failed or was rate-limited. Please try again later.")
            return
            
        # Save to CSV
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Successfully downloaded {len(df)} days of data and saved to: {csv_path}")
        
    except Exception as e:
        print(f"An error occurred during download: {e}")

if __name__ == "__main__":
    download_compound_market_data()