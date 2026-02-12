# Plot Opening Price Time Series for All Stocks
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Find all CSV files in the folder
csv_files = glob.glob("Historical_Data_*.csv")

plt.figure(figsize=(12, 7))

for file in csv_files:
    # Extract stock name from filename
    stock_name = file.split("_")[-1].replace(".csv", "")
    
    # Load CSV
    df = pd.read_csv(file)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
    
    # Sort by Date
    df = df.sort_values('Date')
    
    # Plot Opening price
    plt.plot(df['Date'], df['Open'], label=stock_name)

# Labels and formatting
plt.title("Opening Price Time Series for All Stocks")
plt.xlabel("Date")
plt.ylabel("Opening Price")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Save image
plt.savefig("All_Stocks_Opening_Price.png")
print("Saved as All_Stocks_Opening_Price.png")