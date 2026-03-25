import pandas as pd
import matplotlib.pyplot as plt
import glob
import matplotlib.dates as mdates
import seaborn as sns

# Use seaborn style
sns.set(style="whitegrid")

# Find all CSV files
csv_files = glob.glob("Historical_Data_*.csv")

plt.figure(figsize=(14, 8))

for file in csv_files:
    # Extract stock name
    stock_name = file.split("_")[-1].replace(".csv", "")
    
    # Load CSV
    df = pd.read_csv(file)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
    
    # Sort by Date and take last 1000 rows
    df = df.sort_values('Date')[-1000:]
    
    # Plot with thicker lines
    plt.plot(df['Date'], df['Open'], label=stock_name, linewidth=2)

# Formatting
plt.title("Opening Price Time Series for All Stocks", fontsize=18)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Opening Price (USD)", fontsize=14)

# Rotate and format x-axis ticks for better readability
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save image
plt.savefig("All_Stocks_Opening_Price.png", dpi=300)
print("Saved as All_Stocks_Opening_Price.png")