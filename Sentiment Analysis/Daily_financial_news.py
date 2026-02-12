import pandas as pd
import os
# Load the CSV file into a DataFrame
#news_df = pd.read_csv( 'analyst_ratings_processed.csv', low_memory=False)
#print(news_df.columns)
#print(news_df.info())
#news_df['date'] = pd.to_datetime(news_df['date'], utc=True, errors='coerce')
#news_df['date_only'] = news_df['date'].dt.date
#sorted_df = news_df.sort_values(by='date_only', ascending=False)
target_tickers = ['GOOG', 'NVDA', 'META', 'TSLA', 'AAPL']
#filtered_df = sorted_df[sorted_df['stock'].isin(target_tickers)].copy()
#print(filtered_df['stock'].value_counts()) # Shows how many rows per company
#columns_to_see = ['date_only', 'date', 'title', 'stock']
#print(filtered_df[columns_to_see].head(10))  # Display the first few rows of the DataFrame
#print(filtered_df[columns_to_see].tail(10))  # Display the last

raw_df = pd.read_csv('raw_analyst_ratings.csv', low_memory=False)
print(raw_df.columns)
raw_filtered_df = raw_df[raw_df['stock'].isin(target_tickers)].copy()
print(raw_filtered_df['stock'].value_counts()) # Shows how many rows per company
raw_filtered_df['date'] = pd.to_datetime(raw_filtered_df['date'], utc=True, errors='coerce')
raw_filtered_df['date_only'] = raw_filtered_df['date'].dt.date
raw_sorted_df = raw_filtered_df.sort_values(by='date_only', ascending=False)
columns_to_see = ['headline', 'date', 'url', 'stock']
print(raw_sorted_df[columns_to_see].head(10))  # Display the first few rows of the DataFrame
print(raw_sorted_df[columns_to_see].tail(10))  # Display the last