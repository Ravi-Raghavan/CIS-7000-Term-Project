import pandas as pd
import os
# Load the CSV file into a DataFrame
csv_path = os.path.join('Sentiment Analysis', 'all_musk_posts.csv')
quotes_csv_path = os.path.join('Sentiment Analysis', 'musk_quote_tweets.csv')
if os.path.exists(csv_path) and os.path.exists(quotes_csv_path):
    tweets_df = pd.read_csv(csv_path, low_memory=False)
    quotes_df = pd.read_csv(quotes_csv_path, low_memory=False)

    print(tweets_df.columns)
    print(quotes_df.columns)

    print(tweets_df.info())
    print(quotes_df.info())

    tweets_df['createdAt'] = pd.to_datetime(tweets_df['createdAt'])
    sorted_df = tweets_df.sort_values(by='createdAt', ascending=False)
    columns_to_see = ['fullText', 'createdAt']
    #print(tweets_df.head())  # Display the first few rows of the DataFrame
    print(sorted_df[columns_to_see].head(10))  # Display the column names
    print(sorted_df[columns_to_see].tail(10))  # Display the last 10 rows of the selected columns