import pandas as pd
from datetime import date, timedelta


# TODO create a method for joining word vector embeddings with price data


def aggregate_sentiment_scores(subreddit:str):
    # removes unused columns
    comments_df = pd.read_csv(f'dataset/{subreddit}-comments.csv', engine='python', error_bad_lines=False)
    comments_df = comments_df.drop(['comment_id', 'comment_body'], axis=1)
    comments_df = comments_df[['Date', 'neg', 'neu', 'pos', 'compound']]
    comments_df['Date'] = pd.to_datetime(comments_df['Date'], errors='coerce').dt.date
    comments_df = comments_df.groupby(comments_df['Date'])['neg', 'neu', 'pos', 'compound'].mean()
    comments_df = comments_df.reset_index()

    price_df = pd.read_csv('dataset/daily_stock_price_data.csv', engine='python', error_bad_lines=False)
    price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce').dt.date

    training_df = comments_df.merge(price_df, how='left')
    training_df = training_df.fillna(method='bfill')

    # assign Date to index column
    training_df = training_df.set_index('Date')
    # re order columns to group by sentiment score first, then price data, with 'Close' as last column for label
    training_df = training_df.reindex(
        columns=[
            'neg', 'neu', 'pos', 'compound', # sentiment scores
            'Open', 'High', 'Low', 'Adj Close', 'Volume', # price data
            'Close' # label
        ]
    )
    training_df.to_csv('dataset/training_data.csv')
