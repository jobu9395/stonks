import pandas as pd
from datetime import date, timedelta
import warnings

warnings.filterwarnings('ignore')


# TODO create a method for joining word vector embeddings with price data
def aggregate_embeddings(subreddit: str) -> None:
    comments_df = pd.read_csv(f'dataset/{subreddit}-comments.csv', engine='python', error_bad_lines=False)
    comments_df = comments_df.drop(['comment_id'], axis=1)
    comments_df = comments_df[['Date', 'comment_body']]
    
    comments_df['Date'] = pd.to_datetime(comments_df['Date'], errors='coerce').dt.date
    # comments_df = comments_df.groupby(comments_df['Date'])['neg', 'neu', 'pos', 'compound'].mean()
    comments_df = comments_df.reset_index()

    price_df = pd.read_csv('dataset/daily_stock_price_data.csv', engine='python', error_bad_lines=False)
    price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce').dt.date
    
    training_df = price_df.merge(comments_df, how='right')
    training_df = training_df.fillna(method='bfill')
    
    training_df = training_df.set_index('Date')
    training_df = training_df.drop('Adj Close', axis=1)
    # re order columns to group by sentiment score first, then price data, with 'Close' as last column for label
    training_df = training_df.reindex(
        columns=[
            'Close',  # label
            'comment_body',
        ]
    )
    training_df = training_df.sort_values(by='Date')
    training_df.to_csv('dataset/training_data.csv')


def aggregate_sentiment_scores(subreddit:str, option:str):
    # removes unused columns
    comments_df = pd.read_csv(f'dataset/{subreddit}-comments.csv', engine='python', error_bad_lines=False)
    comments_df = comments_df.drop(['comment_id', 'comment_body'], axis=1)
    comments_df = comments_df[['Date', 'neg', 'neu', 'pos', 'compound']]
    comments_df['Date'] = pd.to_datetime(comments_df['Date'], errors='coerce').dt.date

    if option == 'trading_day_granularity':
        comments_df = comments_df.groupby(comments_df['Date'])['neg', 'neu', 'pos', 'compound'].mean()
    comments_df = comments_df.reset_index()

    price_df = pd.read_csv('dataset/daily_stock_price_data.csv', engine='python', error_bad_lines=False)
    price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce').dt.date

    if option == 'trading_day_granularity':
        training_df = price_df.merge(comments_df, how='left')
    elif option == 'all_comments_granularity':
        training_df = comments_df.merge(price_df, how='left')

    training_df = training_df.fillna(method='bfill')
    # assign Date to index column, drop 'adj close' column to prevent data leakage
    training_df = training_df.set_index('Date')
    training_df = training_df.drop('Adj Close', axis=1)
    # re order columns to group by sentiment score first, then price data, with 'Close' as last column for label
    training_df = training_df.reindex(
        columns=[
            'Close',  # label
            'neg', 'neu', 'pos', 'compound', # sentiment scores
            'Open', 'High', 'Low', 'Volume', # price data
        ]
    )
    if option == 'trading_day_granularity':
        training_df.to_csv('dataset/training_data.csv')
    elif option == 'all_comments_granularity':
        training_df.to_csv('dataset/training_data_all_comments.csv')
