import pandas as pd
from datetime import date, timedelta


# TODO don't drop all but one stock, use dataframe pivot on STOCK global var, change this script to method
# TODO make method accept list of stocks and then join specific comments to pivoted price data by stock
def join_yfinance_with_reddit_comments():
    comments_df = pd.read_csv('dataset/wallstreetbets-comments.csv')
    comments_df['Date'] = pd.to_datetime(comments_df['Date']).dt.date

    # only bring in comments for AMC
    comments_df = comments_df[comments_df['comment_ticker'] == 'AMC']

    # drops all price data except for AMC's, drops the "Date" column and keeps the "date" columns to enable proper join
    price_df = pd.read_csv('dataset/daily_stock_prices.csv')
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    price_df = price_df.drop(['Date', 'GME'], axis=1)

    # left join to include non trading day's comments
    training_df = comments_df.merge(
        price_df,
        how='left',
    )

    print(f"\ndata stats pre backfill of prices: \n \n {training_df.describe()}")

    # TODO experiment with forward fill
    # back fills price data
    training_df = training_df.fillna(method='bfill')

    # further cleaning, uses date as index column
    training_df = training_df.drop(['comment_ticker', 'comment_id'], axis=1)
    training_df = training_df[['Date', 'comment_body', 'AMC']]
    training_df['price'] = training_df['AMC']
    training_df = training_df.drop('AMC', axis=1)
    training_df = training_df.set_index('Date')
    training_df.index = pd.to_datetime(training_df.index)

    print(f"\ndata stats post backfill of prices, with further cleaning: \n \n {training_df.describe()}")

    print("\ntraining dataframe head")
    print(training_df.head())

    training_df.to_csv('dataset/training_data.csv')



def aggregate_sentiment_scores():
    # removes unused columns
    comments_df = pd.read_csv('dataset/wallstreetbets-comments.csv')
    df = comments_df.drop(['comment_id', 'comment_body'], axis=1)
    df = df.iloc[: , 1:]
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    # aggreages scores by day
    df = df.groupby(df['Date'])['neg', 'neu', 'pos', 'compound'].mean()
    print(df.columns, df.index)
    # load yfinance data
    price_df = pd.read_csv('dataset/daily_stock_price_data.csv')
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    print(price_df.columns, price_df.index)
    training_df = df.merge(price_df, how='left')
    training_df = training_df.fillna(method='bfill')
    print(training_df.head())


if __name__ == "__main__":
    # join_yfinance_with_reddit_comments()
    aggregate_sentiment_scores()
