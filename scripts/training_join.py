import pandas as pd
from datetime import date, timedelta


# TODO don't drop all but one stock, use dataframe pivot on STOCK global var, change this script to method
# TODO make method accept list of stocks and then join specific comments to pivoted price data by stock
def join_yfinance_with_reddit_comments():
    comments_df = pd.read_csv('dataset/wallstreetbets-comments.csv')
    comments_df['date'] = pd.to_datetime(comments_df['date']).dt.date

    # only bring in comments for AMC
    comments_df = comments_df[comments_df['comment_ticker'] == 'AMC']

    # drops all price data except for AMC's, drops the "Date" column and keeps the "date" columns to enable proper join
    price_df = pd.read_csv('dataset/daily_stock_prices.csv')
    price_df['date'] = pd.to_datetime(price_df['Date']).dt.date
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
    training_df = training_df[['date', 'comment_body', 'AMC']]
    training_df['price'] = training_df['AMC']
    training_df = training_df.drop('AMC', axis=1)
    training_df = training_df.set_index('date')
    training_df.index = pd.to_datetime(training_df.index)

    print(f"\ndata stats post backfill of prices, with further cleaning: \n \n {training_df.describe()}")

    print("\ntraining dataframe head")
    print(training_df.head())

    training_df.to_csv('dataset/training_data.csv')


if __name__ == "__main__":
    join_yfinance_with_reddit_comments()
