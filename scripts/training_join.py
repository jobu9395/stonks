import pandas as pd

comments_df = pd.read_csv('dataset/wallstreetbets-comments.csv')
comments_df['date'] = pd.to_datetime(comments_df['date']).dt.date

# TODO change this so that you can join stock comments on
# only bring in comments for AMC
comments_df = comments_df[comments_df['comment_ticker'] == 'AMC']

print("comments dataframe")
print(comments_df.head())

# TODO don't drop all but one stock, use dataframe pivot on STOCK global var, change this script to method
price_df = pd.read_csv('dataset/daily_stock_prices.csv')
price_df['date'] = pd.to_datetime(price_df['Date']).dt.date
price_df = price_df.drop(['Date', 'GME', 'PLTR', 'CRSR', 'VOO'], axis=1)

print("price dataframe")
print(price_df.head())


training_df = comments_df.merge(
    price_df,
    how='inner',
)

training_df = training_df.drop(['comment_ticker', 'comment_id', 'distinguished_comment'], axis=1)
training_df = training_df[['date', 'comment_body', 'AMC']]
training_df['price'] = training_df['AMC']
training_df = training_df.drop('AMC', axis=1)

print("training dataframe head")
print(training_df.head(n=30))
print(len(training_df))

training_df.to_csv('dataset/training_data.csv')
