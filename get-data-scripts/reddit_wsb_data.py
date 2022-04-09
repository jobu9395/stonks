import os
import praw
import pandas as pd
from reddit import connect

# sets up reddit client
reddit = connect()  ## to use this, make a Reddit app. Client ID is in top left corner, client secret is given, and user agent is the username that the app is under

subreddit = reddit.subreddit("wallstreetbets")

# sets up
payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
print(payload)
first_table = payload[0]
second_table = payload[1]
df = first_table
ordered_list_of_tickers = df[df.columns[0]]
string_list_of_tickers = []
for element in ordered_list_of_tickers:
    string_list_of_tickers.append('' + element + '')


# print(string_list_of_tickers)
ordered_list_of_tickers.to_csv('ticker_list.csv', index=False)


def ticker_list():
    return string_list_of_tickers
