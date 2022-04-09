import os
import praw
import pandas as pd
from reddit import connect, Subreddit


def get_s_and_b_data():
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
    ordered_list_of_tickers.to_csv('./datasets/ticker_list.csv', index=False)

def get_wsb_data():
    ''' Comments, '''
    pass

def ticker_list():
    return string_list_of_tickers

subreddit = Subreddit(time_frame=0, name="investing")
connection = subreddit.connect_subreddit()
print(connection)
