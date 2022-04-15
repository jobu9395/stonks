import yfinance as yf
import pandas as pd
import numpy as np



STOCKS =[
    "AMC"
]


def get_daily_stock_prices():
    # gets daily stock data for
    data = yf.download(
        tickers=" ".join(STOCKS),
        period="10y",
        interval="1d",
    )

    data.to_csv('dataset/daily_stock_prices_all_numbers.csv')
    print(data)


get_daily_stock_prices()


