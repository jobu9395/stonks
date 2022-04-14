import yfinance as yf
import pandas as pd
import numpy as np

STOCKS =[
    "GME",
    "AMC"
]

def get_daily_stock_prices():
    # gets daily stock data for
    data = yf.download(
        tickers=" ".join(STOCKS),
        period="5y",
        interval="1d",
    )

    adjusted_daily_closes = data['Adj Close'][STOCKS]
    adjusted_daily_closes.to_csv('dataset/daily_stock_prices.csv')
