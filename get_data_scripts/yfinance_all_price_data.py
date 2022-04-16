import yfinance as yf
import pandas as pd
import numpy as np


def get_daily_stock_prices(stock_tickers):
    data = yf.download(
        tickers=" ".join(stock_tickers),
        period="10y",
        interval="1d",
    )

    data.to_csv('dataset/daily_stock_price_data.csv')
    print(data)
