import models.train
from get_data_scripts.reddit_data import get_post_statistics
# from get_data_scripts.yahoo_finance_data import get_daily_stock_prices
from scripts.training_join import join_yfinance_with_reddit_comments
from get_data_scripts.yfinance_all_price_data import get_daily_stock_prices
# from models.rnn import train_model


STOCKS = ['AMC']


def main():
    get_post_statistics("wallstreetbets")
    get_daily_stock_prices(STOCKS)
    # print(f"scraped stock data for: {STOCKS}, training model")
    # train_model()
    models.train.main()


if __name__ == "__main__":
    main()
