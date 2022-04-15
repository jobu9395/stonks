from get_data_scripts.reddit_data import get_post_statistics
from get_data_scripts.yahoo_finance_data import get_daily_stock_prices
from scripts.training_join import join_yfinance_with_reddit_comments


def main():
    get_post_statistics("wallstreetbets")
    get_daily_stock_prices()
    join_yfinance_with_reddit_comments()


if __name__ == "__main__":
    main()
