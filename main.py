from get_data_scripts.reddit_data import get_post_statistics
from get_data_scripts.yahoo_finance_data import get_daily_stock_prices


def main():
    get_post_statistics("wallstreetbets")
    get_daily_stock_prices()
    get_post_statistics("investing")


if __name__ == "__main__":
    main()
