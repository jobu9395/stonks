# import models.train
from get_data_scripts.reddit_data import get_post_statistics
from scripts.training_join import join_yfinance_with_reddit_comments, aggregate_sentiment_scores
from get_data_scripts.yahoo_finance_data import get_daily_stock_prices
from models.rnn import train_model


SUBREDDIT = "wallstreetbets"
STOCKS = [
    'AMC',
]
NUM_POSTS = 500


def main():
    # get_post_statistics(subreddit="wallstreetbets", stocks=STOCKS, num_posts=500)
    # get_daily_stock_prices(stocks=STOCKS)
    # join_yfinance_with_reddit_comments()
    aggregate_sentiment_scores()
    print(f"scraped stock data for: {STOCKS}, training model")
    train_model()
    # models.train.main()


if __name__ == "__main__":
    main()
