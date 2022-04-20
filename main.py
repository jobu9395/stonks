# import models.train
from get_data_scripts.reddit_data import get_post_statistics
from scripts.training_join import aggregate_sentiment_scores
from get_data_scripts.yahoo_finance_data import get_daily_stock_prices
from models.rnn import train_model
import warnings

warnings.filterwarnings('ignore')

SUBREDDIT = "wallstreetbets"
STOCKS = [
    'AMC',
]
NUM_POSTS = 500


def main():
    # get_post_statistics(subreddit=SUBREDDIT, stocks=STOCKS, num_posts=NUM_POSTS)
    # get_daily_stock_prices(stocks=STOCKS)
    aggregate_sentiment_scores(subreddit=SUBREDDIT)
    print(f"scraped stock data for: {STOCKS}, training model")
    train_model()
    # models.train.main()


if __name__ == "__main__":
    main()
