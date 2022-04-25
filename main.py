# import models.train
from get_data_scripts.reddit_data import get_post_statistics
from scripts.training_join import aggregate_sentiment_scores
from get_data_scripts.yahoo_finance_data import get_daily_stock_prices
from models.lstm_no_log import train_model
import warnings
warnings.filterwarnings('ignore')

SUBREDDIT = "wallstreetbets"
STOCKS = [
    'AMC',
]
NUM_POSTS = 500

# join option can be either 'trading_day_granularity' or 'all_comments_granularity'
# OPTION = 'trading_day_granularity'
OPTION = 'all_comments_granularity'


def main():
    get_post_statistics(subreddit=SUBREDDIT, stocks=STOCKS, num_posts=NUM_POSTS)
    get_daily_stock_prices(stocks=STOCKS)
    print(f"Scraped daily stock data for the following stock: {STOCKS}.")
    aggregate_sentiment_scores(subreddit=SUBREDDIT, option=OPTION)
    print(f"Joined data at the {OPTION} for the following stock: {STOCKS}, now training model...")
    train_model()
    print("Model succesfully trained.")


if __name__ == "__main__":
    main()
