# https://trello.com/c/wXZXOQrL/2-praw-api-for-getting-custom-dataset
import pandas as pd
import reddit_client
import datetime as dt

SUBREDDIT = reddit_client.connect("wallstreetbets")
STOCKS = ["GME", "AMC", "PLTR", "CRSR", "VOO", "IRNT", "RIOT"]


# sets up
def scrape_wikipedia_for_sp_500():
    payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = payload[0]
    ordered_list_of_tickers = df[df.columns[0]]
    ordered_list_of_tickers.to_csv('dataset/ticker_list.csv')


def get_post_statistics():
    submission_statistics = []
    for ticker in STOCKS:
        for submission in SUBREDDIT.search(ticker, limit=10):
            d = {}
            d['ticker'] = ticker
            d['num_comments'] = submission.num_comments
            d['score'] = submission.score
            d['upvote_ratio'] = submission.upvote_ratio
            d['date'] = dt.datetime.fromtimestamp(submission.created_utc)
            d['domain'] = submission.domain
            d['num_crossposts'] = submission.num_crossposts
            d['author'] = submission.author
            submission_statistics.append(d)

    submission_statistics_df = pd.DataFrame(submission_statistics)
    submission_statistics_df.to_csv('dataset/test_pltr.csv')


if __name__ == "__main__":
    get_post_statistics()
