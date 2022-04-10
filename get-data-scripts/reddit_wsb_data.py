# https://trello.com/c/wXZXOQrL/2-praw-api-for-getting-custom-dataset
import pandas as pd
import reddit_client
import datetime as dt
from praw.models import MoreComments

SUBREDDIT = reddit_client.connect("wallstreetbets")
STOCKS = [
    "GME",
    "AMC",
    "PLTR",
    "CRSR",
    "VOO",
]


# sets up
def scrape_wikipedia_for_sp_500():
    payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = payload[0]
    ordered_list_of_tickers = df[df.columns[0]]
    ordered_list_of_tickers.to_csv('dataset/ticker_list.csv')


def get_post_statistics():
    submission_statistics = []
    comment_list = []
    for ticker in STOCKS:
        for submission in SUBREDDIT.search(ticker, limit=100):
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

            comments = submission.comments.list()
            for comment in comments:
                if isinstance(comment, MoreComments):
                    continue
                d_comment = {}
                d_comment['comment_ticker'] = ticker
                d_comment['comment_id'] = comment
                d_comment['comment_body'] = comment.body
                d_comment['date'] = dt.datetime.fromtimestamp(comment.created_utc)
                comment_list.append(d_comment)

    submission_statistics_df = pd.DataFrame(submission_statistics)
    submission_statistics_df.sort_values(by='date')
    submission_statistics_df.to_csv('dataset/posts.csv')

    comments_df = pd.DataFrame(comment_list)
    comments_df.sort_values(by='date')
    comments_df.to_csv('dataset/comments.csv')


if __name__ == "__main__":
    get_post_statistics()
