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
        for submission in SUBREDDIT.search(ticker, limit=500):
            dict_post = {}
            dict_post['ticker'] = ticker
            dict_post['post_id'] = submission.id
            dict_post['date'] = dt.datetime.fromtimestamp(submission.created_utc)
            dict_post['title'] = submission.title
            dict_post['post_name'] = submission.name
            dict_post['score'] = submission.score
            dict_post['upvote_ratio'] = submission.upvote_ratio
            dict_post['num_comments'] = submission.num_comments
            dict_post['distiniguished_post'] = submission.distinguished
            submission_statistics.append(dict_post)

            comments = submission.comments.list()
            for comment in comments:
                if isinstance(comment, MoreComments):
                    continue
                dict_comment = {}
                dict_comment['comment_ticker'] = ticker
                dict_comment['comment_id'] = comment
                dict_comment['date'] = dt.datetime.fromtimestamp(comment.created_utc)
                dict_comment['comment_body'] = comment.body
                dict_comment['distinguished_comment'] = comment.distinguished
                comment_list.append(dict_comment)

    submission_statistics_df = pd.DataFrame(submission_statistics)
    submission_statistics_df.sort_values(by='date')
    submission_statistics_df.to_csv('dataset/posts.csv')

    comments_df = pd.DataFrame(comment_list)
    comments_df.sort_values(by='date')
    comments_df.to_csv('dataset/comments.csv')


if __name__ == "__main__":
    get_post_statistics()
