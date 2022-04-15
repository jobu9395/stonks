# https://trello.com/c/wXZXOQrL/2-praw-api-for-getting-custom-dataset
import pandas as pd
import datetime as dt
from get_data_scripts import reddit_client
from praw.models import MoreComments
from scripts import data_clean


# TODO change this from global vars to params passed into `get_post_statistics()`
STOCKS = [
    "GME",
    "AMC"
]
AMOUNT = 500


# def scrape_wikipedia_for_sp_500():
#     payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
#     df = payload[0]
#     ordered_list_of_tickers = df[df.columns[0]]
#     ordered_list_of_tickers.to_csv('dataset/ticker_list.csv')


def get_post_statistics(subreddit: str) -> None:
    sub = reddit_client.connect(subreddit)
    submission_statistics = []
    comment_list = []
    for ticker in STOCKS:
        for submission in sub.search(ticker, limit=AMOUNT):
            dict_post = {}
            dict_post['ticker'] = ticker
            dict_post['post_id'] = submission.id
            dict_post['date'] = dt.datetime.fromtimestamp(submission.created_utc)
            dict_post['title'] = submission.title
            dict_post['post_name'] = submission.name
            dict_post['score'] = submission.score
            dict_post['upvote_ratio'] = submission.upvote_ratio
            dict_post['num_comments'] = submission.num_comments
            if dict_post['date'] > dt.datetime(2017, 4, 1):
                submission_statistics.append(dict_post)

            comments = submission.comments.list()
            comment_set = set()
            for comment in comments:
                if isinstance(comment, MoreComments):
                    continue
                body = data_clean.clean(comment.body)
                if body and comment not in comment_set:
                    comment_set.add(comment)
                    dict_comment = {}
                    dict_comment['comment_ticker'] = ticker
                    dict_comment['comment_id'] = comment
                    dict_comment['date'] = dt.datetime.fromtimestamp(comment.created_utc)
                    dict_comment['comment_body'] = body

                comment_list.append(dict_comment)

    submission_statistics_df = pd.DataFrame(submission_statistics)
    submission_statistics_df.sort_values(by='date')
    submission_statistics_df.to_csv(f'dataset/{subreddit}-posts.csv')

    comments_df = pd.DataFrame(comment_list)
    comments_df.sort_values(by='date')
    comments_df.to_csv(f'dataset/{subreddit}-comments.csv')
