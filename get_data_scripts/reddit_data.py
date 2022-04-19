# https://trello.com/c/wXZXOQrL/2-praw-api-for-getting-custom-dataset
import pandas as pd
import datetime as dt
from get_data_scripts import reddit_client
from praw.models import MoreComments
from scripts import data_clean, sentiment


def get_post_statistics(subreddit: str, stocks: str, num_posts: str) -> None:
    sub = reddit_client.connect(subreddit)
    submission_statistics = []
    comment_list = []
    for ticker in stocks:
        for submission in sub.search(ticker, limit=num_posts):
            dict_post = {}
            dict_post['ticker'] = ticker
            dict_post['post_id'] = submission.id
            dict_post['Date'] = dt.datetime.fromtimestamp(submission.created_utc)
            dict_post['title'] = submission.title
            dict_post['post_name'] = submission.name
            dict_post['score'] = submission.score
            dict_post['upvote_ratio'] = submission.upvote_ratio
            dict_post['num_comments'] = submission.num_comments
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
                    dict_comment['Date'] = dt.datetime.fromtimestamp(comment.created_utc)
                    dict_comment['comment_body'] = body

                comment_list.append(dict_comment)
                print("new comment added")

    submission_statistics_df = pd.DataFrame(submission_statistics)
    submission_statistics_df.sort_values(by='Date')
    submission_statistics_df.to_csv(f'dataset/{subreddit}-posts.csv')

    comments_df = pd.DataFrame(comment_list)
    sentiment_scores = comments_df['comment_body'].apply(lambda c: pd.Series(sentiment.score_comments(c)))
    comments_df = pd.concat([comments_df, sentiment_scores], axis=1)
    comments_df.sort_values(by='Date')
    comments_df.to_csv(f'dataset/{subreddit}-comments.csv')
