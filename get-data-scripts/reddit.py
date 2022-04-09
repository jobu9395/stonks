import praw
import os
from dotenv import load_dotenv

class Subreddit:
    """Connect to a specific subreddit"""
    def __init__(self, time_frame, name):
        self.time_frame = time_frame
        self.name = name
        self.reddit = connect()

    # sets up reddit client
    def connect_subreddit(self):
        subreddit = self.reddit.subreddit(self.name)
        return subreddit

def connect():
    """Connect to reddit"""

    load_dotenv()

    client_id = os.getenv('client_id')
    client_secret = os.getenv('client_secret')
    user_agent = os.getenv('user_agent')
    username = os.getenv('username')
    password = os.getenv('password')

    reddit = None
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            username=username,
            password=password
        )
    except:
        print("Could not connect to Reddit. ")

    return reddit
