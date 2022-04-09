import praw
import os
from dotenv import load_dotenv

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
