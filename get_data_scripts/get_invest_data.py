import os
import praw
from dotenv import load_dotenv
from reddit import Reddit, Subreddit
from scripts import data_clean

load_dotenv()

client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
user_agent = os.getenv('user_agent')
username = os.getenv('username')
password = os.getenv('password')
print(username)
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

investments = reddit.subreddit("investments").hot(limit=25)

for p in investments:
    print(p)