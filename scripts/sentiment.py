import nltk
import pandas as pd
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def score_comments(sentence):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(sentence)
    return scores
