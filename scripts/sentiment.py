import nltk
import pandas as pd
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def score_comments(sentence):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(sentence)
    return scores

# test_comment1 = " ".join(['thats', 'works', '8k', 'filing', 'bit', 'vague', 'didnt', 'see', 'ratio', 'going', 'filing', 'gamestop', 'plans', 'increase', 'authorized', 'shares', '300mm', '1bn', 'shares', 'purpose', 'split', 'didnt', 'actually', 'say', 'split', 'ratio', 'going', 'sort', 'odd', 'increase', 'perhaps', 'plan', 'also', 'raise', 'capital', 'well', 'actually', 'say', 'split', 'seems', 'like', 'theres', 'lot', 'speculation', 'running', 'around\n\na', 'stock', 'split', 'change', 'valuation', 'company', 'term', 'share', 'dividend', 'often', 'used', 'bit', 'misnomer', 'company', 'isnt', 'actually', 'issuing', 'dividend', 'additional', 'value\n\nsplits', 'dont', 'change', 'option', 'valuations', 'either', 'strikes', 'get', 'adjusted\n\nno', 'idea', 'mean', 'shorters', 'aare', 'obligated', 'anyone', 'holding', 'short', 'position', 'simply', 'shares', 'adjusted', 'way', 'long', 'position', 'adjusted'])
# test_comment2 = " ".join(['holy', 'shit', 'thats', '15', 'million', 'away', 'limit', 'one', 'share'])

# df = pd.DataFrame([test_comment1, test_comment2], columns=['comments'])

# test = df['comments'].apply(lambda c: pd.Series(score_comments(c)))
# df = pd.concat([df, test], axis=1)
# print(df)