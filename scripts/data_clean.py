import nltk
import re
nltk.download("stopwords")

from string import punctuation
from typing import List
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
STOPWORDS.update(['deleted', '', 'edit', '\n'])
WORD_LENGTH = 16


def clean(sentence: str) -> List[str]:
    """Cleans up sentences by removing stop words and punctuation"""
    # remove emojois
    demojied = remove_emojis(sentence)
    words = demojied.translate(str.maketrans('', '', punctuation)).split(' ')
    return " ".join([w.lower() for w in words if w.lower() not in STOPWORDS and len(w.lower()) < WORD_LENGTH])

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)