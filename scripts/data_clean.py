import nltk
nltk.download("stopwords")

from calendar import c
from string import punctuation
from typing import List
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')


def clean(sentence: str) -> List[str]:
    """Cleans up sentences by removing stop words and punctuation"""
    words = sentence.translate(str.maketrans('', '', punctuation)).split(' ')
    return [w.lower() for w in words if w.lower() not in STOPWORDS]
