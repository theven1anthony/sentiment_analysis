import re
import string
from typing import List
import pandas as pd
import numpy as np


class TextCleaner:
    """Nettoyage et preprocessing des tweets pour l'analyse de sentiment."""

    def __init__(self, remove_urls: bool = True, remove_mentions: bool = True,
                 remove_hashtags: bool = False, lowercase: bool = True):
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.lowercase = lowercase

    def clean_tweet(self, text: str) -> str:
        """Nettoie un tweet individuel."""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Suppression des URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Suppression des mentions (@username)
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        else:
            text = text.replace("@", "")

        # Suppression des hashtags (#tag)
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        else:
            text = text.replace("#", "")

        # Suppression des caractères non-ASCII
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # Suppression de la ponctuation excessive
        text = re.sub(r'[' + string.punctuation + ']+', ' ', text)

        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)

        # Conversion en minuscules
        if self.lowercase:
            text = text.lower()

        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Nettoie une liste de tweets."""
        return [self.clean_tweet(text) for text in texts]


def load_sentiment140_data(file_path: str, sample_size: int = None) -> pd.DataFrame:
    """Charge le dataset Sentiment140."""
    columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

    if sample_size:
        # Échantillonnage aléatoire pour les tests
        df = pd.read_csv(file_path, names=columns, encoding='latin1',
                        skiprows=lambda i: i > 0 and np.random.random() > sample_size/1600000)
    else:
        df = pd.read_csv(file_path, names=columns, encoding='latin1')

    # Conversion des labels (0->0, 4->1)
    df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})

    return df[['text', 'sentiment']]


def preprocess_dataset(df: pd.DataFrame, cleaner: TextCleaner = None) -> pd.DataFrame:
    """Preprocessing complet du dataset."""
    if cleaner is None:
        cleaner = TextCleaner()

    # Nettoyage des textes
    df = df.copy()
    df['cleaned_text'] = df['text'].apply(cleaner.clean_tweet)

    # Suppression des tweets vides après nettoyage
    df = df[df['cleaned_text'].str.len() > 0]

    return df.reset_index(drop=True)