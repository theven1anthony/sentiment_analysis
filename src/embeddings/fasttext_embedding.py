"""
Module d'embedding FastText pour l'analyse de sentiment.
Utilise Gensim avec FastText pour gérer les mots hors vocabulaire.
"""

import numpy as np
from gensim.models import FastText
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FastTextEmbedding:
    """Génère des embeddings FastText pour les textes."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
        sg: int = 0,  # 0 = CBOW, 1 = Skip-gram
        min_n: int = 3,  # Taille min des n-grammes de caractères
        max_n: int = 6   # Taille max des n-grammes de caractères
    ):
        """
        Initialise le modèle FastText.

        Args:
            vector_size: Dimension des vecteurs d'embeddings
            window: Taille de la fenêtre contextuelle
            min_count: Fréquence minimale des mots
            workers: Nombre de threads pour l'entraînement
            epochs: Nombre d'itérations d'entraînement
            sg: Architecture (0=CBOW, 1=Skip-gram)
            min_n: Taille minimale des n-grammes de caractères
            max_n: Taille maximale des n-grammes de caractères
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        self.min_n = min_n
        self.max_n = max_n
        self.model: Optional[FastText] = None

    def fit(self, texts: List[str]) -> 'FastTextEmbedding':
        """
        Entraîne le modèle FastText sur le corpus.

        Args:
            texts: Liste de textes prétraités (tokens séparés par espaces)

        Returns:
            self pour chaînage
        """
        logger.info(f"Entraînement FastText sur {len(texts)} documents...")

        sentences = [text.split() for text in texts]

        self.model = FastText(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=self.sg,
            min_n=self.min_n,
            max_n=self.max_n
        )

        logger.info(f"FastText terminé - Vocabulaire: {len(self.model.wv)} mots")

        return self

    def transform(self, texts: List[str], max_len: int = 50, average: bool = False) -> np.ndarray:
        """
        Transforme les textes en embeddings.
        FastText gère automatiquement les mots hors vocabulaire via n-grammes de caractères.

        Args:
            texts: Liste de textes à transformer
            max_len: Longueur maximale des séquences (ignoré si average=True)
            average: Si True, retourne embeddings moyennés (n_samples, vector_size) pour Dense
                     Si False, retourne séquences (n_samples, max_len, vector_size) pour LSTM

        Returns:
            Matrice numpy de shape (n_samples, vector_size) si average=True
            ou (n_samples, max_len, vector_size) si average=False
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant transformation")

        if average:
            return self._get_averaged_embeddings(texts)
        else:
            return self._get_sequences(texts, max_len)

    def _get_sequences(self, texts: List[str], max_len: int) -> np.ndarray:
        """
        Transforme les textes en séquences de vecteurs pour LSTM.
        FastText gère automatiquement les OOV via n-grammes.

        Args:
            texts: Liste de textes à transformer
            max_len: Longueur maximale des séquences

        Returns:
            Matrice numpy (n_samples, max_len, vector_size)
        """
        n_samples = len(texts)
        sequences = np.zeros((n_samples, max_len, self.vector_size), dtype=np.float32)

        for i, text in enumerate(texts):
            words = text.split()[:max_len]
            for j, word in enumerate(words):
                sequences[i, j] = self.model.wv[word]

        return sequences

    def _get_averaged_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Transforme les textes en embeddings moyennés pour Dense.
        FastText gère automatiquement les OOV via n-grammes.

        Args:
            texts: Liste de textes à transformer

        Returns:
            Matrice numpy (n_samples, vector_size)
        """
        n_samples = len(texts)
        embeddings = np.zeros((n_samples, self.vector_size), dtype=np.float32)

        for i, text in enumerate(texts):
            words = text.split()
            if words:
                embeddings[i] = np.mean([self.model.wv[word] for word in words], axis=0)

        return embeddings

    def fit_transform(self, texts: List[str], max_len: int = 50, average: bool = False) -> np.ndarray:
        """
        Entraîne le modèle et transforme les textes en une seule étape.

        Args:
            texts: Liste de textes
            max_len: Longueur maximale des séquences (ignoré si average=True)
            average: Si True, retourne embeddings moyennés pour Dense, sinon séquences pour LSTM

        Returns:
            Matrice numpy (n_samples, vector_size) si average=True
            ou (n_samples, max_len, vector_size) si average=False
        """
        self.fit(texts)
        return self.transform(texts, max_len=max_len, average=average)

    def get_params(self) -> dict:
        """Retourne les paramètres du modèle pour logging MLflow."""
        return {
            'embedding_type': 'fasttext',
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'epochs': self.epochs,
            'architecture': 'skip-gram' if self.sg == 1 else 'cbow',
            'min_n': self.min_n,
            'max_n': self.max_n
        }