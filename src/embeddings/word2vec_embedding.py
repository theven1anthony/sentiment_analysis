"""
Module d'embedding Word2Vec pour l'analyse de sentiment.
Utilise Gensim pour créer des embeddings de mots entraînés sur le corpus.
"""

import numpy as np
from gensim.models import Word2Vec
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class Word2VecEmbedding:
    """Génère des embeddings Word2Vec pour les textes."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
        sg: int = 0,  # 0 = CBOW, 1 = Skip-gram
    ):
        """
        Initialise le modèle Word2Vec.

        Args:
            vector_size: Dimension des vecteurs d'embeddings
            window: Taille de la fenêtre contextuelle
            min_count: Fréquence minimale des mots
            workers: Nombre de threads pour l'entraînement
            epochs: Nombre d'itérations d'entraînement
            sg: Architecture (0=CBOW, 1=Skip-gram)
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        self.model: Optional[Word2Vec] = None

    def fit(self, texts: List[str]) -> "Word2VecEmbedding":
        """
        Entraîne le modèle Word2Vec sur le corpus.

        Args:
            texts: Liste de textes prétraités (tokens séparés par espaces)

        Returns:
            self pour chaînage
        """
        logger.info(f"Entraînement Word2Vec sur {len(texts)} documents...")

        sentences = [text.split() for text in texts]

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=self.sg,
        )

        logger.info(f"Word2Vec terminé - Vocabulaire: {len(self.model.wv)} mots")

        return self

    def transform(self, texts: List[str], max_len: int = 50, average: bool = False) -> np.ndarray:
        """
        Transforme les textes en embeddings.

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
                if word in self.model.wv:
                    sequences[i, j] = self.model.wv[word]

        return sequences

    def _get_averaged_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Transforme les textes en embeddings moyennés pour Dense.

        Args:
            texts: Liste de textes à transformer

        Returns:
            Matrice numpy (n_samples, vector_size)
        """
        n_samples = len(texts)
        embeddings = np.zeros((n_samples, self.vector_size), dtype=np.float32)

        for i, text in enumerate(texts):
            words = [w for w in text.split() if w in self.model.wv]
            if words:
                embeddings[i] = np.mean([self.model.wv[w] for w in words], axis=0)

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

    def transform_batched(
        self, texts: List[str], max_len: int = 50, average: bool = False, batch_size: int = 5000
    ) -> np.ndarray:
        """
        Transforme les textes en embeddings par batches (économie mémoire).

        Args:
            texts: Liste de textes à transformer
            max_len: Longueur maximale des séquences
            average: Si True, embeddings moyennés, sinon séquences
            batch_size: Taille des batches (défaut: 5000)

        Returns:
            Matrice numpy des embeddings
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant transformation")

        logger.info(f"Transformation par batches ({batch_size} textes/batch)...")

        n_samples = len(texts)
        n_batches = (n_samples + batch_size - 1) // batch_size

        batches = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_texts = texts[start_idx:end_idx]

            logger.info(f"  Batch {i+1}/{n_batches} ({start_idx}:{end_idx})...")
            batch_emb = self.transform(batch_texts, max_len=max_len, average=average)
            batches.append(batch_emb)

        logger.info("Concaténation des batches...")
        return np.vstack(batches) if average else np.concatenate(batches, axis=0)

    def get_params(self) -> dict:
        """Retourne les paramètres du modèle pour logging MLflow."""
        return {
            "embedding_type": "word2vec",
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "epochs": self.epochs,
            "architecture": "skip-gram" if self.sg == 1 else "cbow",
        }
