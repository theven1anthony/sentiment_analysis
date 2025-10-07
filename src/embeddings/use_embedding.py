"""
Module d'embedding Universal Sentence Encoder (USE) pour l'analyse de sentiment.
Utilise TensorFlow Hub pour charger le modèle USE pré-entraîné.
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class USEEmbedding:
    """Génère des embeddings Universal Sentence Encoder pour les textes."""

    def __init__(
        self,
        model_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4"
    ):
        """
        Initialise le modèle Universal Sentence Encoder.

        Args:
            model_url: URL du modèle USE sur TensorFlow Hub
        """
        self.model_url = model_url
        self.model: Optional[tf.keras.Model] = None

    def fit(self, texts: List[str]) -> 'USEEmbedding':
        """
        Charge le modèle USE pré-entraîné.
        Pas d'entraînement nécessaire pour USE (modèle pré-entraîné).

        Args:
            texts: Liste de textes (non utilisé, mais requis pour API cohérente)

        Returns:
            self pour chaînage
        """
        logger.info("Chargement du modèle Universal Sentence Encoder...")

        self.model = hub.load(self.model_url)

        logger.info("USE chargé avec succès")
        return self

    def transform(self, texts: List[str], batch_size: int = 1000) -> np.ndarray:
        """
        Transforme les textes en embeddings USE par batches.

        Args:
            texts: Liste de textes à transformer
            batch_size: Taille des batches pour l'encodage

        Returns:
            Matrice numpy (n_samples, 512) - USE produit des vecteurs de 512 dimensions
        """
        if self.model is None:
            raise ValueError("Le modèle doit être chargé avant transformation")

        logger.info(f"Transformation de {len(texts)} textes avec USE (batches de {batch_size})...")

        # Encoder par batches pour économiser la mémoire
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model(batch_texts)
            all_embeddings.append(np.array(batch_embeddings))

        return np.vstack(all_embeddings)

    def fit_transform(self, texts: List[str], batch_size: int = 1000) -> np.ndarray:
        """
        Charge le modèle et transforme les textes en une seule étape.

        Args:
            texts: Liste de textes
            batch_size: Taille des batches pour l'encodage

        Returns:
            Matrice numpy (n_samples, 512)
        """
        self.fit(texts)
        return self.transform(texts, batch_size=batch_size)

    def get_params(self) -> dict:
        """Retourne les paramètres du modèle pour logging MLflow."""
        return {
            'embedding_type': 'use',
            'model_url': self.model_url,
            'vector_size': 512  # USE produit toujours des vecteurs de 512 dimensions
        }