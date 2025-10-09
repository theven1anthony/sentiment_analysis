"""
Modèle MLflow pyfunc custom pour Word2Vec + TensorFlow/Keras.
Encapsule le pipeline complet : preprocessing + embedding + prédiction.
"""

import mlflow
import numpy as np
import pandas as pd
from typing import Union, List


class Word2VecSentimentModel(mlflow.pyfunc.PythonModel):
    """
    Modèle MLflow pyfunc custom qui encapsule :
    - TextCleaner (preprocessing)
    - Word2VecEmbedding (transformation en embeddings)
    - Modèle TensorFlow/Keras (prédiction)
    """

    def load_context(self, context):
        """
        Charge le modèle et ses dépendances depuis le contexte MLflow.

        Args:
            context: MLflow context contenant les artifacts
        """
        import pickle
        import tensorflow as tf
        import sys
        import os

        # Ajouter src au path pour imports (MLflow package le code source)
        model_dir = os.path.dirname(context.artifacts["keras_model"])
        parent_dir = os.path.dirname(model_dir)
        if os.path.exists(os.path.join(parent_dir, "preprocessing")):
            sys.path.insert(0, parent_dir)

        from preprocessing.text_cleaner import TextCleaner

        # 1. Charger le modèle TensorFlow
        self.keras_model = tf.keras.models.load_model(
            context.artifacts["keras_model"]
        )

        # 2. Charger l'embedding Word2Vec
        with open(context.artifacts["word2vec_embedding"], 'rb') as f:
            self.word2vec_embedding = pickle.load(f)

        # 3. Initialiser le TextCleaner
        self.text_cleaner = TextCleaner()

        # 4. Récupérer la technique depuis le fichier artifact
        technique_path = context.artifacts.get("technique")
        if technique_path and os.path.exists(technique_path):
            with open(technique_path, 'r') as f:
                self.technique = f.read().strip()
        else:
            self.technique = "stemming"  # Valeur par défaut

        self.max_len = self.keras_model.input_shape[1]  # Récupérer max_len depuis le modèle

    def predict(self, context, model_input):
        """
        Fait une prédiction sur les textes d'entrée.

        Args:
            context: MLflow context (non utilisé ici)
            model_input: pandas DataFrame avec colonne 'text' OU liste de strings

        Returns:
            numpy array avec prédictions (0 ou 1) et confidences
        """
        # Gérer différents formats d'entrée
        if isinstance(model_input, pd.DataFrame):
            if 'text' not in model_input.columns:
                raise ValueError("DataFrame doit contenir une colonne 'text'")
            texts = model_input['text'].tolist()
        elif isinstance(model_input, list):
            texts = model_input
        elif isinstance(model_input, str):
            texts = [model_input]
        else:
            raise ValueError(f"Format d'entrée non supporté: {type(model_input)}")

        # Pipeline de prédiction
        # 1. Preprocessing
        cleaned_texts = self.text_cleaner.preprocess_with_techniques(
            texts,
            technique=self.technique
        )

        # 2. Transformation en embeddings Word2Vec
        embeddings = self.word2vec_embedding.transform(
            cleaned_texts,
            max_len=self.max_len,
            average=False  # Séquences pour LSTM
        )

        # 3. Prédiction
        predictions_proba = self.keras_model.predict(embeddings, verbose=0)
        predictions = (predictions_proba > 0.5).astype(int).flatten()
        confidences = np.where(
            predictions == 1,
            predictions_proba.flatten(),
            1 - predictions_proba.flatten()
        )

        # Retourner DataFrame avec prédictions et confidences
        return pd.DataFrame({
            'sentiment': predictions,
            'confidence': confidences
        })