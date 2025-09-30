"""
Modèle simple de classification de sentiment avec scikit-learn.
Implémente un modèle de référence (baseline) pour comparer les performances.
Répond aux critères d'évaluation CE3: modèle de référence pour comparaison.
"""

import time
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


class SimpleSentimentModel:
    """
    Modèle simple de classification de sentiment utilisant Logistic Regression + TF-IDF.
    Sert de modèle de référence (baseline) pour évaluer les modèles plus complexes.
    """

    def __init__(self):
        """Initialise le modèle simple avec Logistic Regression + TF-IDF."""
        self.pipeline = None
        self.training_time = None
        self.feature_names = None

    def _create_vectorizer(self):
        """
        Crée le vectoriseur TF-IDF optimisé pour texte déjà préprocessé.
        Le texte est déjà nettoyé, en minuscules, et les négations gérées intelligemment.
        """
        return TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None,  # Pas de stop words - déjà géré dans preprocessing
            lowercase=False,  # Déjà en minuscules après preprocessing
            min_df=2,
            max_df=0.95,
            token_pattern=r'\b\w+\b'  # Pattern simple pour tokens déjà nettoyés
        )

    def _create_classifier(self):
        """Crée le classifieur Logistic Regression."""
        return LogisticRegression(random_state=42, max_iter=1000)

    def create_pipeline(self):
        """Crée le pipeline de traitement."""
        vectorizer = self._create_vectorizer()
        classifier = self._create_classifier()

        self.pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])

        return self.pipeline

    def train(self, X_train: pd.Series, y_train: pd.Series,
              X_val: pd.Series = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Entraîne le modèle.

        Args:
            X_train: Textes d'entraînement
            y_train: Labels d'entraînement
            X_val: Textes de validation (optionnel)
            y_val: Labels de validation (optionnel)

        Returns:
            Dictionnaire avec les métriques d'entraînement
        """
        # Créer le pipeline si nécessaire
        if self.pipeline is None:
            self.create_pipeline()

        # Mesurer le temps d'entraînement
        start_time = time.time()

        # Entraînement
        self.pipeline.fit(X_train, y_train)

        self.training_time = time.time() - start_time

        # Récupérer les noms des features
        self.feature_names = self.pipeline.named_steps['vectorizer'].get_feature_names_out()

        # Évaluation sur les données d'entraînement
        train_pred = self.pipeline.predict(X_train)
        train_metrics = self._compute_metrics(y_train, train_pred)

        results = {
            'model_type': 'logistic',
            'vectorizer_type': 'tfidf',
            'training_time': self.training_time,
            'n_features': len(self.feature_names),
            'train_metrics': train_metrics
        }

        # Évaluation sur les données de validation si disponibles
        if X_val is not None and y_val is not None:
            val_pred = self.pipeline.predict(X_val)
            val_metrics = self._compute_metrics(y_val, val_pred)
            results['val_metrics'] = val_metrics

        return results

    def predict(self, X: pd.Series) -> np.ndarray:
        """
        Fait des prédictions.

        Args:
            X: Textes à prédire

        Returns:
            Prédictions (0 ou 1)
        """
        if self.pipeline is None:
            raise ValueError("Le modèle n'est pas encore entraîné")

        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """
        Calcule les probabilités de prédiction.

        Args:
            X: Textes à prédire

        Returns:
            Probabilités pour chaque classe
        """
        if self.pipeline is None:
            raise ValueError("Le modèle n'est pas encore entraîné")

        return self.pipeline.predict_proba(X)

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, float]:
        """
        Évalue le modèle sur les données de test.

        Args:
            X_test: Textes de test
            y_test: Labels de test

        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        predictions = self.predict(X_test)
        return self._compute_metrics(y_test, predictions)

    def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcule les métriques de classification.

        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions

        Returns:
            Dictionnaire avec les métriques
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

    def save_model(self, filepath: str):
        """
        Sauvegarde le modèle.

        Args:
            filepath: Chemin pour sauvegarder le modèle
        """
        if self.pipeline is None:
            raise ValueError("Le modèle n'est pas encore entraîné")

        model_data = {
            'pipeline': self.pipeline,
            'model_type': 'logistic',
            'vectorizer_type': 'tfidf',
            'training_time': self.training_time,
            'feature_names': self.feature_names
        }

        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str):
        """
        Charge un modèle sauvegardé.

        Args:
            filepath: Chemin du modèle à charger
        """
        model_data = joblib.load(filepath)

        self.pipeline = model_data['pipeline']
        self.training_time = model_data.get('training_time')
        self.feature_names = model_data.get('feature_names')

    def hyperparameter_tuning(self, X_train: pd.Series, y_train: pd.Series,
                            cv: int = 5) -> Dict[str, Any]:
        """
        Optimisation des hyperparamètres.
        Répond aux critères d'évaluation CE5: optimisation d'au moins un hyperparamètre.

        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            cv: Nombre de folds pour la validation croisée

        Returns:
            Résultats de l'optimisation
        """
        # Créer le pipeline
        self.create_pipeline()

        # Paramètres à optimiser pour Logistic Regression + TF-IDF
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l2'],
            'vectorizer__max_features': [5000, 10000],
            'vectorizer__ngram_range': [(1, 1), (1, 2)]
        }

        # Grid search
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time

        # Mettre à jour le pipeline avec les meilleurs paramètres
        self.pipeline = grid_search.best_estimator_

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'tuning_time': tuning_time,
            'cv_results': grid_search.cv_results_
        }


def train_simple_model(X_train: pd.Series, y_train: pd.Series,
                      X_val: pd.Series, y_val: pd.Series,
                      X_test: pd.Series, y_test: pd.Series) -> Tuple[SimpleSentimentModel, Dict[str, float], Dict[str, str]]:
    """
    Entraîne le modèle simple de référence (Logistic Regression + TF-IDF).
    Format standard MLflow: retourne (model, metrics, artifacts).

    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        X_test, y_test: Données de test

    Returns:
        Tuple (model, metrics, artifacts) pour MLflow
    """
    print("Entraînement du modèle simple (Logistic Regression + TF-IDF)...")

    # Initialiser le modèle
    model = SimpleSentimentModel()

    # Entraînement
    train_results = model.train(X_train, y_train, X_val, y_val)

    # Évaluation sur les données de test
    test_metrics = model.evaluate(X_test, y_test)

    print(f"  - Accuracy test: {test_metrics['accuracy']:.4f}")
    print(f"  - F1-score test: {test_metrics['f1_score']:.4f}")
    print(f"  - Temps d'entraînement: {train_results['training_time']:.2f}s")

    # Format pour MLflow
    metrics = {
        'accuracy': test_metrics['accuracy'],
        'f1_score': test_metrics['f1_score'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'training_time': train_results['training_time']
    }

    artifacts = {}  # Pas d'artifacts spéciaux pour le modèle simple

    return model.pipeline, metrics, artifacts