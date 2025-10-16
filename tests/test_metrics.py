#!/usr/bin/env python3
"""
Tests unitaires pour src/evaluation/metrics.py
Démontre la testabilité des fonctions de calcul de métriques.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from typing import Dict

from src.evaluation.metrics import (
    ModelEvaluator,
    create_data_splits,
    log_training_history,
    MLflowLoggerProtocol,
    PlotBackendProtocol
)


class TestModelEvaluator:
    """Tests pour la classe ModelEvaluator."""

    def test_evaluate_model_basic_metrics(self):
        """Teste le calcul des métriques de base."""
        evaluator = ModelEvaluator()

        # Données de test parfaites
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])

        metrics = evaluator.evaluate_model(y_true, y_pred, model_name="test_model")

        assert metrics['model_name'] == "test_model"
        assert metrics['accuracy'] == 1.0  # 100% exact
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0

    def test_evaluate_model_with_errors(self):
        """Teste le calcul avec des erreurs de prédiction."""
        evaluator = ModelEvaluator()

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1])  # 2 erreurs

        metrics = evaluator.evaluate_model(y_true, y_pred)

        assert metrics['accuracy'] < 1.0  # Moins de 100%
        assert 'f1_score' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics

    def test_evaluate_model_with_probabilities(self):
        """Teste le calcul de l'AUC avec probabilités."""
        evaluator = ModelEvaluator()

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])

        metrics = evaluator.evaluate_model(y_true, y_pred, y_pred_proba=y_pred_proba)

        assert 'auc_score' in metrics
        assert 0 <= metrics['auc_score'] <= 1.0

    def test_evaluate_model_with_training_time(self):
        """Teste l'ajout du temps d'entraînement."""
        evaluator = ModelEvaluator()

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = evaluator.evaluate_model(y_true, y_pred, training_time=120.5)

        assert 'training_time' in metrics
        assert metrics['training_time'] == 120.5

    def test_evaluate_model_confusion_matrix(self):
        """Teste le calcul de la matrice de confusion."""
        evaluator = ModelEvaluator()

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])  # 1 FP, 1 FN

        metrics = evaluator.evaluate_model(y_true, y_pred)

        assert 'confusion_matrix' in metrics
        assert metrics['true_negatives'] == 1
        assert metrics['false_positives'] == 1
        assert metrics['false_negatives'] == 1
        assert metrics['true_positives'] == 1

    def test_evaluate_model_stores_history(self):
        """Teste que les résultats sont stockés dans l'historique."""
        evaluator = ModelEvaluator()

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        assert len(evaluator.results_history) == 0

        evaluator.evaluate_model(y_true, y_pred, model_name="model1")
        assert len(evaluator.results_history) == 1

        evaluator.evaluate_model(y_true, y_pred, model_name="model2")
        assert len(evaluator.results_history) == 2

    def test_evaluate_model_classification_report(self):
        """Teste la génération du rapport de classification."""
        evaluator = ModelEvaluator()

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])

        metrics = evaluator.evaluate_model(y_true, y_pred)

        assert 'classification_report' in metrics
        assert isinstance(metrics['classification_report'], dict)


class TestCompareModels:
    """Tests pour la fonction compare_models."""

    def test_compare_models_basic(self):
        """Teste la comparaison de base de plusieurs modèles."""
        evaluator = ModelEvaluator()

        models_results = {
            'model1': {
                'accuracy': 0.85,
                'precision': 0.84,
                'recall': 0.86,
                'f1_score': 0.85,
                'training_time': 120.0
            },
            'model2': {
                'accuracy': 0.90,
                'precision': 0.89,
                'recall': 0.91,
                'f1_score': 0.90,
                'training_time': 150.0
            }
        }

        df_comparison = evaluator.compare_models(models_results)

        assert len(df_comparison) == 2
        assert 'Model' in df_comparison.columns
        assert 'Accuracy' in df_comparison.columns
        assert 'F1-Score' in df_comparison.columns
        assert 'Training Time (s)' in df_comparison.columns

    def test_compare_models_sorted_by_f1(self):
        """Teste que les modèles sont triés par F1-Score décroissant."""
        evaluator = ModelEvaluator()

        models_results = {
            'model_low': {'accuracy': 0.7, 'precision': 0.7, 'recall': 0.7, 'f1_score': 0.70, 'training_time': 100},
            'model_high': {'accuracy': 0.9, 'precision': 0.9, 'recall': 0.9, 'f1_score': 0.90, 'training_time': 120},
            'model_mid': {'accuracy': 0.8, 'precision': 0.8, 'recall': 0.8, 'f1_score': 0.80, 'training_time': 110}
        }

        df_comparison = evaluator.compare_models(models_results)

        # Vérifier l'ordre décroissant
        assert df_comparison.iloc[0]['Model'] == 'model_high'
        assert df_comparison.iloc[1]['Model'] == 'model_mid'
        assert df_comparison.iloc[2]['Model'] == 'model_low'

    def test_compare_models_with_auc(self):
        """Teste la comparaison avec AUC."""
        evaluator = ModelEvaluator()

        models_results = {
            'model1': {
                'accuracy': 0.85,
                'precision': 0.84,
                'recall': 0.86,
                'f1_score': 0.85,
                'auc_score': 0.88,
                'training_time': 120.0
            }
        }

        df_comparison = evaluator.compare_models(models_results)

        assert 'AUC' in df_comparison.columns
        assert df_comparison.iloc[0]['AUC'] == 0.88


class TestGetBestModel:
    """Tests pour la fonction get_best_model."""

    def test_get_best_model_by_f1_score(self):
        """Teste l'identification du meilleur modèle par F1-Score."""
        evaluator = ModelEvaluator()

        models_results = {
            'model1': {'f1_score': 0.85, 'accuracy': 0.86},
            'model2': {'f1_score': 0.90, 'accuracy': 0.88},
            'model3': {'f1_score': 0.80, 'accuracy': 0.92}
        }

        best_name, best_results = evaluator.get_best_model(models_results)

        assert best_name == 'model2'
        assert best_results['f1_score'] == 0.90

    def test_get_best_model_by_accuracy(self):
        """Teste l'identification du meilleur modèle par Accuracy."""
        evaluator = ModelEvaluator()

        models_results = {
            'model1': {'f1_score': 0.85, 'accuracy': 0.86},
            'model2': {'f1_score': 0.90, 'accuracy': 0.88},
            'model3': {'f1_score': 0.80, 'accuracy': 0.92}
        }

        best_name, best_results = evaluator.get_best_model(models_results, metric='accuracy')

        assert best_name == 'model3'
        assert best_results['accuracy'] == 0.92


class TestGenerateEvaluationReport:
    """Tests pour la fonction generate_evaluation_report."""

    def test_generate_evaluation_report_structure(self):
        """Teste la structure du rapport généré."""
        evaluator = ModelEvaluator()

        models_results = {
            'model1': {
                'accuracy': 0.85,
                'precision': 0.84,
                'recall': 0.86,
                'f1_score': 0.85,
                'training_time': 120.0
            }
        }

        report = evaluator.generate_evaluation_report(models_results)

        assert isinstance(report, str)
        assert "RAPPORT D'ÉVALUATION" in report
        assert "CHOIX DES MÉTRIQUES" in report
        assert "SYNTHÈSE COMPARATIVE" in report
        assert "MEILLEUR MODÈLE" in report
        assert "RECOMMANDATIONS" in report

    def test_generate_evaluation_report_includes_metrics(self):
        """Teste que le rapport inclut les métriques."""
        evaluator = ModelEvaluator()

        models_results = {
            'test_model': {
                'accuracy': 0.92,
                'precision': 0.91,
                'recall': 0.93,
                'f1_score': 0.92,
                'training_time': 100.5
            }
        }

        report = evaluator.generate_evaluation_report(models_results)

        assert "test_model" in report
        assert "0.92" in report or "0.9200" in report  # F1-Score
        assert "100.50" in report  # Training time


class TestCreateDataSplits:
    """Tests pour la fonction create_data_splits."""

    def test_create_data_splits_proportions(self):
        """Teste les proportions des splits."""
        # Créer un DataFrame de test
        df = pd.DataFrame({
            'cleaned_text': [f'text_{i}' for i in range(100)],
            'sentiment': [0 if i < 50 else 1 for i in range(100)]
        })

        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(
            df, test_size=0.2, val_size=0.1, random_state=42
        )

        # Vérifier les proportions (approximativement)
        total = len(df)
        assert len(y_test) == pytest.approx(total * 0.2, abs=2)  # 20% test
        assert len(y_val) == pytest.approx(total * 0.1, abs=2)   # 10% val
        assert len(y_train) == pytest.approx(total * 0.7, abs=2) # 70% train

    def test_create_data_splits_no_overlap(self):
        """Teste qu'il n'y a pas de fuite entre les splits."""
        df = pd.DataFrame({
            'cleaned_text': [f'unique_text_{i}' for i in range(50)],
            'sentiment': [i % 2 for i in range(50)]
        })

        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(df, random_state=42)

        # Convertir en listes pour vérifier l'absence d'overlap
        train_texts = set(X_train.tolist())
        val_texts = set(X_val.tolist())
        test_texts = set(X_test.tolist())

        # Aucun overlap entre les splits
        assert len(train_texts & val_texts) == 0
        assert len(train_texts & test_texts) == 0
        assert len(val_texts & test_texts) == 0

    def test_create_data_splits_stratification(self):
        """Teste la stratification des splits."""
        # Dataset avec répartition 30/70
        df = pd.DataFrame({
            'cleaned_text': [f'text_{i}' for i in range(100)],
            'sentiment': [0 if i < 30 else 1 for i in range(100)]
        })

        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(df, random_state=42)

        # Vérifier que la répartition est maintenue (environ 30/70)
        train_ratio = y_train.value_counts(normalize=True)[1]
        val_ratio = y_val.value_counts(normalize=True)[1]
        test_ratio = y_test.value_counts(normalize=True)[1]

        assert train_ratio == pytest.approx(0.7, abs=0.1)
        assert val_ratio == pytest.approx(0.7, abs=0.1)
        assert test_ratio == pytest.approx(0.7, abs=0.1)


class TestLogTrainingHistoryWithMocks:
    """Tests pour log_training_history avec dépendances mockées."""

    def test_log_training_history_with_mocked_mlflow(self):
        """Teste le logging avec MLflow mocké."""
        mock_mlflow = Mock(spec=MLflowLoggerProtocol)
        mock_plot = Mock(spec=PlotBackendProtocol)

        history = {
            'loss': [0.5, 0.4, 0.3],
            'accuracy': [0.7, 0.8, 0.9],
            'val_loss': [0.6, 0.5, 0.4],
            'val_accuracy': [0.65, 0.75, 0.85]
        }

        log_training_history(
            history,
            model_name="test_model",
            mlflow_logger=mock_mlflow,
            plot_backend=mock_plot
        )

        # Vérifier que log_metric a été appelé
        assert mock_mlflow.log_metric.call_count > 0

        # Vérifier que log_artifact a été appelé (pour le graphique)
        assert mock_mlflow.log_artifact.call_count == 1

    def test_log_training_history_metric_names(self):
        """Teste que les noms de métriques sont correctement formatés."""
        mock_mlflow = Mock(spec=MLflowLoggerProtocol)
        mock_plot = Mock(spec=PlotBackendProtocol)

        history = {
            'loss': [0.5],
            'val_loss': [0.6]
        }

        log_training_history(
            history,
            model_name="test",
            mlflow_logger=mock_mlflow,
            plot_backend=mock_plot
        )

        # Vérifier les appels avec les bons noms
        calls = mock_mlflow.log_metric.call_args_list

        # Extraire les noms de métriques
        metric_names = [call[0][0] for call in calls]

        assert 'train_loss' in metric_names
        assert 'val_loss' in metric_names

    def test_log_training_history_plot_saved(self):
        """Teste que le graphique est sauvegardé."""
        mock_mlflow = Mock(spec=MLflowLoggerProtocol)
        mock_plot = Mock(spec=PlotBackendProtocol)

        history = {
            'loss': [0.5, 0.4],
            'val_loss': [0.6, 0.5]
        }

        log_training_history(
            history,
            mlflow_logger=mock_mlflow,
            plot_backend=mock_plot
        )

        # Vérifier que savefig a été appelé
        assert mock_plot.savefig.call_count == 1

        # Vérifier que close a été appelé
        assert mock_plot.close.call_count == 1