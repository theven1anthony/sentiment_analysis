#!/usr/bin/env python3
"""
Tests unitaires pour src/utils/train_utils.py
Démontre la testabilité du code refactoré avec injection de dépendances.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from typing import List

from src.utils.train_utils import (
    select_techniques_to_run,
    filter_metrics_for_mlflow,
    create_comparison_dataframe,
    find_best_technique,
    save_report,
    load_and_preprocess_data,
    create_train_val_test_splits,
    print_training_header,
    log_common_mlflow_info,
    OutputLogger,
    MLflowLoggerProtocol
)


class TestSelectTechniquesToRun:
    """Tests pour select_techniques_to_run (fonction pure)."""

    def test_both_techniques(self):
        result = select_techniques_to_run('both')
        assert result == [('stemming', 'text_stemmed'), ('lemmatization', 'text_lemmatized')]

    def test_stemming_only(self):
        result = select_techniques_to_run('stemming')
        assert result == [('stemming', 'text_stemmed')]

    def test_lemmatization_only(self):
        result = select_techniques_to_run('lemmatization')
        assert result == [('lemmatization', 'text_lemmatized')]


class TestFilterMetricsForMlflow:
    """Tests pour filter_metrics_for_mlflow (fonction pure)."""

    def test_filters_unwanted_metrics(self):
        metrics = {
            'accuracy': 0.95,
            'f1_score': 0.94,
            'confusion_matrix': [[100, 10], [5, 85]],
            'classification_report': 'some text',
            'model_name': 'test_model'
        }

        result = filter_metrics_for_mlflow(metrics)

        assert 'accuracy' in result
        assert 'f1_score' in result
        assert 'confusion_matrix' not in result
        assert 'classification_report' not in result
        assert 'model_name' not in result


class TestCreateComparisonDataframe:
    """Tests pour create_comparison_dataframe (fonction pure)."""

    def test_creates_comparison_dataframe(self):
        results = {
            'stemming': {
                'metrics': {
                    'accuracy': 0.85,
                    'f1_score': 0.84,
                    'auc_score': 0.86,
                    'training_time': 120.5
                }
            },
            'lemmatization': {
                'metrics': {
                    'accuracy': 0.87,
                    'f1_score': 0.86,
                    'auc_score': 0.88,
                    'training_time': 135.2
                }
            }
        }

        df = create_comparison_dataframe(results)

        assert len(df) == 2
        assert 'Technique' in df.columns
        assert 'Accuracy' in df.columns
        assert 'F1-Score' in df.columns
        assert 'AUC' in df.columns
        assert 'Time (s)' in df.columns


class TestFindBestTechnique:
    """Tests pour find_best_technique (fonction pure)."""

    def test_finds_best_by_f1_score(self):
        results = {
            'stemming': {'metrics': {'f1_score': 0.84, 'accuracy': 0.85}},
            'lemmatization': {'metrics': {'f1_score': 0.86, 'accuracy': 0.83}}
        }

        best_tech, best_metrics = find_best_technique(results, metric='f1_score')

        assert best_tech == 'lemmatization'
        assert best_metrics['f1_score'] == 0.86

    def test_finds_best_by_accuracy(self):
        results = {
            'stemming': {'metrics': {'f1_score': 0.84, 'accuracy': 0.85}},
            'lemmatization': {'metrics': {'f1_score': 0.86, 'accuracy': 0.83}}
        }

        best_tech, best_metrics = find_best_technique(results, metric='accuracy')

        assert best_tech == 'stemming'
        assert best_metrics['accuracy'] == 0.85


class TestSaveReportWithMockTimestamp:
    """Tests pour save_report avec timestamp injectable."""

    def test_save_report_with_custom_timestamp(self, tmp_path):
        report_content = "Test report content"
        report_name = "test_report"

        # Fonction de timestamp mockée
        def mock_timestamp_fn():
            return "20250101_120000"

        # Utiliser tmp_path comme répertoire de rapports
        reports_dir = str(tmp_path / "reports")

        result_path = save_report(
            report=report_content,
            report_name=report_name,
            reports_dir=reports_dir,
            timestamp_fn=mock_timestamp_fn
        )

        # Vérifier que le fichier a été créé avec le bon nom
        assert result_path == f"{reports_dir}/{report_name}_20250101_120000.txt"

        # Vérifier le contenu
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert content == report_content


class TestLoadAndPreprocessDataWithMocks:
    """Tests pour load_and_preprocess_data avec dépendances mockées."""

    def test_load_and_preprocess_with_mocked_dependencies(self):
        # Mock du logger
        mock_logger = Mock(spec=OutputLogger)

        # Mock du TextCleaner
        mock_cleaner = Mock()
        mock_cleaner.preprocess_with_techniques.return_value = ['text1', 'text2', 'text3']

        # Mock du data loader
        mock_df = pd.DataFrame({
            'text': ['tweet1', 'tweet2', 'tweet3'],
            'sentiment': [0, 1, 0]
        })
        mock_data_loader = Mock(return_value=mock_df)

        # Appeler la fonction avec les mocks
        result_df, info = load_and_preprocess_data(
            data_path="/fake/path.csv",
            sample_size=100,
            cleaner=mock_cleaner,
            data_loader=mock_data_loader,
            logger=mock_logger
        )

        # Vérifier que le data loader a été appelé
        mock_data_loader.assert_called_once_with("/fake/path.csv", sample_size=100)

        # Vérifier que le logger a été utilisé
        assert mock_logger.info.call_count > 0

        # Vérifier que le cleaner a été utilisé
        assert mock_cleaner.preprocess_with_techniques.call_count == 2  # stemming et lemmatization


class TestCreateTrainValTestSplitsWithMockLogger:
    """Tests pour create_train_val_test_splits avec logger mocké."""

    def test_splits_with_mocked_logger(self):
        # Mock du logger
        mock_logger = Mock(spec=OutputLogger)

        # Créer un DataFrame de test avec suffisamment de données pour stratification
        df = pd.DataFrame({
            'text': ['text' + str(i) for i in range(20)],
            'sentiment': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                         0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

        # Appeler la fonction
        splits = create_train_val_test_splits(
            df=df,
            test_size=0.3,
            val_ratio=0.5,
            random_state=42,
            logger=mock_logger
        )

        # Vérifier la structure des splits
        assert 'train_idx' in splits
        assert 'val_idx' in splits
        assert 'test_idx' in splits

        # Vérifier que le logger a été appelé
        assert mock_logger.info.call_count > 0


class TestPrintFunctionsWithMockLogger:
    """Tests pour les fonctions print avec logger mocké."""

    def test_print_training_header_with_mock_logger(self):
        mock_logger = Mock(spec=OutputLogger)

        print_training_header(
            model_name="TEST_MODEL",
            technique="stemming",
            description="Test description",
            additional_info={'param1': 'value1'},
            logger=mock_logger
        )

        # Vérifier que le logger a été appelé plusieurs fois
        assert mock_logger.info.call_count >= 5


class TestMLflowLoggerWithMock:
    """Tests pour log_common_mlflow_info avec MLflow mocké."""

    def test_log_common_mlflow_info_with_mock(self):
        # Mock du MLflow logger
        mock_mlflow = Mock(spec=MLflowLoggerProtocol)

        log_common_mlflow_info(
            description="Test experiment",
            model_type="test_neural",
            additional_tags={'tag1': 'value1', 'tag2': 'value2'},
            mlflow_logger=mock_mlflow
        )

        # Vérifier que set_tag a été appelé
        assert mock_mlflow.set_tag.call_count >= 3  # description + model_type + 2 tags

        # Vérifier les appels spécifiques
        mock_mlflow.set_tag.assert_any_call("description", "Test experiment")
        mock_mlflow.set_tag.assert_any_call("model_type", "test_neural")
        mock_mlflow.set_tag.assert_any_call("tag1", "value1")
        mock_mlflow.set_tag.assert_any_call("tag2", "value2")
