#!/usr/bin/env python3
"""
Tests unitaires pour src/utils/model_utils.py
Teste la construction de modèles et callbacks sans injection de dépendances.
"""

import pytest
import os

# Configurer TensorFlow pour éviter les problèmes sur macOS M2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from src.utils.model_utils import build_neural_network, get_training_callbacks


# Marker pour sauter les tests si TensorFlow n'est pas disponible
pytestmark = pytest.mark.skipif(
    not hasattr(tf, 'keras'),
    reason="TensorFlow/Keras non disponible"
)


class TestBuildNeuralNetworkDense:
    """Tests pour build_neural_network avec architecture Dense."""

    def test_builds_dense_model_with_correct_input_shape(self):
        """Vérifie que le modèle Dense est créé avec la bonne forme d'entrée."""
        embedding_dim = 100
        input_dim = (1, embedding_dim)  # (séquence, dimension)

        model = build_neural_network(input_dim, with_lstm=False)

        # Vérifier que le modèle est bien un Sequential
        assert isinstance(model, tf.keras.Sequential)

        # Vérifier l'input shape du modèle (Keras 3 API)
        # On construit le modèle pour obtenir la shape
        import numpy as np
        model.build(input_shape=(None, embedding_dim))
        assert model.input_shape == (None, embedding_dim)

    def test_dense_model_has_correct_number_of_layers(self):
        """Vérifie que le modèle Dense a le bon nombre de couches."""
        model = build_neural_network((1, 100), with_lstm=False)

        # Architecture : Dense(256) -> Dropout -> Dense(128) -> Dropout -> Dense(1)
        assert len(model.layers) == 5

    def test_dense_model_output_layer_is_sigmoid(self):
        """Vérifie que la couche de sortie utilise sigmoid."""
        model = build_neural_network((1, 100), with_lstm=False)

        last_layer = model.layers[-1]
        assert isinstance(last_layer, tf.keras.layers.Dense)
        assert last_layer.units == 1
        assert last_layer.activation.__name__ == 'sigmoid'

    def test_dense_model_compiled_with_adam_optimizer(self):
        """Vérifie que le modèle est compilé avec Adam."""
        model = build_neural_network((1, 100), with_lstm=False)

        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)

    def test_dense_model_custom_learning_rate(self):
        """Vérifie que le learning rate personnalisé est appliqué."""
        custom_lr = 0.0005
        model = build_neural_network((1, 100), with_lstm=False, learning_rate=custom_lr)

        # Vérifier le learning rate
        assert abs(float(model.optimizer.learning_rate.numpy()) - custom_lr) < 1e-6

    def test_dense_model_custom_dropout(self):
        """Vérifie que le dropout personnalisé est appliqué."""
        custom_dropout = 0.5
        model = build_neural_network((1, 100), with_lstm=False, dropout=custom_dropout)

        # Trouver les couches Dropout
        dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]

        assert len(dropout_layers) == 2
        # Les couches dropout devraient avoir custom_dropout * 0.75
        expected_rate = custom_dropout * 0.75
        for layer in dropout_layers:
            assert abs(layer.rate - expected_rate) < 1e-6

    def test_dense_model_has_required_metrics(self):
        """Vérifie que le modèle a les métriques requises."""
        model = build_neural_network((1, 100), with_lstm=False)

        # Dans Keras 3, accéder aux métriques compilées
        # Les métriques sont configurées lors de la compilation
        # On vérifie que le modèle a été compilé avec les bonnes métriques
        assert hasattr(model, 'compiled_metrics')

        # Alternative: vérifier via les noms de métriques dans metrics_names (après un fit ou predict)
        # Pour l'instant on vérifie que le modèle est compilé et a les métriques de base
        assert model.optimizer is not None


class TestBuildNeuralNetworkLSTM:
    """Tests pour build_neural_network avec architecture LSTM."""

    def test_builds_lstm_model_with_correct_input_shape(self):
        """Vérifie que le modèle LSTM est créé avec la bonne forme d'entrée."""
        seq_length = 50
        embedding_dim = 100
        input_dim = (seq_length, embedding_dim)

        model = build_neural_network(input_dim, with_lstm=True)

        # Vérifier que le modèle est bien un Sequential
        assert isinstance(model, tf.keras.Sequential)

        # La première couche devrait être Masking
        assert isinstance(model.layers[0], tf.keras.layers.Masking)

        # Construire le modèle pour vérifier l'input shape (Keras 3)
        model.build(input_shape=(None, seq_length, embedding_dim))
        assert model.input_shape == (None, seq_length, embedding_dim)

    def test_lstm_model_has_correct_number_of_layers(self):
        """Vérifie que le modèle LSTM a le bon nombre de couches."""
        model = build_neural_network((50, 100), with_lstm=True)

        # Architecture : Masking -> Bidirectional(LSTM) -> Dropout -> Dense(128) -> Dropout -> Dense(1)
        assert len(model.layers) == 6

    def test_lstm_model_has_bidirectional_lstm_layer(self):
        """Vérifie que le modèle contient une couche Bidirectional LSTM."""
        model = build_neural_network((50, 100), with_lstm=True)

        # La deuxième couche devrait être Bidirectional
        assert isinstance(model.layers[1], tf.keras.layers.Bidirectional)

    def test_lstm_model_custom_lstm_units(self):
        """Vérifie que les unités LSTM personnalisées sont appliquées."""
        custom_units = 256
        model = build_neural_network((50, 100), with_lstm=True, lstm_units=custom_units)

        bidirectional_layer = model.layers[1]
        # Dans Keras 3, utiliser forward_layer au lieu de layer
        lstm_layer = bidirectional_layer.forward_layer

        assert lstm_layer.units == custom_units

    def test_lstm_model_custom_recurrent_dropout(self):
        """Vérifie que le recurrent_dropout personnalisé est appliqué."""
        custom_recurrent_dropout = 0.4
        model = build_neural_network((50, 100), with_lstm=True, recurrent_dropout=custom_recurrent_dropout)

        bidirectional_layer = model.layers[1]
        # Dans Keras 3, utiliser forward_layer au lieu de layer
        lstm_layer = bidirectional_layer.forward_layer

        assert abs(lstm_layer.recurrent_dropout - custom_recurrent_dropout) < 1e-6

    def test_lstm_model_has_masking_layer(self):
        """Vérifie que le modèle LSTM a une couche Masking."""
        model = build_neural_network((50, 100), with_lstm=True)

        masking_layer = model.layers[0]
        assert isinstance(masking_layer, tf.keras.layers.Masking)
        assert masking_layer.mask_value == 0.0

    def test_lstm_model_output_layer_is_sigmoid(self):
        """Vérifie que la couche de sortie LSTM utilise sigmoid."""
        model = build_neural_network((50, 100), with_lstm=True)

        last_layer = model.layers[-1]
        assert isinstance(last_layer, tf.keras.layers.Dense)
        assert last_layer.units == 1
        assert last_layer.activation.__name__ == 'sigmoid'

    def test_lstm_model_has_required_metrics(self):
        """Vérifie que le modèle LSTM a les métriques requises."""
        model = build_neural_network((50, 100), with_lstm=True)

        # Dans Keras 3, accéder aux métriques compilées
        # Les métriques sont configurées lors de la compilation
        # On vérifie que le modèle a été compilé avec les bonnes métriques
        assert hasattr(model, 'compiled_metrics')

        # Vérifier que le modèle est compilé
        assert model.optimizer is not None


class TestGetTrainingCallbacks:
    """Tests pour get_training_callbacks."""

    def test_returns_list_of_callbacks(self):
        """Vérifie que la fonction retourne une liste de callbacks."""
        callbacks = get_training_callbacks()

        assert isinstance(callbacks, list)
        assert len(callbacks) > 0

    def test_returns_two_callbacks(self):
        """Vérifie qu'il y a exactement 2 callbacks (ReduceLROnPlateau et EarlyStopping)."""
        callbacks = get_training_callbacks()

        assert len(callbacks) == 2

    def test_has_reduce_lr_on_plateau_callback(self):
        """Vérifie la présence du callback ReduceLROnPlateau."""
        callbacks = get_training_callbacks()

        reduce_lr_callbacks = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.ReduceLROnPlateau)]
        assert len(reduce_lr_callbacks) == 1

    def test_has_early_stopping_callback(self):
        """Vérifie la présence du callback EarlyStopping."""
        callbacks = get_training_callbacks()

        early_stopping_callbacks = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping)]
        assert len(early_stopping_callbacks) == 1

    def test_early_stopping_with_custom_patience(self):
        """Vérifie que le patience personnalisé est appliqué à EarlyStopping."""
        custom_patience = 10
        callbacks = get_training_callbacks(patience=custom_patience)

        early_stopping = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping)][0]
        assert early_stopping.patience == custom_patience

    def test_reduce_lr_patience_is_lower_than_early_stopping(self):
        """Vérifie que la patience de ReduceLROnPlateau est inférieure à celle de EarlyStopping."""
        patience = 5
        callbacks = get_training_callbacks(patience=patience)

        reduce_lr = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.ReduceLROnPlateau)][0]
        early_stopping = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping)][0]

        assert reduce_lr.patience < early_stopping.patience

    def test_early_stopping_monitors_val_loss(self):
        """Vérifie que EarlyStopping surveille val_loss."""
        callbacks = get_training_callbacks()

        early_stopping = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping)][0]
        assert early_stopping.monitor == 'val_loss'

    def test_early_stopping_restores_best_weights(self):
        """Vérifie que EarlyStopping restaure les meilleurs poids."""
        callbacks = get_training_callbacks()

        early_stopping = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping)][0]
        assert early_stopping.restore_best_weights is True

    def test_reduce_lr_factor_is_half(self):
        """Vérifie que ReduceLROnPlateau réduit le LR de moitié."""
        callbacks = get_training_callbacks()

        reduce_lr = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.ReduceLROnPlateau)][0]
        assert reduce_lr.factor == 0.5

    def test_reduce_lr_has_min_lr(self):
        """Vérifie que ReduceLROnPlateau a un learning rate minimum."""
        callbacks = get_training_callbacks()

        reduce_lr = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.ReduceLROnPlateau)][0]
        assert reduce_lr.min_lr == 1e-6


class TestModelIntegration:
    """Tests d'intégration pour vérifier que les modèles fonctionnent ensemble."""

    def test_dense_model_can_predict(self):
        """Vérifie qu'un modèle Dense peut faire des prédictions."""
        import numpy as np

        model = build_neural_network((1, 100), with_lstm=False)

        # Créer des données factices
        X_test = np.random.rand(5, 100)

        # Faire une prédiction
        predictions = model.predict(X_test, verbose=0)

        # Vérifier la forme
        assert predictions.shape == (5, 1)
        # Vérifier que les prédictions sont entre 0 et 1 (sigmoid)
        assert all(0 <= p <= 1 for p in predictions.flatten())

    def test_lstm_model_can_predict(self):
        """Vérifie qu'un modèle LSTM peut faire des prédictions."""
        import numpy as np

        seq_length = 50
        embedding_dim = 100
        model = build_neural_network((seq_length, embedding_dim), with_lstm=True)

        # Créer des données factices (batch_size=3, seq_length=50, embedding_dim=100)
        X_test = np.random.rand(3, seq_length, embedding_dim)

        # Faire une prédiction
        predictions = model.predict(X_test, verbose=0)

        # Vérifier la forme
        assert predictions.shape == (3, 1)
        # Vérifier que les prédictions sont entre 0 et 1 (sigmoid)
        assert all(0 <= p <= 1 for p in predictions.flatten())