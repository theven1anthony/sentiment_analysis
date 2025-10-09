import tensorflow as tf


def build_neural_network(input_dim: tuple[int, int], with_lstm: bool = False, lstm_units: int = 128,
                         dropout: float = 0.4, recurrent_dropout: float = 0.3, learning_rate: float = 0.001):
    if with_lstm:
        # Architecture avec Bidirectional LSTM sur séquences
        model = tf.keras.Sequential([
            tf.keras.layers.Masking(mask_value=0.0, input_shape=(input_dim[0], input_dim[1])),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, recurrent_dropout=recurrent_dropout)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(dropout * 0.75),  # Dropout légèrement réduit pour couche finale
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    else:
        # Architecture Dense optimisée pour embeddings de documents
        embedding_dim = input_dim[1]
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(embedding_dim,)),
            tf.keras.layers.Dropout(dropout * 0.75),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(dropout * 0.75),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.F1Score(name='f1_score', threshold=0.5)
        ]
    )

    return model


def get_training_callbacks(patience: int = 5):
    """
    Crée les callbacks pour l'entraînement Keras.

    Args:
        patience: Nombre d'epochs sans amélioration avant early stopping

    Returns:
        Liste de callbacks Keras
    """
    return [
        # Réduction du learning rate si plateau (toujours actif)
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Divise le LR par 2
            patience=max(1, patience - 1),  # 1 epoch avant early stopping
            min_delta=0.001,  # Sensible aux petites améliorations
            min_lr=1e-6,
            verbose=1
        ),
        # Arrêt anticipé si pas d'amélioration (toujours actif)
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',  # Sur valeur minimale
            patience=patience,
            min_delta=0.001,  # Amélioration minimale absolue requise (0.1%)
            restore_best_weights=True,  # Restaure les meilleurs poids
            verbose=1
        )
    ]