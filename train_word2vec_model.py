#!/usr/bin/env python3
"""
Script d'entraînement du modèle Word2Vec avec réseau de neurones.
Benchmark pour comparer différentes configurations d'embeddings.

Usage:
    python train_word2vec_model.py --technique=stemming --with-lstm
    python train_word2vec_model.py --technique=lemmatization
    python train_word2vec_model.py --technique=both --sample-size=50000
"""

import os
import sys
import pickle
import mlflow
import mlflow.tensorflow
import click
import time
import logging

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from embeddings.word2vec_embedding import Word2VecEmbedding
from utils.model_utils import build_neural_network, get_training_callbacks
from evaluation.metrics import ModelEvaluator, log_training_history
from models.word2vec_sentiment_model import Word2VecSentimentModel

# Import des utilitaires mutualisés
from utils.train_utils import (
    load_and_preprocess_data,
    create_train_val_test_splits,
    select_techniques_to_run,
    prepare_data_for_training,
    filter_metrics_for_mlflow,
    log_common_mlflow_info,
    create_comparison_dataframe,
    find_best_technique,
    generate_comparison_report,
    save_report,
    log_summary_run,
    print_training_header,
    print_results_summary,
    print_comparison,
    print_completion_message
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_word2vec_model(X_train, y_train, X_val, y_val, X_test, y_test,
                         technique_name, with_lstm, vector_size=100):
    """
    Entraîne un modèle Word2Vec + réseau de neurones.

    Args:
        X_train, X_val, X_test: Textes prétraités
        y_train, y_val, y_test: Labels
        technique_name: Nom de la technique de prétraitement
        with_lstm: Si True, utilise architecture LSTM, sinon Dense
        vector_size: Dimension des vecteurs Word2Vec

    Returns:
        model, metrics, vocab_size
    """
    train_start = time.time()

    # 1. Création des embeddings Word2Vec
    logger.info(f"Entraînement Word2Vec (vector_size={vector_size})...")
    w2v = Word2VecEmbedding(vector_size=vector_size, epochs=10, sg=1)
    w2v.fit(X_train.tolist())
    vocab_size = len(w2v.model.wv)

    # 2. Transformation des textes
    logger.info(f"Transformation en embeddings (LSTM={with_lstm})...")
    if with_lstm:
        # Séquences pour LSTM
        max_len = 50
        X_train_emb = w2v.transform(X_train.tolist(), max_len=max_len, average=False)
        X_val_emb = w2v.transform(X_val.tolist(), max_len=max_len, average=False)
        X_test_emb = w2v.transform(X_test.tolist(), max_len=max_len, average=False)
        input_dim = (max_len, vector_size)
    else:
        # Embeddings moyennés pour Dense
        X_train_emb = w2v.transform(X_train.tolist(), average=True)
        X_val_emb = w2v.transform(X_val.tolist(), average=True)
        X_test_emb = w2v.transform(X_test.tolist(), average=True)
        input_dim = (None, vector_size)

    logger.info(f"Shape des embeddings: {X_train_emb.shape}")

    # 3. Construction du modèle
    logger.info(f"Construction du modèle (with_lstm={with_lstm})...")
    model = build_neural_network(
        input_dim=input_dim,
        with_lstm=with_lstm,
        lstm_units=128,
        learning_rate=0.001
    )

    logger.info(f"Nombre de paramètres: {model.count_params():,}")

    # 4. Entraînement
    logger.info("Entraînement du modèle...")
    callbacks = get_training_callbacks(patience=5)

    # Reshape labels pour F1Score (batch,) → (batch, 1)
    y_train_reshaped = y_train.reshape(-1, 1)
    y_val_reshaped = y_val.reshape(-1, 1)

    history = model.fit(
        X_train_emb, y_train_reshaped,
        validation_data=(X_val_emb, y_val_reshaped),
        epochs=30,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - train_start

    # 5. Évaluation avec ModelEvaluator
    logger.info("Évaluation sur le test set...")
    y_pred_proba = model.predict(X_test_emb, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(
        y_test, y_pred, y_pred_proba,
        model_name=f"word2vec_{technique_name}",
        training_time=training_time
    )

    # Ajouter métriques supplémentaires
    metrics['epochs_trained'] = len(history.history['loss'])
    metrics['vocab_size'] = vocab_size

    return model, metrics, history, w2v


@click.command()
@click.option('--technique',
              type=click.Choice(['stemming', 'lemmatization', 'both']),
              default='both',
              help='Technique de prétraitement à utiliser')
@click.option('--description',
              default='',
              help='Description personnalisée pour l\'expérimentation')
@click.option('--experiment-name',
              default='word2vec_models',
              help='Nom de l\'expérimentation MLflow')
@click.option('--sample-size',
              type=int,
              default=50000,
              help='Taille de l\'échantillon pour benchmark')
@click.option('--with-lstm',
              is_flag=True,
              default=False,
              help='Utiliser architecture LSTM au lieu de Dense')
@click.option('--vector-size',
              type=int,
              default=100,
              help='Dimension des vecteurs Word2Vec')
def main(technique, description, experiment_name, sample_size, with_lstm, vector_size):
    """Entraîne un modèle Word2Vec avec réseau de neurones."""

    # Header
    print_training_header(
        "WORD2VEC + NEURAL NETWORK",
        technique,
        description,
        {
            "Architecture": 'LSTM' if with_lstm else 'Dense',
            "Sample size": sample_size,
            "Vector size": vector_size
        }
    )

    mlflow.set_experiment(experiment_name)

    try:
        # 1. Chargement et prétraitement
        print("Etape 1/4 - Chargement et prétraitement...")
        start_time = time.time()

        df_processed, _ = load_and_preprocess_data(
            "data/training.1600000.processed.noemoticon.csv",
            sample_size=sample_size
        )

        # 2. Création des splits
        print("\nEtape 2/4 - Création des splits...")
        splits_data = create_train_val_test_splits(df_processed)

        # 3. Entraînement
        print(f"\nEtape 3/4 - Entraînement (technique={technique})...")
        techniques_to_run = select_techniques_to_run(technique)
        results = {}

        for technique_name, text_column in techniques_to_run:
            print(f"\n=== TECHNIQUE: {technique_name.upper()} ===")

            # Préparer les données
            X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_training(
                df_processed, splits_data, text_column
            )

            # Entraîner avec MLflow
            arch_name = 'lstm' if with_lstm else 'dense'
            run_name = f"word2vec_{technique_name}_{arch_name}"

            with mlflow.start_run(run_name=run_name):
                model, metrics, history, w2v = train_word2vec_model(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    technique_name, with_lstm, vector_size
                )

                # Log paramètres
                mlflow.log_params({
                    'technique': technique_name,
                    'architecture': 'LSTM' if with_lstm else 'Dense',
                    'vector_size': vector_size,
                    'sample_size': len(df_processed),
                    'vocab_size': metrics['vocab_size']
                })

                # Log métriques
                mlflow.log_metrics(filter_metrics_for_mlflow(metrics))

                # Log courbes d'entraînement
                log_training_history(history.history, model_name=run_name)

                # Sauvegarder le modèle complet en tant que pyfunc MLflow
                print(f"\nSauvegarde du modèle pyfunc MLflow...")

                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Sauvegarder le modèle Keras
                    keras_model_path = os.path.join(tmpdir, "keras_model.keras")
                    model.save(keras_model_path)

                    # Sauvegarder l'embedding Word2Vec
                    w2v_path = os.path.join(tmpdir, "word2vec_embedding.pkl")
                    with open(w2v_path, 'wb') as f:
                        pickle.dump(w2v, f)

                    # Créer le fichier de métadonnées technique
                    technique_path = os.path.join(tmpdir, "technique.txt")
                    with open(technique_path, 'w') as f:
                        f.write(technique_name)

                    # Définir les artifacts
                    artifacts = {
                        "keras_model": keras_model_path,
                        "word2vec_embedding": w2v_path,
                        "technique": technique_path
                    }

                    # Définir le conda_env avec versions compatibles
                    import tensorflow as tf_version
                    conda_env = {
                        'channels': ['defaults', 'conda-forge'],
                        'dependencies': [
                            f'python={sys.version_info.major}.{sys.version_info.minor}',
                            'pip',
                            {
                                'pip': [
                                    f'mlflow=={mlflow.__version__}',
                                    f'tensorflow=={tf_version.__version__}',
                                    'numpy>=1.26.0',
                                    'pandas>=2.0.0',
                                    'scikit-learn>=1.3.0',
                                    'gensim>=4.3.2',
                                    'nltk>=3.9.0'
                                ]
                            }
                        ],
                        'name': 'word2vec_sentiment_env'
                    }

                    # Logger le modèle pyfunc
                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=Word2VecSentimentModel(),
                        artifacts=artifacts,
                        conda_env=conda_env
                    )

                    print(f"✓ Modèle pyfunc MLflow sauvegardé (Keras + Word2Vec + preprocessing)")

                # Tags
                log_common_mlflow_info(description, "word2vec_neural")

                results[technique_name] = {
                    'model': model,
                    'metrics': metrics
                }

                # Afficher résultats
                print_results_summary(
                    technique_name,
                    metrics,
                    additional_metrics=['epochs_trained', 'vocab_size']
                )

        # 4. Comparaison et rapport
        print("\nEtape 4/4 - Génération du rapport...")

        # Créer tableau comparatif
        df_comparison = create_comparison_dataframe(results)

        if len(results) > 1:
            # Afficher comparaison
            best_technique, best_metrics = find_best_technique(results)
            print_comparison(df_comparison, best_technique)

            # Créer run summary MLflow
            arch_name = 'lstm' if with_lstm else 'dense'
            log_summary_run(
                run_name=f"word2vec_{arch_name}_summary",
                technique_used=technique,
                best_technique=best_technique,
                best_metrics=best_metrics,
                total_samples=len(df_processed),
                techniques_tested=len(results),
                additional_params={
                    'architecture': 'LSTM' if with_lstm else 'Dense',
                    'vector_size': vector_size
                },
                description=description
            )

            # Générer et sauvegarder rapport
            arch_description = 'Bidirectional LSTM' if with_lstm else 'Dense Neural Network'
            report = generate_comparison_report(
                model_name="WORD2VEC + " + ("LSTM" if with_lstm else "DENSE"),
                architecture=arch_description,
                df_comparison=df_comparison,
                best_technique=best_technique,
                best_metrics=best_metrics,
                dataset_size=len(df_processed),
                additional_config={
                    "Word2Vec": f"Skip-gram, vector_size={vector_size}"
                },
                conclusion="Ce modèle utilise Word2Vec pour les embeddings et servira de référence pour comparer avec d'autres techniques d'embedding (FastText, USE, BERT)."
            )

            report_path = save_report(report, f"word2vec_{arch_name}")

            # Logger le rapport dans MLflow
            with mlflow.start_run(run_name=f"word2vec_{arch_name}_summary", nested=True):
                mlflow.log_artifact(report_path, "reports")

            print(f"\nRapport sauvegardé: {report_path}")

        else:
            # Une seule technique
            technique_name = list(results.keys())[0]
            best_metrics = results[technique_name]['metrics']

        # Message de fin
        total_time = time.time() - start_time
        print_completion_message(total_time)

    except Exception as e:
        print(f"\nErreur: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()