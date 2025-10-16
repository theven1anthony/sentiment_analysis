#!/usr/bin/env python3
"""
Script d'entraînement du modèle Universal Sentence Encoder (USE) avec réseau de neurones.
Benchmark pour comparer différentes configurations d'embeddings.

USE utilise un modèle transformer pré-entraîné qui encode des phrases complètes.
IMPORTANT: USE produit uniquement des embeddings sentence-level (512 dimensions).
Il ne peut être utilisé qu'avec architecture Dense (pas de LSTM).

Usage:
    python train_use_model.py --technique=stemming
    python train_use_model.py --technique=lemmatization
    python train_use_model.py --technique=both --sample-size=50000
"""

import os
import sys
import mlflow
import mlflow.tensorflow
import click
import time
import logging

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from embeddings.use_embedding import USEEmbedding
from utils.model_utils import build_neural_network, get_training_callbacks
from evaluation.metrics import ModelEvaluator, log_training_history

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


def train_use_model(X_train, y_train, X_val, y_val, X_test, y_test, technique_name):
    """
    Entraîne un modèle USE + réseau de neurones Dense.

    Args:
        X_train, X_val, X_test: Textes prétraités
        y_train, y_val, y_test: Labels
        technique_name: Nom de la technique de prétraitement

    Returns:
        model, metrics, history
    """
    train_start = time.time()

    # 1. Chargement du modèle USE pré-entraîné
    logger.info("Chargement Universal Sentence Encoder...")
    use = USEEmbedding()
    use.fit(X_train.tolist())

    # 2. Transformation des textes
    logger.info("Transformation en embeddings USE (sentence-level, 512 dim)...")
    X_train_emb = use.transform(X_train.tolist())
    X_val_emb = use.transform(X_val.tolist())
    X_test_emb = use.transform(X_test.tolist())

    logger.info(f"Shape des embeddings: {X_train_emb.shape}")

    # 3. Construction du modèle Dense
    logger.info("Construction du modèle Dense...")
    input_dim = (None, 512)  # USE produit toujours 512 dimensions
    model = build_neural_network(
        input_dim=input_dim,
        with_lstm=False,  # USE uniquement avec Dense
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
        model_name=f"use_{technique_name}",
        training_time=training_time
    )

    # Ajouter métriques supplémentaires
    metrics['epochs_trained'] = len(history.history['loss'])

    return model, metrics, history


@click.command()
@click.option('--technique',
              type=click.Choice(['stemming', 'lemmatization', 'both']),
              default='both',
              help='Technique de prétraitement à utiliser')
@click.option('--description',
              default='',
              help='Description personnalisée pour l\'expérimentation')
@click.option('--experiment-name',
              default='use_models',
              help='Nom de l\'expérimentation MLflow')
@click.option('--sample-size',
              type=int,
              default=50000,
              help='Taille de l\'échantillon pour benchmark')
def main(technique, description, experiment_name, sample_size):
    """Entraîne un modèle Universal Sentence Encoder avec réseau de neurones Dense."""

    # Header
    print_training_header(
        "UNIVERSAL SENTENCE ENCODER + DENSE",
        technique,
        description,
        {
            "Architecture": "Dense (USE produit embeddings sentence-level)",
            "Sample size": sample_size,
            "Vector size": "512 (fixe pour USE)"
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
            run_name = f"use_{technique_name}_dense"

            with mlflow.start_run(run_name=run_name):
                model, metrics, history = train_use_model(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    technique_name
                )

                # Log paramètres
                mlflow.log_params({
                    'technique': technique_name,
                    'architecture': 'Dense',
                    'vector_size': 512,
                    'sample_size': len(df_processed),
                    'model_type': 'use_pretrained'
                })

                # Log métriques
                mlflow.log_metrics(filter_metrics_for_mlflow(metrics))

                # Log courbes d'entraînement
                log_training_history(history.history, model_name=run_name)

                # Log modèle
                mlflow.tensorflow.log_model(model, "model")

                # Tags
                log_common_mlflow_info(description, "use_neural")

                results[technique_name] = {
                    'model': model,
                    'metrics': metrics
                }

                # Afficher résultats
                print_results_summary(
                    technique_name,
                    metrics,
                    additional_metrics=['epochs_trained']
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
            log_summary_run(
                run_name="use_dense_summary",
                technique_used=technique,
                best_technique=best_technique,
                best_metrics=best_metrics,
                total_samples=len(df_processed),
                techniques_tested=len(results),
                additional_params={
                    'architecture': 'Dense',
                    'vector_size': 512
                },
                description=description
            )

            # Générer et sauvegarder rapport
            report = generate_comparison_report(
                model_name="UNIVERSAL SENTENCE ENCODER + DENSE",
                architecture="Dense Neural Network",
                df_comparison=df_comparison,
                best_technique=best_technique,
                best_metrics=best_metrics,
                dataset_size=len(df_processed),
                additional_config={
                    "USE": "Modèle transformer pré-entraîné, embeddings sentence-level (512 dim)"
                },
                advantages="- Modèle pré-entraîné sur large corpus (meilleure qualité d'embeddings)\n- Encode des phrases complètes (capture le contexte global)\n- State-of-the-art pour sentence embeddings",
                conclusion="Ce modèle utilise USE pour les embeddings et servira de référence pour comparer avec d'autres techniques d'embedding (Word2Vec, FastText, BERT)."
            )

            report_path = save_report(report, "use_dense")

            # Logger le rapport dans MLflow
            with mlflow.start_run(run_name="use_dense_summary", nested=True):
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