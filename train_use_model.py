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
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import click
from datetime import datetime
import time
import logging

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.text_cleaner import TextCleaner, load_sentiment140_data
from embeddings.use_embedding import USEEmbedding
from utils.model_utils import build_neural_network, get_training_callbacks
from evaluation.metrics import ModelEvaluator, log_training_history
from sklearn.model_selection import train_test_split

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

    print("=== BENCHMARK UNIVERSAL SENTENCE ENCODER + DENSE ===\n")
    print(f"Technique: {technique}")
    print(f"Architecture: Dense (USE produit embeddings sentence-level)")
    print(f"Sample size: {sample_size}")
    print(f"Vector size: 512 (fixe pour USE)")
    print(f"Description: {description}\n")

    mlflow.set_experiment(experiment_name)

    try:
        # 1. Chargement des données
        print("Etape 1/4 - Chargement des données...")
        start_time = time.time()

        df = load_sentiment140_data(
            "data/training.1600000.processed.noemoticon.csv",
            sample_size=sample_size
        )
        print(f"Dataset chargé: {len(df):,} échantillons")
        print(f"Distribution: {df['sentiment'].value_counts().to_dict()}")

        # 2. Prétraitement
        print("\nEtape 2/4 - Prétraitement...")
        cleaner = TextCleaner()

        df_processed = df.copy()

        # Stemming
        print("  Stemming...")
        df_processed['text_stemmed'] = cleaner.preprocess_with_techniques(
            df['text'].tolist(), technique='stemming'
        )
        df_processed = df_processed[df_processed['text_stemmed'].str.len() > 0].reset_index(drop=True)

        # Lemmatization
        print("  Lemmatization...")
        df_processed['text_lemmatized'] = cleaner.preprocess_with_techniques(
            df_processed['text'].tolist(), technique='lemmatization'
        )
        df_processed = df_processed[df_processed['text_lemmatized'].str.len() > 0].reset_index(drop=True)

        print(f"Textes valides: {len(df_processed):,}")

        # Création des splits
        print("  Création des splits train/val/test (70/15/15)...")
        train_idx, temp_idx = train_test_split(
            range(len(df_processed)), test_size=0.3, random_state=42,
            stratify=df_processed['sentiment']
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42,
            stratify=df_processed.iloc[temp_idx]['sentiment']
        )

        splits_data = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }

        print(f"  Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

        # 3. Entraînement
        print(f"\nEtape 3/4 - Entraînement (technique={technique})...")

        techniques_to_run = []
        if technique == 'both':
            techniques_to_run = [('stemming', 'text_stemmed'), ('lemmatization', 'text_lemmatized')]
        elif technique == 'stemming':
            techniques_to_run = [('stemming', 'text_stemmed')]
        else:
            techniques_to_run = [('lemmatization', 'text_lemmatized')]

        results = {}

        for technique_name, text_column in techniques_to_run:
            print(f"\n=== TECHNIQUE: {technique_name.upper()} ===")

            # Préparer les données
            X_train = df_processed.iloc[splits_data['train_idx']][text_column]
            X_val = df_processed.iloc[splits_data['val_idx']][text_column]
            X_test = df_processed.iloc[splits_data['test_idx']][text_column]
            y_train = df_processed.iloc[splits_data['train_idx']]['sentiment'].values
            y_val = df_processed.iloc[splits_data['val_idx']]['sentiment'].values
            y_test = df_processed.iloc[splits_data['test_idx']]['sentiment'].values

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

                # Log métriques (sans confusion_matrix et classification_report)
                metrics_to_log = {k: v for k, v in metrics.items()
                                 if k not in ['confusion_matrix', 'classification_report', 'model_name']}
                mlflow.log_metrics(metrics_to_log)

                # Log courbes d'entraînement
                log_training_history(history.history, model_name=run_name)

                # Log modèle
                mlflow.tensorflow.log_model(model, "model")

                # Tags
                if description:
                    mlflow.set_tag("description", description)
                mlflow.set_tag("model_type", "use_neural")

                results[technique_name] = {
                    'model': model,
                    'metrics': metrics
                }

                print(f"\nRésultats {technique_name}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
                print(f"  AUC: {metrics.get('auc_score', 0):.4f}")
                print(f"  Training time: {metrics['training_time']:.1f}s")
                print(f"  Epochs: {metrics['epochs_trained']}")

        # 4. Comparaison et rapport
        print("\nEtape 4/4 - Génération du rapport...")

        # Tableau comparatif
        comparison_data = []
        for tech_name, res in results.items():
            m = res['metrics']
            comparison_data.append({
                'Technique': tech_name.capitalize(),
                'Accuracy': m['accuracy'],
                'F1-Score': m['f1_score'],
                'AUC': m.get('auc_score', 0),
                'Time (s)': m['training_time']
            })

        df_comparison = pd.DataFrame(comparison_data)

        if len(results) > 1:
            print("\nCOMPARAISON:")
            print(df_comparison.to_string(index=False, float_format='%.4f'))

            best_technique = max(results.keys(),
                               key=lambda k: results[k]['metrics']['f1_score'])
            print(f"\nMeilleure technique: {best_technique.upper()}")

            best_metrics = results[best_technique]['metrics']

            # Créer une run summary pour MLflow
            with mlflow.start_run(run_name="use_dense_summary"):
                mlflow.set_tag("summary_type", "final_report")
                if description:
                    mlflow.set_tag("description", description)

                # Log paramètres
                mlflow.log_params({
                    'technique_used': technique,
                    'best_technique': best_technique,
                    'architecture': 'Dense',
                    'total_samples': len(df_processed),
                    'techniques_tested': len(results),
                    'vector_size': 512
                })

                # Log métriques du meilleur modèle
                summary_metrics = {
                    'best_accuracy': best_metrics['accuracy'],
                    'best_f1_score': best_metrics['f1_score'],
                    'best_precision': best_metrics['precision'],
                    'best_recall': best_metrics['recall']
                }
                if 'auc_score' in best_metrics:
                    summary_metrics['best_auc_score'] = best_metrics['auc_score']

                mlflow.log_metrics(summary_metrics)

                # Sauvegarder le rapport
                report = f"""RAPPORT DE COMPARAISON - UNIVERSAL SENTENCE ENCODER + DENSE

OBJECTIF:
Comparer les techniques de prétraitement pour le modèle USE

CONFIGURATION:
- Architecture: Dense Neural Network
- USE: Modèle transformer pré-entraîné, embeddings sentence-level (512 dim)
- Dataset: {len(df_processed)} échantillons

AVANTAGE USE:
- Modèle pré-entraîné sur large corpus (meilleure qualité d'embeddings)
- Encode des phrases complètes (capture le contexte global)
- State-of-the-art pour sentence embeddings

RÉSULTATS COMPARATIFS:
{df_comparison.to_string(index=False, float_format='%.4f')}

MODÈLE RETENU: {best_technique.capitalize()}
- Accuracy: {best_metrics['accuracy']:.4f}
- F1-Score: {best_metrics['f1_score']:.4f}
- Précision: {best_metrics['precision']:.4f}
- Rappel: {best_metrics['recall']:.4f}
- Temps d'entraînement: {best_metrics['training_time']:.2f}s
- Epochs entraînés: {best_metrics['epochs_trained']}

CONCLUSION:
Ce modèle utilise USE pour les embeddings et servira de référence pour comparer avec d'autres techniques d'embedding (Word2Vec, FastText, BERT).
"""

                report_path = f"reports/use_dense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)

                mlflow.log_artifact(report_path, "reports")

                print(f"\nRapport sauvegardé: {report_path}")

        else:
            # Une seule technique - log quand même
            technique_name = list(results.keys())[0]
            best_metrics = results[technique_name]['metrics']

        total_time = time.time() - start_time
        print(f"\n=== BENCHMARK TERMINÉ ===")
        print(f"Temps total: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"MLflow UI: http://localhost:5001")

    except Exception as e:
        print(f"\nErreur: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()