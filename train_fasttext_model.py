#!/usr/bin/env python3
"""
Script d'entraînement du modèle FastText avec réseau de neurones.
Benchmark pour comparer différentes configurations d'embeddings.

FastText améliore Word2Vec en gérant les mots hors vocabulaire via n-grammes de caractères.

Usage:
    python train_fasttext_model.py --technique=stemming --with-lstm
    python train_fasttext_model.py --technique=lemmatization
    python train_fasttext_model.py --technique=both --sample-size=50000
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
from embeddings.fasttext_embedding import FastTextEmbedding
from utils.model_utils import build_neural_network, get_training_callbacks
from evaluation.metrics import ModelEvaluator, log_training_history
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_fasttext_model(X_train, y_train, X_val, y_val, X_test, y_test,
                         technique_name, with_lstm, vector_size=100):
    """
    Entraîne un modèle FastText + réseau de neurones.

    Args:
        X_train, X_val, X_test: Textes prétraités
        y_train, y_val, y_test: Labels
        technique_name: Nom de la technique de prétraitement
        with_lstm: Si True, utilise architecture LSTM, sinon Dense
        vector_size: Dimension des vecteurs FastText

    Returns:
        model, metrics, vocab_size
    """
    train_start = time.time()

    # 1. Création des embeddings FastText
    logger.info(f"Entraînement FastText (vector_size={vector_size})...")
    ft = FastTextEmbedding(vector_size=vector_size, epochs=10, sg=1, min_n=3, max_n=6)
    ft.fit(X_train.tolist())
    vocab_size = len(ft.model.wv)

    # 2. Transformation des textes
    logger.info(f"Transformation en embeddings (LSTM={with_lstm})...")
    if with_lstm:
        # Séquences pour LSTM
        max_len = 50
        X_train_emb = ft.transform(X_train.tolist(), max_len=max_len, average=False)
        X_val_emb = ft.transform(X_val.tolist(), max_len=max_len, average=False)
        X_test_emb = ft.transform(X_test.tolist(), max_len=max_len, average=False)
        input_dim = (max_len, vector_size)
    else:
        # Embeddings moyennés pour Dense
        X_train_emb = ft.transform(X_train.tolist(), average=True)
        X_val_emb = ft.transform(X_val.tolist(), average=True)
        X_test_emb = ft.transform(X_test.tolist(), average=True)
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
        model_name=f"fasttext_{technique_name}",
        training_time=training_time
    )

    # Ajouter métriques supplémentaires
    metrics['epochs_trained'] = len(history.history['loss'])
    metrics['vocab_size'] = vocab_size

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
              default='fasttext_models',
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
              help='Dimension des vecteurs FastText')
def main(technique, description, experiment_name, sample_size, with_lstm, vector_size):
    """Entraîne un modèle FastText avec réseau de neurones."""

    print("=== BENCHMARK FASTTEXT + NEURAL NETWORK ===\n")
    print(f"Technique: {technique}")
    print(f"Architecture: {'LSTM' if with_lstm else 'Dense'}")
    print(f"Sample size: {sample_size}")
    print(f"Vector size: {vector_size}")
    print(f"N-grammes: 3-6 caractères (gestion OOV)")
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
            arch_name = 'lstm' if with_lstm else 'dense'
            run_name = f"fasttext_{technique_name}_{arch_name}"

            with mlflow.start_run(run_name=run_name):
                model, metrics, history = train_fasttext_model(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    technique_name, with_lstm, vector_size
                )

                # Log paramètres
                mlflow.log_params({
                    'technique': technique_name,
                    'architecture': 'LSTM' if with_lstm else 'Dense',
                    'vector_size': vector_size,
                    'sample_size': len(df_processed),
                    'vocab_size': metrics['vocab_size'],
                    'min_n': 3,
                    'max_n': 6
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
                mlflow.set_tag("model_type", "fasttext_neural")

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
                print(f"  Vocab size: {metrics['vocab_size']}")

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
            arch_name = 'lstm' if with_lstm else 'dense'
            with mlflow.start_run(run_name=f"fasttext_{arch_name}_summary"):
                mlflow.set_tag("summary_type", "final_report")
                if description:
                    mlflow.set_tag("description", description)

                # Log paramètres
                mlflow.log_params({
                    'technique_used': technique,
                    'best_technique': best_technique,
                    'architecture': 'LSTM' if with_lstm else 'Dense',
                    'total_samples': len(df_processed),
                    'techniques_tested': len(results),
                    'vector_size': vector_size,
                    'min_n': 3,
                    'max_n': 6
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
                report = f"""RAPPORT DE COMPARAISON - FASTTEXT + {'LSTM' if with_lstm else 'DENSE'}

OBJECTIF:
Comparer les techniques de prétraitement pour le modèle FastText

CONFIGURATION:
- Architecture: {'Bidirectional LSTM' if with_lstm else 'Dense Neural Network'}
- FastText: Skip-gram, vector_size={vector_size}, n-grammes=[3,6]
- Dataset: {len(df_processed)} échantillons

AVANTAGE FASTTEXT:
- Gère les mots hors vocabulaire via n-grammes de caractères
- Robuste aux typos et variations orthographiques

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
Ce modèle utilise FastText pour les embeddings et servira de référence pour comparer avec d'autres techniques d'embedding (Word2Vec, USE, BERT).
"""

                report_path = f"reports/fasttext_{arch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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