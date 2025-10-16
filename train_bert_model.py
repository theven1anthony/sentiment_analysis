#!/usr/bin/env python3
"""
Script d'entraînement du modèle BERT pour classification de sentiment.
Benchmark pour comparer BERT avec les autres approches.

BERT utilise le fine-tuning d'un modèle transformer pré-entraîné.
Architecture: BERT-base-uncased (110M paramètres).

Usage:
    python train_bert_model.py --technique=stemming --epochs=3
    python train_bert_model.py --technique=lemmatization --batch-size=8
    python train_bert_model.py --technique=both --sample-size=50000
"""

import os
import sys
import pandas as pd
import mlflow
import click
import time
import logging

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.text_cleaner import TextCleaner, load_sentiment140_data
from embeddings.bert_embedding import BERTEmbedding
from evaluation.metrics import ModelEvaluator
from sklearn.model_selection import train_test_split

# Import des utilitaires mutualisés (partiel pour BERT)
from utils.train_utils import (
    select_techniques_to_run,
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


def train_bert_model(X_train, y_train, X_val, y_val, X_test, y_test,
                     technique_name, max_length=64, batch_size=16, epochs=3, learning_rate=2e-5):
    """
    Entraîne un modèle BERT pour classification de sentiment.

    Args:
        X_train, X_val, X_test: Textes prétraités (pandas Series)
        y_train, y_val, y_test: Labels (numpy arrays)
        technique_name: Nom de la technique de prétraitement
        max_length: Longueur maximale des séquences
        batch_size: Taille des batchs
        epochs: Nombre d'époques
        learning_rate: Taux d'apprentissage

    Returns:
        bert, metrics
    """
    train_start = time.time()

    # 1. Initialisation BERT
    logger.info(f"Initialisation BERT (max_length={max_length}, batch_size={batch_size}, epochs={epochs})...")
    bert = BERTEmbedding(
        model_name='bert-base-uncased',
        max_length=max_length,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        output_dir=f'./models/bert_{technique_name}_tmp'
    )

    # Convertir en listes une seule fois
    X_train_list = X_train.tolist()
    y_train_list = y_train.tolist()
    X_val_list = X_val.tolist()
    y_val_list = y_val.tolist()
    X_test_list = X_test.tolist()

    # 2. Fine-tuning
    logger.info("Fine-tuning BERT (peut prendre plusieurs heures)...")
    bert.fit(X_train_list, y_train_list, X_val_list, y_val_list)

    training_time = time.time() - train_start

    # 3. Évaluation (par batches pour éviter OOM)
    logger.info("Évaluation sur le test set...")
    y_pred = bert.predict(X_test_list)
    y_pred_proba = bert.predict_proba(X_test_list)[:, 1]

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(
        y_test, y_pred, y_pred_proba,
        model_name=f"bert_{technique_name}",
        training_time=training_time
    )

    # Libérer mémoire
    del X_train_list, y_train_list, X_val_list, y_val_list, X_test_list
    import gc
    gc.collect()

    return bert, metrics


@click.command()
@click.option('--technique',
              type=click.Choice(['stemming', 'lemmatization', 'both']),
              default='both',
              help='Technique de prétraitement à utiliser')
@click.option('--description',
              default='',
              help='Description personnalisée pour l\'expérimentation')
@click.option('--experiment-name',
              default='bert_models',
              help='Nom de l\'expérimentation MLflow')
@click.option('--sample-size',
              type=int,
              default=50000,
              help='Taille de l\'échantillon pour benchmark')
@click.option('--max-length',
              type=int,
              default=64,
              help='Longueur maximale des séquences (64 optimal pour tweets)')
@click.option('--batch-size',
              type=int,
              default=32,
              help='Taille des batchs (réduire si RAM insuffisante)')
@click.option('--epochs',
              type=int,
              default=3,
              help='Nombre d\'époques d\'entraînement')
@click.option('--learning-rate',
              type=float,
              default=2e-5,
              help='Taux d\'apprentissage')
def main(technique, description, experiment_name, sample_size, max_length, batch_size, epochs, learning_rate):
    """Entraîne un modèle BERT pour classification de sentiment."""

    print("=== BENCHMARK BERT FINE-TUNING ===\n")
    print(f"Technique: {technique}")
    print(f"Model: bert-base-uncased (110M paramètres)")
    print(f"Sample size: {sample_size}")
    print(f"Max length: {max_length}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Description: {description}\n")
    print("⚠️  BERT prend beaucoup plus de temps que les autres modèles (plusieurs heures sur CPU)\n")

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

        # Stemming
        print("  Stemming...")
        text_stemmed = cleaner.preprocess_with_techniques(
            df['text'].tolist(), technique='stemming'
        )

        # Lemmatization
        print("  Lemmatization...")
        text_lemmatized = cleaner.preprocess_with_techniques(
            df['text'].tolist(), technique='lemmatization'
        )

        # Créer dataframe avec seulement les colonnes nécessaires
        df_processed = pd.DataFrame({
            'sentiment': df['sentiment'],
            'text_stemmed': text_stemmed,
            'text_lemmatized': text_lemmatized
        })

        # Filtrer textes vides
        df_processed = df_processed[
            (df_processed['text_stemmed'].str.len() > 0) &
            (df_processed['text_lemmatized'].str.len() > 0)
        ].reset_index(drop=True)

        print(f"Textes valides: {len(df_processed):,}")

        # Libérer mémoire
        del df
        import gc
        gc.collect()

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
        techniques_to_run = select_techniques_to_run(technique)
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
            run_name = f"bert_{technique_name}"

            with mlflow.start_run(run_name=run_name):
                bert, metrics = train_bert_model(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    technique_name, max_length, batch_size, epochs, learning_rate
                )

                # Log paramètres
                mlflow.log_params({
                    'technique': technique_name,
                    'model_name': 'bert-base-uncased',
                    'max_length': max_length,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'sample_size': len(df_processed),
                    'architecture': 'BERT Transformer'
                })

                # Log métriques
                mlflow.log_metrics(filter_metrics_for_mlflow(metrics))

                # Tags
                log_common_mlflow_info(description, "bert_transformer")

                results[technique_name] = {
                    'model': bert,
                    'metrics': metrics
                }

                # Afficher résultats (temps en min pour BERT)
                print(f"\nRésultats {technique_name}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
                print(f"  AUC: {metrics.get('auc_score', 0):.4f}")
                print(f"  Training time: {metrics['training_time']:.1f}s ({metrics['training_time']/60:.1f} min)")

        # 4. Comparaison et rapport
        print("\nEtape 4/4 - Génération du rapport...")

        # Créer tableau comparatif avec temps en minutes pour BERT
        comparison_data = []
        for tech_name, res in results.items():
            m = res['metrics']
            comparison_data.append({
                'Technique': tech_name.capitalize(),
                'Accuracy': m['accuracy'],
                'F1-Score': m['f1_score'],
                'AUC': m.get('auc_score', 0),
                'Time (min)': m['training_time'] / 60
            })
        df_comparison = pd.DataFrame(comparison_data)

        if len(results) > 1:
            # Afficher comparaison
            best_technique, best_metrics = find_best_technique(results)
            print("\nCOMPARAISON:")
            print(df_comparison.to_string(index=False, float_format='%.4f'))
            print(f"\nMeilleure technique: {best_technique.upper()}")

            # Créer run summary MLflow
            log_summary_run(
                run_name="bert_summary",
                technique_used=technique,
                best_technique=best_technique,
                best_metrics=best_metrics,
                total_samples=len(df_processed),
                techniques_tested=len(results),
                additional_params={
                    'model_name': 'bert-base-uncased',
                    'max_length': max_length,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'learning_rate': learning_rate
                },
                description=description
            )

            # Générer et sauvegarder rapport
            report = generate_comparison_report(
                model_name="BERT FINE-TUNING",
                architecture="BERT-base-uncased Transformer (110M paramètres)",
                df_comparison=df_comparison,
                best_technique=best_technique,
                best_metrics=best_metrics,
                dataset_size=len(df_processed),
                additional_config={
                    "Méthode": "Fine-tuning sur classification binaire",
                    "Max length": f"{max_length} tokens",
                    "Batch size": batch_size,
                    "Epochs": epochs,
                    "Learning rate": learning_rate
                },
                advantages="- Modèle transformer state-of-the-art pré-entraîné\n- Capture contexte bidirectionnel complet\n- Fine-tuning adapte le modèle à la tâche spécifique",
                conclusion="Ce modèle utilise BERT fine-tuné et servira de référence state-of-the-art pour comparer avec les autres approches (TF-IDF, Word2Vec, FastText, USE)."
            )

            report_path = save_report(report, "bert")

            # Logger le rapport dans MLflow
            with mlflow.start_run(run_name="bert_summary", nested=True):
                mlflow.log_artifact(report_path, "reports")

            print(f"\nRapport sauvegardé: {report_path}")

        else:
            # Une seule technique
            technique_name = list(results.keys())[0]
            best_metrics = results[technique_name]['metrics']

        # Message de fin avec temps en minutes pour BERT
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