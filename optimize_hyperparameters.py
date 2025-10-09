#!/usr/bin/env python3
"""
Script d'optimisation d'hyperparamètres pour Word2Vec LSTM.
Utilise Random Search pour trouver la meilleure configuration (Target: F1 ≥ 0.80).

Usage:
    python optimize_hyperparameters.py --n-runs=20 --sample-size=200000
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.pyfunc
import click
from datetime import datetime
import time
import logging
from sklearn.model_selection import train_test_split, ParameterSampler
from scipy.stats import uniform, randint

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.text_cleaner import TextCleaner, load_sentiment140_data
from embeddings.word2vec_embedding import Word2VecEmbedding
from utils.model_utils import build_neural_network, get_training_callbacks
from evaluation.metrics import ModelEvaluator, log_training_history
from models.word2vec_sentiment_model import Word2VecSentimentModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_hyperparams(X_train, y_train, X_val, y_val, X_test, y_test,
                           hyperparams, technique_name='stemming'):
    """
    Entraîne un modèle avec des hyperparamètres spécifiques.

    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Données d'entraînement
        hyperparams: Dictionnaire des hyperparamètres
        technique_name: Technique de preprocessing

    Returns:
        metrics: Dictionnaire des métriques d'évaluation
    """
    train_start = time.time()

    # 1. Créer l'embedding Word2Vec avec hyperparamètres
    logger.info(f"Word2Vec: vector_size={hyperparams['vector_size']}, "
                f"window={hyperparams['window']}, min_count={hyperparams['min_count']}")

    w2v = Word2VecEmbedding(
        vector_size=hyperparams['vector_size'],
        window=hyperparams['window'],
        min_count=hyperparams['min_count'],
        epochs=10,
        sg=1
    )
    w2v.fit(X_train.tolist())
    vocab_size = len(w2v.model.wv)

    # 2. Transformation en embeddings LSTM
    max_len = 50
    X_train_emb = w2v.transform(X_train.tolist(), max_len=max_len, average=False)
    X_val_emb = w2v.transform(X_val.tolist(), max_len=max_len, average=False)
    X_test_emb = w2v.transform(X_test.tolist(), max_len=max_len, average=False)

    logger.info(f"Embeddings shape: {X_train_emb.shape}")

    # 3. Construction du modèle avec hyperparamètres
    logger.info(f"LSTM: units={hyperparams['lstm_units']}, "
                f"dropout={hyperparams['dropout']}, "
                f"recurrent_dropout={hyperparams['recurrent_dropout']}")

    model = build_neural_network(
        input_dim=(max_len, hyperparams['vector_size']),
        with_lstm=True,
        lstm_units=hyperparams['lstm_units'],
        dropout=hyperparams['dropout'],
        recurrent_dropout=hyperparams['recurrent_dropout'],
        learning_rate=hyperparams['learning_rate']
    )

    logger.info(f"Paramètres du modèle: {model.count_params():,}")

    # 4. Entraînement
    logger.info(f"Entraînement: lr={hyperparams['learning_rate']}, batch={hyperparams['batch_size']}")

    callbacks = get_training_callbacks(patience=5)
    y_train_reshaped = y_train.reshape(-1, 1)
    y_val_reshaped = y_val.reshape(-1, 1)

    history = model.fit(
        X_train_emb, y_train_reshaped,
        validation_data=(X_val_emb, y_val_reshaped),
        epochs=30,
        batch_size=hyperparams['batch_size'],
        callbacks=callbacks,
        verbose=2  # Affiche progression (1 ligne par epoch)
    )

    training_time = time.time() - train_start

    # 5. Évaluation
    logger.info(f"Évaluation sur {len(X_test_emb)} exemples...")
    y_pred_proba = model.predict(X_test_emb, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(
        y_test, y_pred, y_pred_proba,
        model_name=f"w2v_opt",
        training_time=training_time
    )

    metrics['epochs_trained'] = len(history.history['loss'])
    metrics['vocab_size'] = vocab_size

    return model, metrics, history, w2v


@click.command()
@click.option('--n-runs', type=int, default=20, help='Nombre de runs Random Search')
@click.option('--sample-size', type=int, default=200000, help='Taille du dataset')
@click.option('--experiment-name', default='hyperparameter_optimization', help='Nom expérimentation MLflow')
@click.option('--technique', default='stemming', help='Technique de preprocessing')
def main(n_runs, sample_size, experiment_name, technique):
    """
    Optimisation d'hyperparamètres avec Random Search.
    """

    print("=== OPTIMISATION D'HYPERPARAMÈTRES (RANDOM SEARCH) ===\n")
    print(f"Nombre de runs: {n_runs}")
    print(f"Sample size: {sample_size}")
    print(f"Technique: {technique}\n")

    # Définir l'espace de recherche
    param_distributions = {
        # Capacité du modèle
        'vector_size': [100, 110, 120],
        'lstm_units': [128, 144],

        # Word2Vec
        'window': [5, 7],
        'min_count': [1, 2],

        # LSTM - Régularisation
        'dropout': [0.3, 0.4],
        'recurrent_dropout': [0.2, 0.3],

        # Entraînement
        'learning_rate': [0.0005, 0.001],
        'batch_size': [32]  # Fixe
    }

    # Générer les combinaisons aléatoires
    param_list = list(ParameterSampler(
        param_distributions,
        n_iter=n_runs,
        random_state=42
    ))

    print(f"Espace de recherche:")
    for key, values in param_distributions.items():
        if len(values) > 1:
            print(f"  {key}: {values}")
    print(f"\nCombinaisons à tester: {n_runs}\n")

    mlflow.set_experiment(experiment_name)

    try:
        # 1. Chargement des données
        print("Étape 1/3 - Chargement des données...")
        start_time = time.time()

        df = load_sentiment140_data(
            "data/training.1600000.processed.noemoticon.csv",
            sample_size=sample_size
        )
        print(f"Dataset chargé: {len(df):,} échantillons")

        # 2. Prétraitement
        print("\nÉtape 2/3 - Prétraitement...")
        cleaner = TextCleaner()

        df_processed = df.copy()
        df_processed['text_stemmed'] = cleaner.preprocess_with_techniques(
            df['text'].tolist(), technique='stemming'
        )
        df_processed = df_processed[df_processed['text_stemmed'].str.len() > 0].reset_index(drop=True)

        print(f"Textes valides: {len(df_processed):,}")

        # Création des splits
        train_idx, temp_idx = train_test_split(
            range(len(df_processed)), test_size=0.3, random_state=42,
            stratify=df_processed['sentiment']
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42,
            stratify=df_processed.iloc[temp_idx]['sentiment']
        )

        X_train = df_processed.iloc[train_idx]['text_stemmed']
        X_val = df_processed.iloc[val_idx]['text_stemmed']
        X_test = df_processed.iloc[test_idx]['text_stemmed']
        y_train = df_processed.iloc[train_idx]['sentiment'].values
        y_val = df_processed.iloc[val_idx]['sentiment'].values
        y_test = df_processed.iloc[test_idx]['sentiment'].values

        print(f"Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

        # 3. Random Search
        print(f"\nÉtape 3/3 - Random Search ({n_runs} runs)...\n")

        results = []
        best_f1 = 0
        best_params = None

        for run_idx, hyperparams in enumerate(param_list, 1):
            print(f"\n{'='*70}")
            print(f"RUN {run_idx}/{n_runs}")
            print(f"{'='*70}")
            print(f"Hyperparamètres:")
            for key, value in hyperparams.items():
                print(f"  {key}: {value}")

            run_name = f"run_{run_idx:02d}_v{hyperparams['vector_size']}_u{hyperparams['lstm_units']}"

            with mlflow.start_run(run_name=run_name):
                try:
                    # Entraîner avec ces hyperparamètres
                    model, metrics, history, w2v = train_with_hyperparams(
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        hyperparams, technique
                    )

                    # Logger les hyperparamètres
                    mlflow.log_params(hyperparams)
                    mlflow.log_param('sample_size', len(df_processed))
                    mlflow.log_param('vocab_size', metrics['vocab_size'])

                    # Logger les métriques
                    metrics_to_log = {k: v for k, v in metrics.items()
                                     if k not in ['confusion_matrix', 'classification_report', 'model_name']}
                    mlflow.log_metrics(metrics_to_log)

                    # Logger les courbes
                    log_training_history(history.history, model_name=run_name)

                    # Sauvegarder le modèle pyfunc si c'est le meilleur
                    if metrics['f1_score'] > best_f1:
                        best_f1 = metrics['f1_score']
                        best_params = hyperparams.copy()

                        # Sauvegarder le modèle pyfunc
                        import tempfile
                        with tempfile.TemporaryDirectory() as tmpdir:
                            keras_model_path = os.path.join(tmpdir, "keras_model.keras")
                            model.save(keras_model_path)

                            w2v_path = os.path.join(tmpdir, "word2vec_embedding.pkl")
                            with open(w2v_path, 'wb') as f:
                                pickle.dump(w2v, f)

                            technique_path = os.path.join(tmpdir, "technique.txt")
                            with open(technique_path, 'w') as f:
                                f.write(technique)

                            artifacts = {
                                "keras_model": keras_model_path,
                                "word2vec_embedding": w2v_path,
                                "technique": technique_path
                            }

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

                            mlflow.pyfunc.log_model(
                                artifact_path="model",
                                python_model=Word2VecSentimentModel(),
                                artifacts=artifacts,
                                conda_env=conda_env
                            )

                        mlflow.set_tag("best_model", "true")

                    # Tags
                    mlflow.set_tag("optimization_run", f"{run_idx}/{n_runs}")

                    # Afficher résultats
                    print(f"\nRésultats:")
                    print(f"  F1-Score: {metrics['f1_score']:.4f} {'🎯 NEW BEST!' if metrics['f1_score'] == best_f1 else ''}")
                    print(f"  Accuracy: {metrics['accuracy']:.4f}")
                    print(f"  AUC: {metrics.get('auc_score', 0):.4f}")
                    print(f"  Training time: {metrics['training_time']:.1f}s")
                    print(f"  Epochs: {metrics['epochs_trained']}")

                    # Stocker pour le rapport final
                    results.append({
                        'run': run_idx,
                        'f1_score': metrics['f1_score'],
                        'accuracy': metrics['accuracy'],
                        'auc_score': metrics.get('auc_score', 0),
                        **hyperparams
                    })

                except Exception as e:
                    print(f"✗ Erreur lors du run {run_idx}: {e}")
                    import traceback
                    traceback.print_exc()

                    # Logger l'erreur dans MLflow
                    mlflow.set_tag("status", "failed")
                    mlflow.set_tag("error", str(e))

                    # Sauvegarder résultats intermédiaires toutes les 5 runs
                    if results and run_idx % 5 == 0:
                        interim_path = f"reports/hyperopt_interim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        os.makedirs(os.path.dirname(interim_path), exist_ok=True)
                        pd.DataFrame(results).to_csv(interim_path, index=False)
                        print(f"  Sauvegarde intermédiaire: {interim_path}")

                    continue

        # 4. Rapport final
        print(f"\n{'='*70}")
        print("RAPPORT FINAL - OPTIMISATION TERMINÉE")
        print(f"{'='*70}\n")

        # Vérifier qu'au moins un run a réussi
        if not results:
            print("❌ ERREUR : Aucun run n'a réussi. Vérifiez les logs ci-dessus.")
            sys.exit(1)

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('f1_score', ascending=False)

        print("Top 5 configurations:")
        print(df_results.head(5).to_string(index=False, float_format='%.4f'))

        print(f"\n🎯 MEILLEURE CONFIGURATION:")
        print(f"  F1-Score: {best_f1:.4f}")
        if best_f1 >= 0.80:
            print(f"  ✅ OBJECTIF ATTEINT (F1 ≥ 0.80) !")
        else:
            print(f"  ⚠️  Objectif non atteint (delta: {0.80 - best_f1:.4f})")

        print(f"\nHyperparamètres:")
        if best_params:
            for key, value in best_params.items():
                print(f"  {key}: {value}")
        else:
            print("  Aucun meilleur modèle identifié")

        # Sauvegarder le rapport
        report_path = f"reports/hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        df_results.to_csv(report_path, index=False)
        print(f"\nRapport sauvegardé: {report_path}")

        total_time = time.time() - start_time
        print(f"\nTemps total: {total_time/3600:.1f}h ({total_time/60:.1f} min)")
        print(f"Temps moyen par run: {total_time/n_runs/60:.1f} min")

        print(f"\nMLflow UI: http://localhost:5001")

    except Exception as e:
        print(f"\nErreur: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()