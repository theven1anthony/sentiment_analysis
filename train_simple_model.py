#!/usr/bin/env python3
"""
Script d'entraînement et d'évaluation du modèle simple.
Démontre le pipeline complet pour le modèle de référence (baseline).

Usage:
    python train_simple_model.py --technique=stemming --description="Test baseline stemming"
    python train_simple_model.py --technique=lemmatization --description="Test baseline lemmatization"
    python train_simple_model.py --technique=both --description="Comparaison stemming vs lemmatization"
"""

import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
import click
from datetime import datetime

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.text_cleaner import TextCleaner, load_sentiment140_data, preprocess_dataset
from models.simple_model import train_simple_model
# from models.mlflow_utils import train_and_register_model, ModelManager  # TEMPORAIRE : commenté pour debug
from evaluation.metrics import ModelEvaluator, create_data_splits


def create_training_function(technique_name, text_column, df_processed, splits_data):
    """
    Crée une fonction d'entraînement compatible avec MLflow utils.
    Utilise les splits pré-calculés pour éviter les incohérences.
    """
    def train_func(**kwargs):
        # Utiliser les splits pré-calculés
        X_train = df_processed.iloc[splits_data['train_idx']][text_column]
        X_val = df_processed.iloc[splits_data['val_idx']][text_column]
        X_test = df_processed.iloc[splits_data['test_idx']][text_column]
        y_train = df_processed.iloc[splits_data['train_idx']]['sentiment']
        y_val = df_processed.iloc[splits_data['val_idx']]['sentiment']
        y_test = df_processed.iloc[splits_data['test_idx']]['sentiment']

        print(f"   - Train: {len(X_train)} échantillons")
        print(f"   - Validation: {len(X_val)} échantillons")
        print(f"   - Test: {len(X_test)} échantillons")

        # Entraînement avec format MLflow standard
        return train_simple_model(X_train, y_train, X_val, y_val, X_test, y_test)

    return train_func


@click.command()
@click.option('--technique',
              type=click.Choice(['stemming', 'lemmatization', 'both']),
              default='both',
              help='Technique de prétraitement à utiliser')
@click.option('--description',
              default='',
              help='Description personnalisée pour l\'expérimentation')
@click.option('--experiment-name',
              default='simple_models',
              help='Nom de l\'expérimentation MLflow')
@click.option('--sample-size',
              type=int,
              default=None,
              help='Taille de l\'échantillon (None = dataset complet)')
@click.option('--handle-negations',
              type=bool,
              default=True,
              help='Activer la gestion intelligente des négations')
@click.option('--handle-emotions',
              type=bool,
              default=True,
              help='Activer la préservation des émoticons')
def main(technique, description, experiment_name, sample_size, handle_negations, handle_emotions):
    """Fonction principale pour entraîner et évaluer le modèle simple."""

    print("=== ENTRAÎNEMENT DU MODÈLE SIMPLE DE SENTIMENT ANALYSIS ===\n")
    print(f"Technique: {technique}")
    print(f"Description: {description}")
    print(f"Expérimentation: {experiment_name}")

    # Configuration MLflow
    mlflow.set_experiment(experiment_name)

    try:
        # 1. Chargement des données
        print("\n=== DEBUT DE L'ENTRAINEMENT ===\n")
        print("Etape 1/6 - Chargement des données...")
        import time
        start_time = time.time()

        # Chargement du dataset Sentiment140
        try:
            print("   Chargement du fichier CSV...")
            df = load_sentiment140_data("data/training.1600000.processed.noemoticon.csv", sample_size=sample_size)
            size_info = "complet" if sample_size is None else f"({sample_size} échantillons)"
            load_time = time.time() - start_time
            print(f"   Dataset Sentiment140 {size_info} chargé: {len(df):,} échantillons en {load_time:.1f}s")
        except FileNotFoundError:
            print("   Erreur: Dataset Sentiment140 non trouvé")
            raise

        print(f"   Distribution des sentiments: {df['sentiment'].value_counts().to_dict()}")

        # 2. Prétraitement des données
        print("\nEtape 2/6 - Prétraitement des données...")
        prep_start = time.time()

        # Application directe des techniques de prétraitement (CE1: au moins 2 techniques)
        print("   Initialisation du nettoyeur de texte...")
        cleaner = TextCleaner()

        print("   Application des techniques de prétraitement avancées:")
        print(f"   Options: handle_negations={handle_negations}, handle_emotions={handle_emotions}")

        stem_start = time.time()
        print("     Stemming (avec gestion négations et émotions)...")
        df_processed = df.copy()
        df_processed['text_stemmed'] = cleaner.preprocess_with_techniques(
            df['text'].tolist(), technique='stemming',
            handle_negations=handle_negations, handle_emotions=handle_emotions
        )
        # Filtrage des tweets vides après preprocessing
        df_processed = df_processed[df_processed['text_stemmed'].str.len() > 0].reset_index(drop=True)
        stem_time = time.time() - stem_start
        print(f"     Stemming terminé en {stem_time:.1f}s - {len(df_processed):,} tweets valides")

        lemma_start = time.time()
        print("     Lemmatization (avec gestion négations et émotions)...")
        df_processed['text_lemmatized'] = cleaner.preprocess_with_techniques(
            df_processed['text'].tolist(), technique='lemmatization',
            handle_negations=handle_negations, handle_emotions=handle_emotions
        )
        # Filtrage des tweets vides après preprocessing
        df_processed = df_processed[df_processed['text_lemmatized'].str.len() > 0].reset_index(drop=True)
        lemma_time = time.time() - lemma_start
        print(f"     Lemmatization terminé en {lemma_time:.1f}s - {len(df_processed):,} tweets valides")

        # Création des splits unifiés pour éviter les incohérences
        print("   Création des splits train/validation/test...")
        from sklearn.model_selection import train_test_split

        # Split initial train/temp (70/30)
        train_idx, temp_idx = train_test_split(
            range(len(df_processed)),
            test_size=0.3,
            random_state=42,
            stratify=df_processed['sentiment']
        )

        # Split temp en validation/test (50/50 = 15% chacun du total)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=42,
            stratify=df_processed.iloc[temp_idx]['sentiment']
        )

        # Stockage des indices pour réutilisation
        splits_data = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }

        print(f"   Splits créés: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

        total_prep_time = time.time() - prep_start
        print(f"   Prétraitement total: {total_prep_time:.1f}s")

        # 3. Entraînement selon la technique choisie
        print(f"\nEtape 3/6 - Entraînement avec technique: {technique}...")
        training_start = time.time()

        techniques_results = {}
        total_techniques = 2 if technique == 'both' else 1
        current_technique = 0

        # Déterminer les techniques à exécuter
        if technique == 'both':
            techniques_to_run = [('stemming', 'text_stemmed'), ('lemmatization', 'text_lemmatized')]
        elif technique == 'stemming':
            techniques_to_run = [('stemming', 'text_stemmed')]
        else:  # lemmatization
            techniques_to_run = [('lemmatization', 'text_lemmatized')]

        # Entraîner les modèles avec MLflow standard
        for technique_name, text_column in techniques_to_run:
            current_technique += 1
            print(f"\n   === TECHNIQUE {current_technique}/{total_techniques}: {technique_name.upper()} ===")
            technique_start = time.time()

            model_name = f"sentiment_simple_{technique_name}"
            model_description = f"Baseline {technique_name}"
            if description:
                model_description += f" - {description}"

            print(f"   Modèle: {model_name}")
            print(f"   Description: {model_description}")

            # Créer la fonction d'entraînement
            print("   Préparation des données pour l'entraînement...")
            train_func = create_training_function(technique_name, text_column, df_processed, splits_data)

            # Entraîner avec MLflow chronométrant la durée totale
            print("   Démarrage de l'entraînement direct...")
            with mlflow.start_run(run_name=model_name, description=model_description if description else None):
                # L'entraînement se fait dans le contexte MLflow pour chronométrer
                model, metrics, artifacts = train_func()

                # Paramètres (hyperparamètres)
                mlflow.log_params({
                    "technique": technique_name,
                    "handle_negations": handle_negations,
                    "handle_emotions": handle_emotions
                })

                # Métriques (performances + statistiques)
                metrics_extended = metrics.copy()
                metrics_extended["dataset_samples"] = len(df)
                metrics_extended["sample_size_used"] = sample_size if sample_size else len(df)
                mlflow.log_metrics(metrics_extended)

                # Modèle
                mlflow.sklearn.log_model(model, "model")

                # Tags (métadonnées descriptives)
                if description:
                    mlflow.set_tag("description", description)
                mlflow.set_tag("model_description", model_description)

            print(f"   Entraînement {technique_name} terminé!")

            # Évaluer pour la comparaison (utilise les mêmes splits)
            print("   Évaluation des performances...")
            eval_start = time.time()

            # Utiliser les mêmes splits que pour l'entraînement
            X_test = df_processed.iloc[splits_data['test_idx']][text_column]
            y_test = df_processed.iloc[splits_data['test_idx']]['sentiment']

            # Prédictions pour comparaison
            print(f"   Génération des prédictions sur {len(X_test):,} échantillons de test...")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

            evaluator = ModelEvaluator()
            eval_metrics = evaluator.evaluate_model(
                y_test, y_pred, y_pred_proba,
                model_name=f"simple_{technique_name}"
            )

            eval_time = time.time() - eval_start
            technique_total_time = time.time() - technique_start

            techniques_results[technique_name] = {
                'model': model,
                'eval_metrics': eval_metrics,
                'training_time': metrics['training_time']  # Conserver le temps d'entraînement
            }

            print(f"   RESULTATS {technique_name.upper()}:")
            print(f"   Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"   F1-Score: {eval_metrics['f1_score']:.4f}")
            print(f"   Temps évaluation: {eval_time:.1f}s")
            print(f"   Temps total technique: {technique_total_time:.1f}s")

        training_total_time = time.time() - training_start
        print(f"\nEntraînement terminé en {training_total_time:.1f}s")

        # 4. Comparaison et sélection (si plusieurs techniques)
        if len(techniques_results) > 1:
            print("\nEtape 4/6 - Comparaison des techniques de prétraitement...")

            # Créer le tableau de comparaison
            comparison_data = []
            for technique_name, results in techniques_results.items():
                metrics = results['eval_metrics']
                comparison_data.append({
                    'Technique': technique_name.capitalize(),
                    'Accuracy': metrics['accuracy'],
                    'F1-Score': metrics['f1_score'],
                    'Précision': metrics['precision'],
                    'Rappel': metrics['recall'],
                    'Temps (s)': results['training_time']
                })

            df_comparison = pd.DataFrame(comparison_data)
            print("\n   COMPARAISON DES TECHNIQUES:")
            print(df_comparison.to_string(index=False, float_format='%.4f'))

            # Sélectionner le meilleur modèle (basé sur F1-score)
            best_technique = max(techniques_results.keys(),
                               key=lambda k: techniques_results[k]['eval_metrics']['f1_score'])
            best_results = techniques_results[best_technique]

            print(f"\n   MEILLEURE TECHNIQUE: {best_technique.upper()}")
            print(f"   - F1-Score: {best_results['eval_metrics']['f1_score']:.4f}")
            print(f"   - Accuracy: {best_results['eval_metrics']['accuracy']:.4f}")
        else:
            # Une seule technique
            technique_name = list(techniques_results.keys())[0]
            best_technique = technique_name
            best_results = techniques_results[technique_name]
            df_comparison = None

        # 5. Génération du rapport
        print("\nEtape 5/6 - Génération du rapport d'évaluation...")
        report_start = time.time()

        if len(techniques_results) > 1:
            report_title = "RAPPORT DE COMPARAISON - TECHNIQUES DE PRÉTRAITEMENT"
            objective = "Comparer les techniques de prétraitement stemming vs lemmatization"
            comparison_section = f"\nRÉSULTATS COMPARATIFS:\n{df_comparison.to_string(index=False, float_format='%.4f')}"
        else:
            report_title = f"RAPPORT D'ÉVALUATION - TECHNIQUE {technique.upper()}"
            objective = f"Entraînement du modèle de référence avec technique {technique}"
            comparison_section = ""

        report = f"""{report_title}

OBJECTIF:
{objective}

CONFIGURATION:
- Algorithme: Logistic Regression
- Vectorisation: TF-IDF
- Dataset: {len(df)} échantillons Sentiment140
- Description: {description if description else 'Aucune'}{comparison_section}

MODÈLE RETENU: {best_technique.capitalize()}
- Accuracy: {best_results['eval_metrics']['accuracy']:.4f}
- F1-Score: {best_results['eval_metrics']['f1_score']:.4f}
- Précision: {best_results['eval_metrics']['precision']:.4f}
- Rappel: {best_results['eval_metrics']['recall']:.4f}
- Temps d'entraînement: {best_results['training_time']:.2f}s

CONCLUSION:
Ce modèle sert de référence (baseline) pour comparer les modèles plus complexes.
"""

        # Sauvegarder le rapport
        report_name = "comparison" if len(techniques_results) > 1 else technique
        report_path = f"reports/{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        report_time = time.time() - report_start
        print(f"\nRapport sauvegardé: {report_path} (en {report_time:.1f}s)")

        # Log du rapport dans MLflow
        print("\nEtape 6/6 - Sauvegarde dans MLflow...")
        mlflow_start = time.time()

        with mlflow.start_run(run_name=f"{technique}_summary"):
            mlflow.set_tag("summary_type", "final_report")
            if description:
                mlflow.set_tag("description", description)

            mlflow.log_params({
                'technique_used': technique,
                'best_technique': best_technique,
                'total_samples': len(df),
                'techniques_tested': len(techniques_results)
            })

            mlflow.log_metrics({
                'best_accuracy': best_results['eval_metrics']['accuracy'],
                'best_f1_score': best_results['eval_metrics']['f1_score']
            })

            mlflow.log_artifact(report_path, "reports")

        mlflow_time = time.time() - mlflow_start
        total_time = time.time() - start_time

        print(f"\nSauvegarde MLflow terminée en {mlflow_time:.1f}s")
        print("\n=== ENTRAINEMENT TERMINE AVEC SUCCES! ===")
        print(f"Temps total d'exécution: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Meilleur modèle: {best_technique.upper()}")
        print(f"Consultez les résultats sur MLflow UI: http://localhost:5001")

    except Exception as e:
        print(f"\n Erreur lors de l'entraînement: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()