#!/usr/bin/env python3
"""
Utilitaires partagés pour les scripts d'entraînement.
Mutualise le code commun entre train_simple_model.py, train_word2vec_model.py,
train_fasttext_model.py, train_use_model.py, et train_bert_model.py.
"""

import os
import sys
import pandas as pd
import mlflow
from datetime import datetime
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.text_cleaner import TextCleaner, load_sentiment140_data


def load_and_preprocess_data(
    data_path: str,
    sample_size: Optional[int] = None,
    handle_negations: bool = True,
    handle_emotions: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Charge et prétraite les données du dataset Sentiment140.

    Args:
        data_path: Chemin vers le fichier CSV
        sample_size: Taille de l'échantillon (None = dataset complet)
        handle_negations: Activer la gestion des négations
        handle_emotions: Activer la préservation des émoticons

    Returns:
        Tuple (df_processed, preprocessing_info)
    """
    print("Chargement des données...")
    df = load_sentiment140_data(data_path, sample_size=sample_size)
    print(f"Dataset chargé: {len(df):,} échantillons")
    print(f"Distribution: {df['sentiment'].value_counts().to_dict()}")

    print("\nPrétraitement des données...")
    cleaner = TextCleaner()

    df_processed = df.copy()

    # Stemming
    print("  Stemming...")
    df_processed['text_stemmed'] = cleaner.preprocess_with_techniques(
        df['text'].tolist(),
        technique='stemming',
        handle_negations=handle_negations,
        handle_emotions=handle_emotions
    )
    df_processed = df_processed[df_processed['text_stemmed'].str.len() > 0].reset_index(drop=True)

    # Lemmatization
    print("  Lemmatization...")
    df_processed['text_lemmatized'] = cleaner.preprocess_with_techniques(
        df_processed['text'].tolist(),
        technique='lemmatization',
        handle_negations=handle_negations,
        handle_emotions=handle_emotions
    )
    df_processed = df_processed[df_processed['text_lemmatized'].str.len() > 0].reset_index(drop=True)

    print(f"Textes valides: {len(df_processed):,}")

    preprocessing_info = {
        'initial_size': len(df),
        'final_size': len(df_processed),
        'handle_negations': handle_negations,
        'handle_emotions': handle_emotions
    }

    return df_processed, preprocessing_info


def create_train_val_test_splits(
    df: pd.DataFrame,
    test_size: float = 0.3,
    val_ratio: float = 0.5,
    random_state: int = 42
) -> Dict[str, List[int]]:
    """
    Crée des splits train/validation/test stratifiés.

    Args:
        df: DataFrame avec colonne 'sentiment'
        test_size: Proportion de données pour validation + test (défaut: 0.3)
        val_ratio: Ratio validation/test dans test_size (défaut: 0.5, soit 15%/15%)
        random_state: Seed pour reproductibilité

    Returns:
        Dict avec clés 'train_idx', 'val_idx', 'test_idx'
    """
    print(f"  Création des splits train/val/test ({int((1-test_size)*100)}/{int(test_size*val_ratio*100)}/{int(test_size*(1-val_ratio)*100)})...")

    # Split initial train/temp (70/30 par défaut)
    train_idx, temp_idx = train_test_split(
        range(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=df['sentiment']
    )

    # Split temp en validation/test (50/50 = 15% chacun du total)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1-val_ratio,
        random_state=random_state,
        stratify=df.iloc[temp_idx]['sentiment']
    )

    splits_data = {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }

    print(f"  Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    return splits_data


def select_techniques_to_run(technique: str) -> List[Tuple[str, str]]:
    """
    Sélectionne les techniques de prétraitement à exécuter.

    Args:
        technique: 'stemming', 'lemmatization', ou 'both'

    Returns:
        Liste de tuples (technique_name, text_column)
    """
    if technique == 'both':
        return [('stemming', 'text_stemmed'), ('lemmatization', 'text_lemmatized')]
    elif technique == 'stemming':
        return [('stemming', 'text_stemmed')]
    else:  # lemmatization
        return [('lemmatization', 'text_lemmatized')]


def prepare_data_for_training(
    df_processed: pd.DataFrame,
    splits_data: Dict[str, List[int]],
    text_column: str
) -> Tuple[pd.Series, pd.Series, pd.Series, Any, Any, Any]:
    """
    Prépare les données (X, y) pour l'entraînement à partir des splits.

    Args:
        df_processed: DataFrame prétraité
        splits_data: Dict avec les indices des splits
        text_column: Nom de la colonne de texte ('text_stemmed' ou 'text_lemmatized')

    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train = df_processed.iloc[splits_data['train_idx']][text_column]
    X_val = df_processed.iloc[splits_data['val_idx']][text_column]
    X_test = df_processed.iloc[splits_data['test_idx']][text_column]
    y_train = df_processed.iloc[splits_data['train_idx']]['sentiment'].values
    y_val = df_processed.iloc[splits_data['val_idx']]['sentiment'].values
    y_test = df_processed.iloc[splits_data['test_idx']]['sentiment'].values

    return X_train, X_val, X_test, y_train, y_val, y_test


def filter_metrics_for_mlflow(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filtre les métriques pour ne garder que celles loggables dans MLflow.

    Args:
        metrics: Dict de métriques brutes

    Returns:
        Dict de métriques filtrées (sans confusion_matrix, classification_report, etc.)
    """
    return {k: v for k, v in metrics.items()
            if k not in ['confusion_matrix', 'classification_report', 'model_name']}


def log_common_mlflow_info(
    description: Optional[str],
    model_type: str,
    additional_tags: Optional[Dict[str, str]] = None
) -> None:
    """
    Log les informations communes dans MLflow (tags, description).

    Args:
        description: Description optionnelle de l'expérimentation
        model_type: Type du modèle (ex: 'word2vec_neural', 'bert_transformer')
        additional_tags: Tags supplémentaires optionnels
    """
    if description:
        mlflow.set_tag("description", description)
    mlflow.set_tag("model_type", model_type)

    if additional_tags:
        for key, value in additional_tags.items():
            mlflow.set_tag(key, value)


def create_comparison_dataframe(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Crée un DataFrame comparatif des résultats de plusieurs techniques.

    Args:
        results: Dict avec structure {technique_name: {'metrics': {...}, ...}}

    Returns:
        DataFrame avec comparaison des métriques
    """
    comparison_data = []
    for tech_name, res in results.items():
        m = res['metrics']
        row = {
            'Technique': tech_name.capitalize(),
            'Accuracy': m['accuracy'],
            'F1-Score': m['f1_score'],
            'AUC': m.get('auc_score', 0),
            'Time (s)': m['training_time']
        }
        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def find_best_technique(results: Dict[str, Dict[str, Any]], metric: str = 'f1_score') -> Tuple[str, Dict[str, Any]]:
    """
    Trouve la meilleure technique selon une métrique donnée.

    Args:
        results: Dict avec structure {technique_name: {'metrics': {...}, ...}}
        metric: Métrique à optimiser (défaut: 'f1_score')

    Returns:
        Tuple (best_technique_name, best_metrics)
    """
    best_technique = max(results.keys(), key=lambda k: results[k]['metrics'][metric])
    best_metrics = results[best_technique]['metrics']

    return best_technique, best_metrics


def generate_comparison_report(
    model_name: str,
    architecture: str,
    df_comparison: pd.DataFrame,
    best_technique: str,
    best_metrics: Dict[str, Any],
    dataset_size: int,
    additional_config: Optional[Dict[str, Any]] = None,
    advantages: Optional[str] = None,
    conclusion: Optional[str] = None
) -> str:
    """
    Génère un rapport de comparaison formaté.

    Args:
        model_name: Nom du modèle (ex: 'WORD2VEC', 'FASTTEXT', 'USE', 'BERT')
        architecture: Description de l'architecture
        df_comparison: DataFrame comparatif
        best_technique: Nom de la meilleure technique
        best_metrics: Métriques de la meilleure technique
        dataset_size: Taille du dataset utilisé
        additional_config: Configuration supplémentaire optionnelle
        advantages: Description des avantages du modèle
        conclusion: Conclusion personnalisée

    Returns:
        Rapport formaté en string
    """
    config_lines = [f"- Architecture: {architecture}", f"- Dataset: {dataset_size} échantillons"]

    if additional_config:
        for key, value in additional_config.items():
            config_lines.append(f"- {key}: {value}")

    config_section = "\n".join(config_lines)

    advantages_section = ""
    if advantages:
        advantages_section = f"\n\nAVANTAGE {model_name}:\n{advantages}"

    default_conclusion = f"Ce modèle utilise {model_name} et servira de référence pour comparer avec d'autres techniques d'embedding."
    conclusion_text = conclusion if conclusion else default_conclusion

    report = f"""RAPPORT DE COMPARAISON - {model_name}

OBJECTIF:
Comparer les techniques de prétraitement pour le modèle {model_name}

CONFIGURATION:
{config_section}{advantages_section}

RÉSULTATS COMPARATIFS:
{df_comparison.to_string(index=False, float_format='%.4f')}

MODÈLE RETENU: {best_technique.capitalize()}
- Accuracy: {best_metrics['accuracy']:.4f}
- F1-Score: {best_metrics['f1_score']:.4f}
- Précision: {best_metrics['precision']:.4f}
- Rappel: {best_metrics['recall']:.4f}
- Temps d'entraînement: {best_metrics['training_time']:.2f}s

CONCLUSION:
{conclusion_text}
"""

    return report


def save_report(
    report: str,
    report_name: str,
    reports_dir: str = "reports"
) -> str:
    """
    Sauvegarde un rapport dans un fichier.

    Args:
        report: Contenu du rapport
        report_name: Nom de base du rapport (sans extension)
        reports_dir: Répertoire de destination

    Returns:
        Chemin complet du rapport sauvegardé
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(reports_dir, f"{report_name}_{timestamp}.txt")

    os.makedirs(reports_dir, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report_path


def log_summary_run(
    run_name: str,
    technique_used: str,
    best_technique: str,
    best_metrics: Dict[str, Any],
    total_samples: int,
    techniques_tested: int,
    additional_params: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None
) -> None:
    """
    Crée une run summary dans MLflow avec les résultats finaux.

    Args:
        run_name: Nom de la run summary
        technique_used: Technique demandée ('stemming', 'lemmatization', 'both')
        best_technique: Meilleure technique trouvée
        best_metrics: Métriques de la meilleure technique
        total_samples: Nombre total d'échantillons
        techniques_tested: Nombre de techniques testées
        additional_params: Paramètres supplémentaires à logger
        description: Description optionnelle
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("summary_type", "final_report")
        if description:
            mlflow.set_tag("description", description)

        # Log paramètres de base
        params = {
            'technique_used': technique_used,
            'best_technique': best_technique,
            'total_samples': total_samples,
            'techniques_tested': techniques_tested
        }

        # Ajouter paramètres supplémentaires
        if additional_params:
            params.update(additional_params)

        mlflow.log_params(params)

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


def print_training_header(
    model_name: str,
    technique: str,
    description: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Affiche un header formaté pour le début de l'entraînement.

    Args:
        model_name: Nom du modèle
        technique: Technique de prétraitement
        description: Description de l'expérimentation
        additional_info: Informations supplémentaires à afficher
    """
    separator = "=" * 60
    print(f"{separator}")
    print(f"ENTRAÎNEMENT {model_name}")
    print(f"{separator}\n")
    print(f"Technique: {technique}")
    print(f"Description: {description}")

    if additional_info:
        for key, value in additional_info.items():
            print(f"{key}: {value}")

    print()


def print_results_summary(
    technique_name: str,
    metrics: Dict[str, Any],
    additional_metrics: Optional[List[str]] = None
) -> None:
    """
    Affiche un résumé formaté des résultats.

    Args:
        technique_name: Nom de la technique
        metrics: Dict des métriques
        additional_metrics: Liste de clés de métriques supplémentaires à afficher
    """
    print(f"\nRésultats {technique_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")

    if 'auc_score' in metrics:
        print(f"  AUC: {metrics.get('auc_score', 0):.4f}")

    print(f"  Training time: {metrics['training_time']:.1f}s")

    if additional_metrics:
        for metric_key in additional_metrics:
            if metric_key in metrics:
                print(f"  {metric_key}: {metrics[metric_key]}")


def print_comparison(df_comparison: pd.DataFrame, best_technique: str) -> None:
    """
    Affiche le tableau comparatif et la meilleure technique.

    Args:
        df_comparison: DataFrame comparatif
        best_technique: Nom de la meilleure technique
    """
    print("\nCOMPARAISON:")
    print(df_comparison.to_string(index=False, float_format='%.4f'))
    print(f"\nMeilleure technique: {best_technique.upper()}")


def print_completion_message(total_time: float, mlflow_ui_url: str = "http://localhost:5001") -> None:
    """
    Affiche le message de fin d'entraînement.

    Args:
        total_time: Temps total d'exécution en secondes
        mlflow_ui_url: URL de l'interface MLflow
    """
    print(f"\n{'=' * 60}")
    print("ENTRAÎNEMENT TERMINÉ")
    print(f"{'=' * 60}")
    print(f"Temps total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"MLflow UI: {mlflow_ui_url}")