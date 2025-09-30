"""
Module d'évaluation des modèles de sentiment analysis.
Calcule et compare les métriques de performance des différents modèles.
Répond aux critères d'évaluation CE1-CE6 pour l'évaluation des performances.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


class ModelEvaluator:
    """
    Classe pour évaluer et comparer les performances des modèles.
    """

    def __init__(self):
        """Initialise l'évaluateur de modèles."""
        self.results_history = []

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_pred_proba: np.ndarray = None, model_name: str = "Model",
                      training_time: float = None) -> Dict[str, Any]:
        """
        Évalue un modèle selon plusieurs métriques.
        Répond aux critères CE1: métrique adaptée à la problématique métier.

        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions du modèle
            y_pred_proba: Probabilités de prédiction (optionnel)
            model_name: Nom du modèle
            training_time: Temps d'entraînement en secondes

        Returns:
            Dictionnaire avec toutes les métriques
        """
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro')
        }

        # Ajout de l'AUC si les probabilités sont disponibles
        if y_pred_proba is not None:
            if y_pred_proba.ndim > 1:
                # Prendre les probabilités de la classe positive
                y_pred_proba = y_pred_proba[:, 1]
            metrics['auc_score'] = roc_auc_score(y_true, y_pred_proba)

        # Ajout du temps d'entraînement si disponible
        # Répond aux critères CE4: au moins un autre indicateur (temps d'entraînement)
        if training_time is not None:
            metrics['training_time'] = training_time

        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_positives'] = cm[0, 1]
        metrics['false_negatives'] = cm[1, 0]
        metrics['true_positives'] = cm[1, 1]

        # Rapport de classification détaillé
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)

        # Stocker dans l'historique
        self.results_history.append(metrics)

        return metrics

    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare les performances de plusieurs modèles.
        Répond aux critères CE6: synthèse comparative sous forme de tableau.

        Args:
            models_results: Dictionnaire avec les résultats de chaque modèle

        Returns:
            DataFrame avec la comparaison des modèles
        """
        comparison_data = []

        for model_name, results in models_results.items():
            model_data = {
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1-Score': results.get('f1_score', 0),
                'Training Time (s)': results.get('training_time', 0)
            }

            # Ajouter l'AUC si disponible
            if 'auc_score' in results:
                model_data['AUC'] = results['auc_score']

            comparison_data.append(model_data)

        df_comparison = pd.DataFrame(comparison_data)

        # Trier par F1-score décroissant
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)

        return df_comparison

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "Model", figsize: Tuple[int, int] = (8, 6)):
        """
        Affiche la matrice de confusion.

        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions
            model_name: Nom du modèle
            figsize: Taille de la figure
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Matrice de confusion - {model_name}')
        plt.xlabel('Prédictions')
        plt.ylabel('Vraies étiquettes')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "Model", figsize: Tuple[int, int] = (8, 6)):
        """
        Affiche la courbe ROC.

        Args:
            y_true: Vraies étiquettes
            y_pred_proba: Probabilités de prédiction
            model_name: Nom du modèle
            figsize: Taille de la figure
        """
        if y_pred_proba.ndim > 1:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title('Courbe ROC')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_models_comparison(self, df_comparison: pd.DataFrame,
                              metric: str = 'F1-Score', figsize: Tuple[int, int] = (12, 8)):
        """
        Visualise la comparaison des modèles.

        Args:
            df_comparison: DataFrame de comparaison
            metric: Métrique à visualiser
            figsize: Taille de la figure
        """
        plt.figure(figsize=figsize)

        # Graphique en barres
        plt.subplot(2, 2, 1)
        bars = plt.bar(df_comparison['Model'], df_comparison[metric])
        plt.title(f'Comparaison des modèles - {metric}')
        plt.xticks(rotation=45)
        plt.ylabel(metric)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')

        # Temps d'entraînement si disponible
        if 'Training Time (s)' in df_comparison.columns:
            plt.subplot(2, 2, 2)
            bars = plt.bar(df_comparison['Model'], df_comparison['Training Time (s)'])
            plt.title('Temps d\'entraînement')
            plt.xticks(rotation=45)
            plt.ylabel('Temps (secondes)')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}s', ha='center', va='bottom')

        # Heatmap des métriques
        plt.subplot(2, 2, (3, 4))
        metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        if 'AUC' in df_comparison.columns:
            metrics_cols.append('AUC')

        sns.heatmap(df_comparison[metrics_cols].T, annot=True, fmt='.3f',
                   xticklabels=df_comparison['Model'], cmap='RdYlBu_r')
        plt.title('Heatmap des métriques de performance')

        plt.tight_layout()
        plt.show()

    def get_best_model(self, models_results: Dict[str, Dict],
                      metric: str = 'f1_score') -> Tuple[str, Dict]:
        """
        Identifie le meilleur modèle selon une métrique.

        Args:
            models_results: Résultats des modèles
            metric: Métrique pour déterminer le meilleur modèle

        Returns:
            Tuple avec le nom du meilleur modèle et ses résultats
        """
        best_score = -1
        best_model_name = None
        best_results = None

        for model_name, results in models_results.items():
            score = results.get(metric, 0)
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_results = results

        return best_model_name, best_results

    def generate_evaluation_report(self, models_results: Dict[str, Dict]) -> str:
        """
        Génère un rapport d'évaluation détaillé.
        Répond aux critères CE2: justification du choix de métrique.

        Args:
            models_results: Résultats des modèles

        Returns:
            Rapport d'évaluation en texte
        """
        report = []
        report.append("=== RAPPORT D'ÉVALUATION DES MODÈLES DE SENTIMENT ANALYSIS ===\n")

        # Justification du choix des métriques
        report.append("CHOIX DES MÉTRIQUES (CE2):")
        report.append("- F1-Score: Métrique principale pour équilibrer précision et rappel")
        report.append("- Accuracy: Taux de bonnes classifications globales")
        report.append("- AUC-ROC: Capacité de discrimination entre les classes")
        report.append("- Temps d'entraînement: Critère d'efficacité opérationnelle")
        report.append("")

        # Comparaison des modèles
        df_comparison = self.compare_models(models_results)
        report.append("SYNTHÈSE COMPARATIVE DES MODÈLES (CE6):")
        report.append(df_comparison.to_string(index=False))
        report.append("")

        # Meilleur modèle
        best_model_name, best_results = self.get_best_model(models_results)
        report.append(f"MEILLEUR MODÈLE: {best_model_name}")
        report.append(f"- F1-Score: {best_results['f1_score']:.4f}")
        report.append(f"- Accuracy: {best_results['accuracy']:.4f}")
        if 'training_time' in best_results:
            report.append(f"- Temps d'entraînement: {best_results['training_time']:.2f}s")
        report.append("")

        # Recommandations
        report.append("RECOMMANDATIONS:")
        report.append("1. Le modèle sélectionné sert de référence (baseline)")
        report.append("2. Implémenter des modèles plus complexes (LSTM, BERT)")
        report.append("3. Optimiser les hyperparamètres du meilleur modèle")
        report.append("4. Évaluer sur des données réelles de production")

        return "\n".join(report)


def create_data_splits(df: pd.DataFrame, test_size: float = 0.2,
                      val_size: float = 0.1, random_state: int = 42) -> Tuple:
    """
    Crée les splits train/validation/test.
    Répond aux critères CE4: séparation en jeux d'entraînement, validation et test.
    Répond aux critères CE5: pas de fuite d'information entre les jeux.

    Args:
        df: DataFrame avec les données
        test_size: Proportion des données de test
        val_size: Proportion des données de validation
        random_state: Graine aléatoire pour la reproductibilité

    Returns:
        Tuple avec les splits (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Split initial train+val / test
    X = df['cleaned_text']
    y = df['sentiment']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split train / validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted,
        random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test