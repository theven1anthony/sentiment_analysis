"""
Utilitaires MLflow pour le tracking et la gestion des modèles
"""

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.transformers
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time
from datetime import datetime


class MLflowTracker:
    """Classe pour simplifier le tracking des expériences MLflow"""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment:
            mlflow.set_experiment(experiment_name)
        else:
            raise ValueError(f"Expérience '{experiment_name}' non trouvée")

    def start_run(self, run_name: str, tags: Dict[str, str] = None):
        """Démarre un nouveau run MLflow"""
        if tags is None:
            tags = {}

        tags.update(
            {"mlflow.runName": run_name, "project": "air_paradis_sentiment", "timestamp": datetime.now().isoformat()}
        )

        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_preprocessing_params(self, params: Dict[str, Any]):
        """Log les paramètres de preprocessing"""
        preprocessing_params = {f"preprocessing.{k}": v for k, v in params.items()}
        mlflow.log_params(preprocessing_params)

    def log_model_params(self, params: Dict[str, Any]):
        """Log les paramètres du modèle"""
        model_params = {f"model.{k}": v for k, v in params.items()}
        mlflow.log_params(model_params)

    def log_training_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log les métriques d'entraînement"""
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"train.{metric_name}", value, step=step)

    def log_validation_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log les métriques de validation"""
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"val.{metric_name}", value, step=step)

    def log_test_metrics(self, metrics: Dict[str, float]):
        """Log les métriques de test finales"""
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"test.{metric_name}", value)

    def log_performance_metrics(self, inference_time: float, model_size_mb: float):
        """Log les métriques de performance"""
        mlflow.log_metric("performance.inference_time_ms", inference_time * 1000)
        mlflow.log_metric("performance.model_size_mb", model_size_mb)

    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None):
        """Log la matrice de confusion comme artifact"""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names or ["Negative", "Positive"],
            yticklabels=class_names or ["Negative", "Positive"],
        )
        plt.title("Matrice de Confusion")
        plt.ylabel("Vraie Classe")
        plt.xlabel("Classe Prédite")

        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        mlflow.log_artifact("confusion_matrix.png", "plots")
        plt.close()

    def log_training_history(self, history: Dict[str, List[float]]):
        """Log l'historique d'entraînement"""
        for metric, values in history.items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(metric, value, step=epoch)

    def log_embeddings_info(self, embedding_type: str, vocab_size: int, embedding_dim: int, model_path: str = None):
        """Log les informations des embeddings"""
        mlflow.log_params(
            {
                "embeddings.type": embedding_type,
                "embeddings.vocab_size": vocab_size,
                "embeddings.dimension": embedding_dim,
            }
        )

        if model_path and Path(model_path).exists():
            mlflow.log_artifact(model_path, "embeddings")


class ModelRegistry:
    """Classe pour gérer le model registry"""

    def __init__(self):
        self.stages = ["None", "Staging", "Production", "Archived"]

    def register_model(self, model_uri: str, model_name: str, description: str = None) -> str:
        """Enregistre un modèle dans le registry"""
        result = mlflow.register_model(model_uri=model_uri, name=model_name, description=description)
        return result.version

    def promote_model(self, model_name: str, version: str, stage: str):
        """Promeut un modèle vers un stage donné"""
        if stage not in self.stages:
            raise ValueError(f"Stage invalide. Stages disponibles: {self.stages}")

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(name=model_name, version=version, stage=stage)

    def get_latest_version(self, model_name: str, stage: str = "Production"):
        """Récupère la dernière version d'un modèle en production"""
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        return versions[0] if versions else None

    def compare_models(self, model_name: str, metrics: List[str] = None):
        """Compare les versions d'un modèle"""
        if metrics is None:
            metrics = ["test.accuracy", "test.f1_score", "performance.inference_time_ms"]

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")

        comparison_data = []
        for version in versions:
            run = client.get_run(version.run_id)
            version_data = {"version": version.version, "stage": version.current_stage, "run_id": version.run_id[:8]}

            for metric in metrics:
                version_data[metric] = run.data.metrics.get(metric, "N/A")

            comparison_data.append(version_data)

        return pd.DataFrame(comparison_data)


def track_experiment(experiment_name: str, run_name: str):
    """Décorateur pour tracker automatiquement une expérience"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = MLflowTracker(experiment_name)
            with tracker.start_run(run_name):
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                mlflow.log_metric("execution_time_seconds", execution_time)
                return result

        return wrapper

    return decorator
