"""
Utilitaires MLflow standard pour la gestion des modèles.
Logique réutilisable pour tous les types de modèles (simple, TensorFlow, BERT).
"""

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch
from typing import Optional, Dict, Any
import warnings


class ModelManager:
    """
    Gestionnaire de modèles MLflow standard.
    Utilise les fonctionnalités natives MLflow Model Registry.
    """

    @staticmethod
    def model_exists(model_name: str, stage: str = "None") -> bool:
        """
        Vérifie si un modèle existe dans le Model Registry.

        Args:
            model_name: Nom du modèle dans le registry
            stage: Stage du modèle (None, Staging, Production)

        Returns:
            True si le modèle existe
        """
        try:
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(model_name, stages=[stage])
            return len(model_version) > 0
        except Exception:
            return False

    @staticmethod
    def load_model(model_name: str, stage: str = "None"):
        """
        Charge un modèle depuis le Model Registry.

        Args:
            model_name: Nom du modèle
            stage: Stage du modèle (None, Staging, Production)

        Returns:
            Modèle chargé ou None si inexistant
        """
        try:
            model_uri = f"models:/{model_name}/{stage}"
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            print(f"Modèle {model_name} non trouvé: {e}")
            return None

    @staticmethod
    def register_sklearn_model(model, model_name: str, description: str = "") -> str:
        """
        Enregistre un modèle scikit-learn dans MLflow.

        Args:
            model: Modèle scikit-learn entraîné
            model_name: Nom pour le registry
            description: Description du modèle

        Returns:
            URI du modèle enregistré
        """
        with mlflow.start_run() as run:
            # Log du modèle
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )

            # Ajout de métadonnées
            if description:
                mlflow.set_tag("model.description", description)

            return f"runs:/{run.info.run_id}/model"

    @staticmethod
    def register_tensorflow_model(model, model_name: str, description: str = "") -> str:
        """
        Enregistre un modèle TensorFlow dans MLflow.

        Args:
            model: Modèle TensorFlow/Keras entraîné
            model_name: Nom pour le registry
            description: Description du modèle

        Returns:
            URI du modèle enregistré
        """
        with mlflow.start_run() as run:
            mlflow.tensorflow.log_model(
                tf_saved_model_dir=None,
                tf_meta_graph_tags=None,
                tf_signature_def_key=None,
                artifact_path="model",
                registered_model_name=model_name
            )

            if description:
                mlflow.set_tag("model.description", description)

            return f"runs:/{run.info.run_id}/model"

    @staticmethod
    def get_or_create_model(
        model_name: str,
        train_func: callable,
        force_retrain: bool = False,
        stage: str = "None",
        **train_kwargs
    ):
        """
        Pattern standard MLflow : charge le modèle existant ou l'entraîne.

        Args:
            model_name: Nom du modèle dans le registry
            train_func: Fonction d'entraînement qui retourne le modèle
            force_retrain: Force le réentraînement même si le modèle existe
            stage: Stage du modèle à charger
            **train_kwargs: Arguments pour la fonction d'entraînement

        Returns:
            Modèle chargé ou nouvellement entraîné
        """
        # Vérifier si le modèle existe déjà
        if not force_retrain and ModelManager.model_exists(model_name, stage):
            print(f"✓ Modèle '{model_name}' trouvé dans le registry, chargement...")
            return ModelManager.load_model(model_name, stage)

        # Entraîner le modèle
        print(f"⚡ Entraînement du modèle '{model_name}'...")
        model = train_func(**train_kwargs)

        # Enregistrer dans MLflow (fait dans la fonction d'entraînement)
        print(f"✓ Modèle '{model_name}' entraîné et enregistré")
        return model

    @staticmethod
    def promote_model(model_name: str, version: int, stage: str):
        """
        Promote un modèle vers un stage (Staging -> Production).

        Args:
            model_name: Nom du modèle
            version: Version à promouvoir
            stage: Stage cible (Staging, Production)
        """
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"✓ Modèle {model_name} v{version} promu vers {stage}")

    @staticmethod
    def list_models() -> Dict[str, Any]:
        """
        Liste tous les modèles enregistrés.

        Returns:
            Dictionnaire avec les informations des modèles
        """
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()

        models_info = {}
        for model in models:
            latest_versions = client.get_latest_versions(model.name, stages=["None", "Staging", "Production"])
            models_info[model.name] = {
                "description": model.description,
                "versions": {v.current_stage: v.version for v in latest_versions}
            }

        return models_info

    @staticmethod
    def compare_models(model_names: list, metric: str = "f1_score") -> Dict[str, float]:
        """
        Compare les performances de plusieurs modèles.

        Args:
            model_names: Liste des noms de modèles à comparer
            metric: Métrique de comparaison

        Returns:
            Dictionnaire {model_name: metric_value}
        """
        client = mlflow.tracking.MlflowClient()
        comparison = {}

        for model_name in model_names:
            try:
                # Récupérer la dernière version
                versions = client.get_latest_versions(model_name, stages=["None"])
                if versions:
                    run_id = versions[0].run_id
                    run = client.get_run(run_id)
                    comparison[model_name] = run.data.metrics.get(metric, 0.0)
            except Exception as e:
                print(f"Erreur pour {model_name}: {e}")
                comparison[model_name] = 0.0

        return comparison


def setup_mlflow_experiment(experiment_name: str) -> str:
    """
    Configure l'expérimentation MLflow de manière standard.

    Args:
        experiment_name: Nom de l'expérimentation

    Returns:
        ID de l'expérimentation
    """
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        # L'expérimentation existe déjà
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_model_metrics(metrics: Dict[str, float], tags: Dict[str, str] = None):
    """
    Log les métriques et tags de manière standardisée.

    Args:
        metrics: Dictionnaire des métriques
        tags: Dictionnaire des tags optionnels
    """
    # Log des métriques
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)

    # Log des tags
    if tags:
        for tag_name, tag_value in tags.items():
            mlflow.set_tag(tag_name, tag_value)


# Pattern d'utilisation recommandé pour tous les modèles
def train_and_register_model(
    model_name: str,
    train_function: callable,
    model_type: str,  # 'sklearn', 'tensorflow', 'pytorch'
    experiment_name: str,
    description: str = "",
    force_retrain: bool = False,
    **train_kwargs
):
    """
    Pattern standard pour entraîner et enregistrer un modèle.

    Args:
        model_name: Nom unique du modèle
        train_function: Fonction qui retourne (model, metrics, artifacts)
        model_type: Type de modèle pour le bon logger MLflow
        experiment_name: Nom de l'expérimentation
        description: Description du modèle
        force_retrain: Force le réentraînement
        **train_kwargs: Arguments pour l'entraînement

    Returns:
        Modèle entraîné/chargé
    """
    # Setup expérimentation
    setup_mlflow_experiment(experiment_name)

    # Vérifier si le modèle existe
    if not force_retrain and ModelManager.model_exists(model_name):
        print(f"✓ Modèle existant '{model_name}' chargé")
        return ModelManager.load_model(model_name)

    # Entraîner le modèle
    with mlflow.start_run(run_name=model_name):
        print(f"⚡ Entraînement de '{model_name}'...")

        # Exécuter l'entraînement
        result = train_function(**train_kwargs)

        if isinstance(result, tuple) and len(result) == 3:
            model, metrics, artifacts = result
        else:
            model = result
            metrics = {}
            artifacts = {}

        # Log métriques
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Log artifacts
        for artifact_name, artifact_path in artifacts.items():
            mlflow.log_artifact(artifact_path, artifact_name)

        # Log tags
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("description", description)

        # Enregistrer le modèle selon son type
        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        elif model_type == "tensorflow":
            mlflow.tensorflow.log_model(model, "model", registered_model_name=model_name)
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(model, "model", registered_model_name=model_name)

        print(f"✓ Modèle '{model_name}' enregistré dans MLflow")

    return ModelManager.load_model(model_name)