"""
Configuration et initialisation MLflow pour Air Paradis - Sentiment Analysis
"""
import mlflow


class MLflowSetup:
    """Classe pour configurer et initialiser l'environnement MLflow"""

    def __init__(self, tracking_uri: str = None):
        self.tracking_uri = tracking_uri
        self.experiments = {
            "simple_models": "Modèles classiques (logistic regression, naive bayes, svm)",
            "advanced_models": "Modèles avancés (LSTM, CNN, ensemble)",
            "bert_models": "Modèles BERT (bert-base, distilbert, roberta)",
            "embeddings_comparison": "Comparaison embeddings (word2vec, glove, fasttext)"
        }

    def setup_mlflow_tracking(self):
        """Configure le tracking URI et initialise MLflow"""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            print(f"MLflow tracking URI configuré: {self.tracking_uri}")
        else:
            # Utilise le tracking local par défaut
            print("MLflow configuré en mode local (./mlruns)")

    def create_experiments(self):
        """Crée toutes les expériences nécessaires"""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        created_experiments = []
        for exp_name, description in self.experiments.items():
            try:
                experiment_id = mlflow.create_experiment(
                    name=exp_name,
                    artifact_location=f"mlruns/{exp_name}",
                    tags={"project": "air_paradis_sentiment", "type": "sentiment_analysis"}
                )
                created_experiments.append((exp_name, experiment_id))
                print(f"Expérience créée: {exp_name} (ID: {experiment_id})")
            except mlflow.exceptions.MlflowException as e:
                if "already exists" in str(e):
                    exp = mlflow.get_experiment_by_name(exp_name)
                    created_experiments.append((exp_name, exp.experiment_id))
                    print(f"Expérience existante: {exp_name} (ID: {exp.experiment_id})")
                else:
                    raise e

        return created_experiments

    def setup_model_registry_tags(self):
        """Configure les tags pour le model registry"""
        registry_tags = {
            "Production": {"stage": "production", "approval": "required"},
            "Staging": {"stage": "staging", "testing": "active"},
            "Archived": {"stage": "archived", "status": "deprecated"}
        }
        return registry_tags

    def initialize_project_structure(self):
        """Initialise la structure complète du projet MLflow"""
        # Configuration du tracking
        self.setup_mlflow_tracking()

        # Création des expériences
        experiments = self.create_experiments()

        # Configuration des tags
        registry_tags = self.setup_model_registry_tags()

        print("\n=== Configuration MLflow terminée ===")
        print(f"Expériences créées: {len(experiments)}")
        print(f"Registry tags configurés: {len(registry_tags)}")

        return {
            "experiments": experiments,
            "registry_tags": registry_tags,
            "tracking_uri": self.tracking_uri
        }


def main():
    """Fonction principale pour initialiser MLflow"""
    setup = MLflowSetup()
    config = setup.initialize_project_structure()

    print("\nPour démarrer le serveur MLflow:")
    print("mlflow ui --host 0.0.0.0 --port 5000")

    return config


if __name__ == "__main__":
    main()