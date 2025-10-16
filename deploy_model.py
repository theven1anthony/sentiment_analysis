#!/usr/bin/env python3
"""
Script de déploiement du meilleur modèle en production.
Déploie un modèle depuis MLflow Model Registry vers models/production/.
"""

import os
import sys
import pickle
import mlflow
import mlflow.pyfunc
import argparse
from datetime import datetime

# Ajouter src au PYTHONPATH pour permettre l'import des modules custom
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configuration
PRODUCTION_DIR = "./models/production"
STAGING_DIR = "./models/staging"


def deploy_model_from_registry(model_name, model_version=1):
    """
    Déploie un modèle depuis MLflow Model Registry.

    Args:
        model_name: Nom du modèle dans MLflow Model Registry
        model_version: Version du modèle (défaut: 1)

    Returns:
        True si succès, False sinon
    """
    print(f"\n=== DÉPLOIEMENT DEPUIS MLFLOW MODEL REGISTRY ===")
    print(f"Modèle: {model_name}")
    print(f"Version: {model_version}")

    try:
        # 1. Obtenir les informations du modèle depuis Model Registry
        client = mlflow.MlflowClient()

        try:
            model_version_info = client.get_model_version(model_name, model_version)
            run_id = model_version_info.run_id
            print(f"\n✓ Modèle trouvé dans Model Registry")
            print(f"  Run ID: {run_id}")
            print(f"  Status: {model_version_info.status}")
        except Exception as e:
            print(f"✗ Modèle '{model_name}' version {model_version} non trouvé: {e}")
            return False

        # 2. Récupérer les métriques et paramètres du run
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        params = run.data.params

        print(f"\nMétriques:")
        print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  AUC: {metrics.get('auc_score', 0):.4f}")

        print(f"\nParamètres:")
        print(f"  Architecture: {params.get('architecture', 'unknown')}")
        print(f"  Technique: {params.get('technique', 'unknown')}")
        print(f"  Vector size: {params.get('vector_size', 'N/A')}")

        # 3. Charger le modèle pyfunc
        model_uri = f"models:/{model_name}/{model_version}"
        print(f"\nChargement du modèle pyfunc depuis: {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✓ Modèle pyfunc chargé (pipeline complet)")

        # 4. Créer le dossier production
        os.makedirs(PRODUCTION_DIR, exist_ok=True)

        # 5. Sauvegarder l'URI du modèle pour l'API
        model_uri_file = os.path.join(PRODUCTION_DIR, "model_uri.txt")
        with open(model_uri_file, 'w') as f:
            f.write(model_uri)
        print(f"\n✓ URI du modèle sauvegardé: {model_uri_file}")
        print(f"  URI: {model_uri}")

        # 6. Créer les métadonnées
        metadata = {
            'model_name': model_name,
            'model_version': model_version,
            'run_id': run_id,
            'model_type': params.get('architecture', 'unknown'),
            'technique': params.get('technique', 'stemming'),
            'vector_size': int(params.get('vector_size', 100)),
            'vocab_size': int(params.get('vocab_size', 0)),
            'f1_score': metrics.get('f1_score'),
            'accuracy': metrics.get('accuracy'),
            'auc_score': metrics.get('auc_score'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'training_time': metrics.get('training_time'),
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'deployment_date': datetime.now().isoformat()
        }

        metadata_path = os.path.join(PRODUCTION_DIR, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Métadonnées sauvegardées: {metadata_path}")

        print(f"\n=== DÉPLOIEMENT TERMINÉ ===")
        print(f"\nProchaines étapes:")
        print(f"1. Adapter l'API (api/main.py) pour charger le modèle TensorFlow")
        print(f"2. Vérifier que l'embedding Word2Vec est disponible")
        print(f"3. Tester: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")

        return True

    except Exception as e:
        print(f"\n✗ Erreur lors du déploiement: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Déploie un modèle depuis MLflow Model Registry en production."""

    parser = argparse.ArgumentParser(
        description="Déploie un modèle depuis MLflow Model Registry vers production"
    )
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Nom du modèle dans MLflow Model Registry (ex: w2v_200K_model)'
    )
    parser.add_argument(
        '--version',
        type=int,
        default=1,
        help='Version du modèle (défaut: 1)'
    )

    args = parser.parse_args()

    print("=== DÉPLOIEMENT DU MODÈLE EN PRODUCTION ===\n")

    success = deploy_model_from_registry(args.name, args.version)

    if not success:
        print(f"\n✗ Échec du déploiement")
        sys.exit(1)


if __name__ == "__main__":
    main()