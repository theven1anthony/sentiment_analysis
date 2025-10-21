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

# Configuration MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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

        # 2. Récupérer les métriques et paramètres du run (optionnel)
        try:
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
        except Exception as e:
            print(f"\n⚠ Impossible de récupérer les métriques/paramètres: {e}")
            print(f"  Le déploiement continuera quand même...")
            metrics = {}
            params = {}

        # 3. Charger le modèle pyfunc
        model_uri = f"models:/{model_name}/{model_version}"
        print(f"\nChargement du modèle pyfunc depuis: {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✓ Modèle pyfunc chargé (pipeline complet)")

        # 4. Créer/nettoyer le dossier production
        import shutil
        if os.path.exists(PRODUCTION_DIR):
            # Garder .gitkeep si présent
            gitkeep_path = os.path.join(PRODUCTION_DIR, ".gitkeep")
            has_gitkeep = os.path.exists(gitkeep_path)

            print(f"\nNettoyage du dossier: {PRODUCTION_DIR}")
            shutil.rmtree(PRODUCTION_DIR)
            os.makedirs(PRODUCTION_DIR, exist_ok=True)

            if has_gitkeep:
                open(gitkeep_path, 'a').close()
        else:
            os.makedirs(PRODUCTION_DIR, exist_ok=True)

        # 5. Télécharger le modèle pyfunc complet depuis MLflow Model Registry
        # Télécharger dans un dossier temporaire d'abord
        temp_download_path = os.path.join(PRODUCTION_DIR, "temp_download")
        print(f"\nTéléchargement du modèle complet dans: {temp_download_path}")

        # Utiliser l'URI du Model Registry (pas du run) car le modèle est stocké là
        # Format: models:/<model_name>/<version>
        artifact_path = f"models:/{model_name}/{model_version}"
        downloaded_path = mlflow.artifacts.download_artifacts(artifact_path, dst_path=temp_download_path)

        print(f"✓ Modèle pyfunc téléchargé depuis Model Registry")

        # 5b. Réorganiser pour créer la structure pyfunc_model/model/ attendue par l'API
        pyfunc_model_path = os.path.join(PRODUCTION_DIR, "pyfunc_model")
        model_subdir = os.path.join(pyfunc_model_path, "model")
        os.makedirs(model_subdir, exist_ok=True)

        # Déplacer tous les fichiers téléchargés dans le sous-dossier model/
        print(f"\nRéorganisation de la structure pour compatibilité avec l'API...")
        for item in os.listdir(temp_download_path):
            src = os.path.join(temp_download_path, item)
            dst = os.path.join(model_subdir, item)
            shutil.move(src, dst)

        # Supprimer le dossier temporaire
        os.rmdir(temp_download_path)
        print(f"✓ Structure créée: {model_subdir}")

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

        # 7. Afficher la taille totale
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(PRODUCTION_DIR)
            for filename in filenames
        )
        total_size_mb = total_size / (1024 * 1024)

        print(f"\n=== DÉPLOIEMENT TERMINÉ ===")
        print(f"\nTaille totale: {total_size_mb:.1f} MB")
        print(f"Emplacement: {os.path.abspath(PRODUCTION_DIR)}")
        print(f"\nProchaines étapes:")
        print(f"1. Tester l'API localement: uvicorn api.main:app --reload")
        print(f"2. Commit sur Git: git add models/production/")
        print(f"3. Push: git push origin main")
        print(f"4. GitHub Actions déploiera automatiquement sur AWS")

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