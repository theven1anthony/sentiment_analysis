#!/usr/bin/env python3
"""
Script d'initialisation complète de l'architecture MLflow pour Air Paradis
Usage: python setup_mlflow.py
"""
import sys
import os
from pathlib import Path

# Ajouter le dossier src au path pour les imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.mlflow_setup import MLflowSetup
from src.monitoring.cloudwatch_integration import CloudWatchMonitor


def main():
    """Fonction principale d'initialisation"""
    print("=== Initialisation Architecture MLflow Air Paradis ===\n")

    # 1. Configuration MLflow
    print("1. Configuration du tracking MLflow...")
    mlflow_setup = MLflowSetup()
    config = mlflow_setup.initialize_project_structure()

    # 2. Création des dossiers nécessaires (MLflow gérera mlruns/ automatiquement)
    print("\n2. Création de la structure de dossiers...")
    folders_to_create = [
        "models/production",
        "models/staging",
        "models/archived",
        "data/raw",
        "data/processed",
        "data/features",
        "logs/mlflow",
        "logs/monitoring"
    ]

    for folder in folders_to_create:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Dossier créé: {folder}")

    print("  ✓ MLflow gérera automatiquement la structure mlruns/")

    # 3. Configuration CloudWatch (optionnel)
    print("\n3. Configuration CloudWatch...")
    try:
        cw_monitor = CloudWatchMonitor()
        print("  ✓ CloudWatch configuré (AWS credentials requis pour fonctionnement)")
    except Exception as e:
        print(f"  ⚠ CloudWatch non configuré: {str(e)}")
        print("  → Configurer AWS CLI avec: aws configure")

    # 4. Création des fichiers de configuration
    print("\n4. Création des fichiers de configuration...")

    # MLflow configuration
    mlflow_config = """# Configuration MLflow - Air Paradis Sentiment Analysis
tracking_uri: http://localhost:5001
experiments:
  - simple_models
  - advanced_models
  - bert_models
  - embeddings_comparison

model_registry:
  stages:
    - None
    - Staging
    - Production
    - Archived

monitoring:
  cloudwatch:
    namespace: "AirParadis/SentimentAnalysis"
    region: "eu-west-1"
  alerts:
    accuracy_threshold: 0.80
    latency_threshold_ms: 500
    misclassified_threshold_5min: 3
"""

    with open("mlflow_config.yaml", "w") as f:
        f.write(mlflow_config)
    print("  ✓ mlflow_config.yaml créé")

    # Docker compose pour MLflow
    docker_compose = """version: '3.8'
services:
  mlflow:
    image: python:3.12-slim
    ports:
      - "5001:5001"
    volumes:
      - ./mlruns:/mlflow/mlruns
      - ./models:/mlflow/models
    working_dir: /mlflow
    command: >
      bash -c "
        pip install mlflow[extras]==2.16.0 boto3 &&
        mlflow server
          --backend-store-uri sqlite:///mlruns/mlflow.db
          --default-artifact-root ./mlruns
          --host 0.0.0.0
          --port 5001
      "
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5001
"""

    with open("docker-compose.mlflow.yml", "w") as f:
        f.write(docker_compose)
    print("  ✓ docker-compose.mlflow.yml créé")

    # 5. Instructions finales
    print("\n=== Configuration terminée ===")
    print("\nProchaines étapes:")
    print("1. Installer les dépendances: pip install -r requirements.txt")
    print("2. Démarrer MLflow: mlflow ui --host 0.0.0.0 --port 5001")
    print("3. Accéder à l'interface: http://localhost:5001")
    print("\nPour AWS CloudWatch:")
    print("4. Configurer AWS CLI: aws configure")
    print("5. Créer un topic SNS pour les alertes")

    print(f"\nExpériences créées: {len(config['experiments'])}")
    for exp_name, exp_id in config['experiments']:
        print(f"  - {exp_name} (ID: {exp_id})")

    print("\nArchitecture MLflow prête pour le développement! 🚀")


if __name__ == "__main__":
    main()