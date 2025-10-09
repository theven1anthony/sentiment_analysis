"""
API FastAPI pour la prédiction de sentiment.
Utilise un modèle MLflow pyfunc encapsulant tout le pipeline.
"""

import os
import sys
import pickle
import uuid
from datetime import datetime, timedelta
from collections import deque
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import mlflow.pyfunc

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.models import (
    PredictRequest,
    PredictResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse
)

# Configuration
MODEL_URI_PATH = os.getenv("MODEL_URI_PATH", "./models/production/model_uri.txt")
MODEL_METADATA_PATH = os.getenv("MODEL_METADATA_PATH", "./models/production/metadata.pkl")
ALERT_WINDOW_MINUTES = 5
ALERT_THRESHOLD = 3

# Initialisation de l'application
app = FastAPI(
    title="Air Paradis Sentiment Analysis API",
    description="API de prédiction de sentiment pour tweets",
    version="1.0.0"
)

# CORS pour permettre les requêtes depuis Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model = None
model_metadata = {}
misclassified_queue = deque()  # Queue pour stocker les erreurs dans la fenêtre de 5 min


def load_model():
    """
    Charge le modèle pyfunc MLflow et ses métadonnées au démarrage.

    Supporte deux modes:
    1. Production (AWS): Variables d'environnement MODEL_NAME + MODEL_STAGE/MODEL_VERSION
    2. Développement: Fichier local model_uri.txt
    """
    global model, model_metadata

    try:
        model_uri = None

        # Mode 1: Variables d'environnement (prioritaire pour déploiement AWS)
        model_name = os.getenv("MODEL_NAME")
        model_version = os.getenv("MODEL_VERSION")
        model_stage = os.getenv("MODEL_STAGE", "Production")

        if model_name:
            # Charger depuis MLflow Model Registry
            if model_stage and not model_version:
                model_uri = f"models:/{model_name}/{model_stage}"
                print(f"✓ Configuration depuis variables d'environnement:")
                print(f"  MODEL_NAME={model_name}")
                print(f"  MODEL_STAGE={model_stage}")
            elif model_version:
                model_uri = f"models:/{model_name}/{model_version}"
                print(f"✓ Configuration depuis variables d'environnement:")
                print(f"  MODEL_NAME={model_name}")
                print(f"  MODEL_VERSION={model_version}")
            else:
                model_uri = f"models:/{model_name}/Production"
                print(f"✓ Configuration depuis variables d'environnement:")
                print(f"  MODEL_NAME={model_name}")
                print(f"  MODEL_STAGE=Production (défaut)")

        # Mode 2: Fallback sur fichier local (développement)
        elif os.path.exists(MODEL_URI_PATH):
            with open(MODEL_URI_PATH, 'r') as f:
                model_uri = f.read().strip()
            print(f"✓ Configuration depuis fichier local:")
            print(f"  Fichier: {MODEL_URI_PATH}")

            # Charger les métadonnées locales si disponibles
            if os.path.exists(MODEL_METADATA_PATH):
                with open(MODEL_METADATA_PATH, 'rb') as f:
                    model_metadata = pickle.load(f)
                print(f"✓ Métadonnées locales chargées:")
                print(f"  Type: {model_metadata.get('model_type', 'unknown')}")
                print(f"  F1-Score: {model_metadata.get('f1_score', 0):.4f}")

        else:
            # Aucune configuration disponible
            print(f"✗ Configuration manquante:")
            print(f"  Option 1 (Production): Définir MODEL_NAME et optionnellement MODEL_STAGE/MODEL_VERSION")
            print(f"  Option 2 (Développement): Exécuter python deploy_best_model.py --name <model_name> --version <version>")
            model = None
            return

        # Charger le modèle pyfunc MLflow
        print(f"\nChargement du modèle depuis: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✓ Modèle pyfunc chargé (pipeline complet: preprocessing + embedding + prédiction)")

        # Récupérer les métadonnées depuis MLflow si pas déjà chargées
        if not model_metadata and model_name:
            try:
                client = mlflow.MlflowClient()
                if model_version:
                    model_version_info = client.get_model_version(model_name, model_version)
                else:
                    # Récupérer la dernière version du stage
                    versions = client.get_latest_versions(model_name, stages=[model_stage])
                    if versions:
                        model_version_info = versions[0]
                    else:
                        raise ValueError(f"Aucune version trouvée pour {model_name}/{model_stage}")

                run = mlflow.get_run(model_version_info.run_id)
                model_metadata = {
                    'model_name': model_name,
                    'model_version': model_version_info.version,
                    'run_id': model_version_info.run_id,
                    'model_type': run.data.params.get('architecture', 'unknown'),
                    'technique': run.data.params.get('technique', 'unknown'),
                    'f1_score': run.data.metrics.get('f1_score'),
                    'accuracy': run.data.metrics.get('accuracy'),
                    'training_date': run.info.start_time
                }
                print(f"✓ Métadonnées récupérées depuis MLflow:")
                print(f"  Type: {model_metadata.get('model_type')}")
                print(f"  F1-Score: {model_metadata.get('f1_score', 0):.4f}")
            except Exception as e:
                print(f"⚠ Impossible de récupérer les métadonnées depuis MLflow: {e}")

    except Exception as e:
        print(f"✗ Erreur lors du chargement du modèle: {e}")
        import traceback
        traceback.print_exc()
        model = None


def clean_old_misclassifications():
    """Supprime les erreurs plus anciennes que ALERT_WINDOW_MINUTES."""
    global misclassified_queue

    cutoff_time = datetime.now() - timedelta(minutes=ALERT_WINDOW_MINUTES)

    while misclassified_queue and misclassified_queue[0] < cutoff_time:
        misclassified_queue.popleft()


@app.on_event("startup")
async def startup_event():
    """Événement de démarrage : charge le modèle."""
    load_model()


@app.get("/", tags=["Root"])
async def root():
    """Endpoint racine avec informations de base."""
    return {
        "message": "Air Paradis Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "feedback": "/feedback",
            "health": "/health",
            "model_info": "/model/info"
        }
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Prédit le sentiment d'un texte.

    Args:
        request: Requête contenant le texte à analyser

    Returns:
        Prédiction avec sentiment (0/1), confiance, et métadonnées
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Exécutez: python deploy_best_model.py --name <model_name> --version <version>"
        )

    try:
        # Le modèle pyfunc gère tout le pipeline (preprocessing + embedding + prédiction)
        # On passe directement le texte brut
        input_df = pd.DataFrame({"text": [request.text]})

        # Prédiction via le modèle pyfunc
        predictions = model.predict(input_df)

        # Extraire sentiment et confidence
        sentiment = int(predictions['sentiment'].iloc[0])
        confidence = float(predictions['confidence'].iloc[0])

        # Générer ID et timestamp
        prediction_id = f"pred_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()

        return PredictResponse(
            sentiment=sentiment,
            confidence=confidence,
            text=request.text,
            prediction_id=prediction_id,
            timestamp=timestamp
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def feedback(request: FeedbackRequest):
    """
    Enregistre un feedback sur une prédiction.
    Déclenche une alerte si 3 erreurs en 5 minutes.

    Args:
        request: Feedback contenant le texte, sentiment prédit et réel

    Returns:
        Statut de l'enregistrement et alerte si nécessaire
    """
    global misclassified_queue

    try:
        # 1. Nettoyer les anciennes erreurs
        clean_old_misclassifications()

        # 2. Vérifier si c'est une erreur de classification
        is_misclassified = request.predicted_sentiment != request.actual_sentiment
        alert_triggered = False

        if is_misclassified:
            # Ajouter l'erreur à la queue
            misclassified_queue.append(datetime.now())

            # Vérifier le seuil d'alerte
            if len(misclassified_queue) >= ALERT_THRESHOLD:
                alert_triggered = True

                # TODO: Intégration AWS CloudWatch / SNS pour envoyer l'alerte
                print(f"⚠️  ALERTE: {len(misclassified_queue)} erreurs en {ALERT_WINDOW_MINUTES} minutes")
                print(f"   Texte: {request.text[:50]}...")
                print(f"   Prédit: {request.predicted_sentiment}, Réel: {request.actual_sentiment}")

        misclassified_count = len(misclassified_queue)

        return FeedbackResponse(
            status="feedback_recorded",
            message="Merci pour votre retour" if not alert_triggered else "Alerte déclenchée: performance du modèle dégradée",
            alert_triggered=alert_triggered,
            misclassified_count=misclassified_count
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'enregistrement du feedback: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """
    Health check de l'API.
    Utilisé par AWS CloudWatch pour monitoring.

    Returns:
        Statut de santé de l'API et du modèle
    """
    model_loaded = model is not None
    status = "healthy" if model_loaded else "unhealthy"
    model_type = model_metadata.get('model_type') if model_metadata else None

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_type=model_type,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Monitoring"])
async def model_info():
    """
    Retourne les informations sur le modèle chargé.

    Returns:
        Métadonnées du modèle (type, métriques, date d'entraînement)
    """
    if not model_metadata:
        raise HTTPException(
            status_code=404,
            detail="Métadonnées du modèle non disponibles"
        )

    return ModelInfoResponse(
        model_type=model_metadata.get('model_type', 'unknown'),
        technique=model_metadata.get('technique', 'unknown'),
        f1_score=model_metadata.get('f1_score'),
        accuracy=model_metadata.get('accuracy'),
        training_date=model_metadata.get('training_date')
    )