"""
Modèles Pydantic pour l'API FastAPI.
Définit les schémas de requête et de réponse.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PredictRequest(BaseModel):
    """Requête de prédiction de sentiment."""

    text: str = Field(..., min_length=1, max_length=1000, description="Texte à analyser")

    class Config:
        json_schema_extra = {"example": {"text": "I love this product! It's amazing!"}}


class PredictResponse(BaseModel):
    """Réponse de prédiction de sentiment."""

    sentiment: int = Field(..., description="Sentiment prédit (0=négatif, 1=positif)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance de la prédiction (0.0-1.0)")
    text: str = Field(..., description="Texte original")
    prediction_id: str = Field(..., description="ID unique de la prédiction")
    timestamp: str = Field(..., description="Timestamp de la prédiction")

    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": 1,
                "confidence": 0.87,
                "text": "I love this product! It's amazing!",
                "prediction_id": "pred_123abc",
                "timestamp": "2025-10-07T13:30:00",
            }
        }


class FeedbackRequest(BaseModel):
    """Requête de feedback sur une prédiction."""

    text: str = Field(..., description="Texte original")
    predicted_sentiment: int = Field(..., ge=0, le=1, description="Sentiment prédit par le modèle")
    actual_sentiment: int = Field(..., ge=0, le=1, description="Sentiment correct (0=négatif, 1=positif)")
    prediction_id: Optional[str] = Field(None, description="ID de la prédiction (optionnel)")
    timestamp: Optional[str] = Field(None, description="Timestamp du feedback")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product!",
                "predicted_sentiment": 0,
                "actual_sentiment": 1,
                "prediction_id": "pred_123abc",
                "timestamp": "2025-10-07T13:30:00",
            }
        }


class FeedbackResponse(BaseModel):
    """Réponse après enregistrement du feedback."""

    status: str = Field(..., description="Statut de l'enregistrement")
    message: str = Field(..., description="Message de confirmation")
    alert_triggered: bool = Field(..., description="True si alerte déclenchée (3 erreurs en 5 min)")
    misclassified_count: int = Field(..., description="Nombre d'erreurs dans la fenêtre de 5 min")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "feedback_recorded",
                "message": "Merci pour votre retour",
                "alert_triggered": False,
                "misclassified_count": 1,
            }
        }


class HealthResponse(BaseModel):
    """Réponse du health check."""

    status: str = Field(..., description="État de santé de l'API")
    model_loaded: bool = Field(..., description="True si le modèle est chargé")
    model_type: Optional[str] = Field(None, description="Type de modèle chargé")
    timestamp: str = Field(..., description="Timestamp du health check")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_type": "logistic_regression",
                "timestamp": "2025-10-07T13:30:00",
            }
        }


class ModelInfoResponse(BaseModel):
    """Informations sur le modèle chargé."""

    model_type: str = Field(..., description="Type de modèle")
    technique: str = Field(..., description="Technique de prétraitement")
    f1_score: Optional[float] = Field(None, description="F1-Score sur le test set")
    accuracy: Optional[float] = Field(None, description="Accuracy sur le test set")
    training_date: Optional[str] = Field(None, description="Date d'entraînement")

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "logistic_regression",
                "technique": "stemming",
                "f1_score": 0.7754,
                "accuracy": 0.7754,
                "training_date": "2025-10-07",
            }
        }
