"""
Intégration Azure Monitor et Application Insights pour le monitoring des modèles en production
"""

import os
import logging
from typing import Dict, Optional
from datetime import datetime

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace, metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider


class AzureMonitor:
    """Classe pour monitorer les performances des modèles via Azure Monitor"""

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialise le monitoring Azure Monitor avec Application Insights

        Args:
            connection_string: Connection String d'Application Insights
                             Si None, utilise la variable d'environnement APPLICATIONINSIGHTS_CONNECTION_STRING
        """
        self.connection_string = connection_string or os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        self.logger = self._setup_logger()

        # Seuils d'alerte (identiques au projet AWS)
        self.alert_thresholds = {
            "accuracy_min": 0.80,
            "latency_max_ms": 500,
            "error_rate_max": 0.05,
            "misclassified_count_5min": 3,
        }

        # Configurer Azure Monitor si connection string disponible
        if self.connection_string:
            try:
                # configure_azure_monitor() lit automatiquement depuis APPLICATIONINSIGHTS_CONNECTION_STRING
                # Il ne faut PAS passer la connection string en paramètre
                configure_azure_monitor()
                self.tracer = trace.get_tracer(__name__)
                self.meter = metrics.get_meter(__name__)
                self.logger.info("Azure Monitor configuré avec succès")
            except Exception as e:
                self.logger.warning(f"Impossible de configurer Azure Monitor: {e}")
                self.tracer = None
                self.meter = None
        else:
            self.logger.warning("Connection String Application Insights non fournie - monitoring désactivé")
            self.tracer = None
            self.meter = None

    def _setup_logger(self):
        """Configure le logger pour Azure Monitor"""
        logger = logging.getLogger("azure_monitor")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def log_prediction_metrics(
        self,
        model_name: str,
        model_version: str,
        prediction_time_ms: float,
        confidence: float,
        sentiment: Optional[int] = None,
        text: Optional[str] = None,
        is_correct: Optional[bool] = None,
    ):
        """
        Log les métriques d'une prédiction dans Application Insights

        Args:
            model_name: Nom du modèle
            model_version: Version du modèle
            prediction_time_ms: Temps de prédiction en millisecondes
            confidence: Niveau de confiance de la prédiction
            sentiment: Sentiment prédit (0=négatif, 1=positif)
            text: Texte du tweet (tronqué si > 200 caractères)
            is_correct: Si la prédiction est correcte (optionnel)
        """
        if not self.tracer:
            return

        try:
            with self.tracer.start_as_current_span("prediction") as span:
                # Ajouter les attributs au span
                span.set_attribute("model.name", model_name)
                span.set_attribute("model.version", model_version)
                span.set_attribute("prediction.latency_ms", prediction_time_ms)
                span.set_attribute("prediction.confidence", confidence)

                # Ajouter le sentiment prédit
                if sentiment is not None:
                    span.set_attribute("prediction.sentiment", sentiment)
                    sentiment_label = "négatif" if sentiment == 0 else "positif"
                    span.set_attribute("prediction.sentiment_label", sentiment_label)

                # Ajouter le texte du tweet (tronqué pour éviter trop de données)
                if text is not None:
                    # Limiter à 200 caractères pour éviter surcharge
                    text_truncated = text[:200] if len(text) > 200 else text
                    span.set_attribute("prediction.text", text_truncated)

                if is_correct is not None:
                    span.set_attribute("prediction.is_correct", is_correct)

                self.logger.info(
                    f"Prédiction loggée: {model_name}@{model_version} - "
                    f"Latence: {prediction_time_ms:.2f}ms, Confiance: {confidence:.3f}, "
                    f"Sentiment: {sentiment_label if sentiment is not None else 'N/A'}"
                )
        except Exception as e:
            self.logger.error(f"Erreur lors du log de métriques: {str(e)}")

    def log_custom_event(self, event_name: str, properties: Dict[str, any] = None):
        """
        Log un événement personnalisé dans Application Insights

        Args:
            event_name: Nom de l'événement
            properties: Propriétés additionnelles de l'événement
        """
        if not self.tracer:
            return

        try:
            with self.tracer.start_as_current_span(event_name) as span:
                if properties:
                    for key, value in properties.items():
                        span.set_attribute(key, value)

                self.logger.info(f"Événement loggé: {event_name} - {properties}")
        except Exception as e:
            self.logger.error(f"Erreur lors du log d'événement: {str(e)}")

    def log_misclassification(self, text: str, predicted_sentiment: int, actual_sentiment: int, confidence: float):
        """
        Log une erreur de classification dans Application Insights

        Args:
            text: Texte mal classifié
            predicted_sentiment: Sentiment prédit
            actual_sentiment: Sentiment réel
            confidence: Niveau de confiance de la prédiction
        """
        properties = {
            "text_preview": text[:100],
            "predicted_sentiment": predicted_sentiment,
            "actual_sentiment": actual_sentiment,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }

        self.log_custom_event("misclassification", properties)
        self.logger.warning(
            f"Misclassification détectée: Prédit={predicted_sentiment}, "
            f"Réel={actual_sentiment}, Confiance={confidence:.3f}"
        )

    def log_alert_triggered(self, alert_type: str, details: Dict[str, any]):
        """
        Log le déclenchement d'une alerte

        Args:
            alert_type: Type d'alerte (ex: "high_error_rate")
            details: Détails de l'alerte
        """
        properties = {"alert_type": alert_type, "timestamp": datetime.now().isoformat(), **details}

        self.log_custom_event("alert_triggered", properties)
        self.logger.error(f"Alerte déclenchée: {alert_type} - {details}")


# Instance globale pour utilisation dans l'API
azure_monitor = AzureMonitor()
