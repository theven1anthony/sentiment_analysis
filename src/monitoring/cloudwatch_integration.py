"""
Intégration AWS CloudWatch pour le monitoring des modèles en production
"""
import boto3
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass


@dataclass
class ModelMetrics:
    """Structure pour les métriques de modèle"""
    timestamp: datetime
    accuracy: float
    latency_ms: float
    throughput_rps: float
    error_rate: float
    prediction_confidence: float


class CloudWatchMonitor:
    """Classe pour monitorer les performances des modèles via CloudWatch"""

    def __init__(self, region_name: str = "eu-west-1", namespace: str = "AirParadis/SentimentAnalysis"):
        self.region_name = region_name
        self.namespace = namespace
        self.cloudwatch = boto3.client('cloudwatch', region_name=region_name)
        self.sns = boto3.client('sns', region_name=region_name)
        self.logger = self._setup_logger()

        # Seuils d'alerte
        self.alert_thresholds = {
            "accuracy_min": 0.80,
            "latency_max_ms": 500,
            "error_rate_max": 0.05,
            "misclassified_count_5min": 3
        }

    def _setup_logger(self):
        """Configure le logger pour CloudWatch"""
        logger = logging.getLogger('cloudwatch_monitor')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def send_custom_metric(self, metric_name: str, value: float,
                          unit: str = "None", dimensions: Dict[str, str] = None):
        """Envoie une métrique personnalisée à CloudWatch"""
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.utcnow()
            }

            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]

            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[metric_data]
            )

            self.logger.info(f"Métrique envoyée: {metric_name} = {value}")

        except Exception as e:
            self.logger.error(f"Erreur envoi métrique {metric_name}: {str(e)}")

    def log_prediction_metrics(self, model_name: str, model_version: str,
                              prediction_time_ms: float, confidence: float,
                              is_correct: Optional[bool] = None):
        """Log les métriques d'une prédiction"""
        dimensions = {
            "ModelName": model_name,
            "ModelVersion": model_version
        }

        # Latence de prédiction
        self.send_custom_metric(
            "PredictionLatency",
            prediction_time_ms,
            "Milliseconds",
            dimensions
        )

        # Confiance de prédiction
        self.send_custom_metric(
            "PredictionConfidence",
            confidence,
            "None",
            dimensions
        )

        # Compteur de prédictions
        self.send_custom_metric(
            "PredictionCount",
            1,
            "Count",
            dimensions
        )

        # Si on connaît la correction de la prédiction
        if is_correct is not None:
            self.send_custom_metric(
                "CorrectPredictions" if is_correct else "IncorrectPredictions",
                1,
                "Count",
                dimensions
            )

    def log_batch_metrics(self, metrics: ModelMetrics, model_name: str, model_version: str):
        """Log un batch de métriques"""
        dimensions = {
            "ModelName": model_name,
            "ModelVersion": model_version
        }

        metrics_to_send = [
            ("Accuracy", metrics.accuracy, "Percent"),
            ("Latency", metrics.latency_ms, "Milliseconds"),
            ("Throughput", metrics.throughput_rps, "Count/Second"),
            ("ErrorRate", metrics.error_rate, "Percent"),
            ("AverageConfidence", metrics.prediction_confidence, "None")
        ]

        for metric_name, value, unit in metrics_to_send:
            self.send_custom_metric(metric_name, value, unit, dimensions)

    def create_dashboard(self, model_name: str):
        """Crée un dashboard CloudWatch pour un modèle"""
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            [self.namespace, "Accuracy", "ModelName", model_name],
                            [self.namespace, "Latency", "ModelName", model_name],
                            [self.namespace, "Throughput", "ModelName", model_name]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region_name,
                        "title": f"Métriques {model_name}"
                    }
                },
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            [self.namespace, "CorrectPredictions", "ModelName", model_name],
                            [self.namespace, "IncorrectPredictions", "ModelName", model_name]
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": self.region_name,
                        "title": f"Prédictions {model_name}"
                    }
                }
            ]
        }

        try:
            self.cloudwatch.put_dashboard(
                DashboardName=f"AirParadis-{model_name}-Metrics",
                DashboardBody=json.dumps(dashboard_body)
            )
            self.logger.info(f"Dashboard créé pour {model_name}")
        except Exception as e:
            self.logger.error(f"Erreur création dashboard: {str(e)}")

    def setup_alerts(self, model_name: str, sns_topic_arn: str):
        """Configure les alarmes CloudWatch"""
        alarms = [
            {
                "AlarmName": f"AirParadis-{model_name}-LowAccuracy",
                "ComparisonOperator": "LessThanThreshold",
                "EvaluationPeriods": 2,
                "MetricName": "Accuracy",
                "Threshold": self.alert_thresholds["accuracy_min"] * 100,
                "AlarmDescription": f"Accuracy trop faible pour {model_name}"
            },
            {
                "AlarmName": f"AirParadis-{model_name}-HighLatency",
                "ComparisonOperator": "GreaterThanThreshold",
                "EvaluationPeriods": 3,
                "MetricName": "Latency",
                "Threshold": self.alert_thresholds["latency_max_ms"],
                "AlarmDescription": f"Latence trop élevée pour {model_name}"
            },
            {
                "AlarmName": f"AirParadis-{model_name}-MisclassifiedTweets",
                "ComparisonOperator": "GreaterThanThreshold",
                "EvaluationPeriods": 1,
                "MetricName": "IncorrectPredictions",
                "Threshold": self.alert_thresholds["misclassified_count_5min"],
                "Statistic": "Sum",
                "Period": 300,  # 5 minutes
                "AlarmDescription": f"Trop de tweets mal classifiés pour {model_name}"
            }
        ]

        for alarm in alarms:
            try:
                self.cloudwatch.put_metric_alarm(
                    AlarmName=alarm["AlarmName"],
                    ComparisonOperator=alarm["ComparisonOperator"],
                    EvaluationPeriods=alarm["EvaluationPeriods"],
                    MetricName=alarm["MetricName"],
                    Namespace=self.namespace,
                    Period=alarm.get("Period", 300),
                    Statistic=alarm.get("Statistic", "Average"),
                    Threshold=alarm["Threshold"],
                    AlarmDescription=alarm["AlarmDescription"],
                    Dimensions=[
                        {"Name": "ModelName", "Value": model_name}
                    ],
                    Unit="None",
                    AlarmActions=[sns_topic_arn],
                    TreatMissingData="notBreaching"
                )
                self.logger.info(f"Alarme créée: {alarm['AlarmName']}")
            except Exception as e:
                self.logger.error(f"Erreur création alarme {alarm['AlarmName']}: {str(e)}")

    def get_model_performance_report(self, model_name: str,
                                   start_time: datetime = None,
                                   end_time: datetime = None) -> Dict[str, Any]:
        """Génère un rapport de performance pour un modèle"""
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.utcnow()

        metrics_to_query = ["Accuracy", "Latency", "Throughput", "ErrorRate"]
        report = {"model_name": model_name, "period": f"{start_time} to {end_time}"}

        for metric in metrics_to_query:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace=self.namespace,
                    MetricName=metric,
                    Dimensions=[{"Name": "ModelName", "Value": model_name}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=3600,  # 1 heure
                    Statistics=["Average", "Maximum", "Minimum"]
                )

                if response["Datapoints"]:
                    datapoints = sorted(response["Datapoints"], key=lambda x: x["Timestamp"])
                    report[metric] = {
                        "average": sum(d["Average"] for d in datapoints) / len(datapoints),
                        "max": max(d["Maximum"] for d in datapoints),
                        "min": min(d["Minimum"] for d in datapoints),
                        "trend": "stable"  # Calcul simple de tendance
                    }
                else:
                    report[metric] = {"status": "no_data"}

            except Exception as e:
                report[metric] = {"error": str(e)}

        return report


class MLflowCloudWatchIntegration:
    """Intègre MLflow avec CloudWatch pour un monitoring unifié"""

    def __init__(self, cloudwatch_monitor: CloudWatchMonitor):
        self.cw_monitor = cloudwatch_monitor

    def sync_mlflow_metrics_to_cloudwatch(self, run_id: str, model_name: str):
        """Synchronise les métriques MLflow vers CloudWatch"""
        import mlflow

        try:
            run = mlflow.get_run(run_id)
            metrics = run.data.metrics

            # Mapping des métriques MLflow vers CloudWatch
            metric_mapping = {
                "test.accuracy": ("Accuracy", "Percent"),
                "test.f1_score": ("F1Score", "None"),
                "performance.inference_time_ms": ("Latency", "Milliseconds"),
                "performance.model_size_mb": ("ModelSize", "Megabytes")
            }

            dimensions = {
                "ModelName": model_name,
                "RunId": run_id[:8]
            }

            for mlflow_metric, (cw_metric, unit) in metric_mapping.items():
                if mlflow_metric in metrics:
                    value = metrics[mlflow_metric]
                    if mlflow_metric == "test.accuracy":
                        value *= 100  # Convertir en pourcentage

                    self.cw_monitor.send_custom_metric(
                        cw_metric, value, unit, dimensions
                    )

        except Exception as e:
            self.cw_monitor.logger.error(f"Erreur sync MLflow->CloudWatch: {str(e)}")

    def create_model_performance_alert(self, model_name: str, mlflow_run_id: str,
                                     sns_topic_arn: str):
        """Crée des alertes basées sur les métriques MLflow"""
        self.sync_mlflow_metrics_to_cloudwatch(mlflow_run_id, model_name)
        self.cw_monitor.setup_alerts(model_name, sns_topic_arn)
        self.cw_monitor.create_dashboard(model_name)