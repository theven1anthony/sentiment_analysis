"""
Script pour générer des feedbacks de démonstration avec misclassifications.
Permet de tester les alertes et les logs Azure Application Insights.

Usage:
    python scripts/generate_demo_feedbacks.py --count 5 --trigger-alert
"""

import requests
import time
import random
import argparse
from datetime import datetime
from typing import Dict, List


# Tweets de démonstration avec misclassifications intentionnelles
MISCLASSIFIED_EXAMPLES = [
    {
        "text": "Flight was okay I guess, nothing special but got there",
        "predicted": 1,  # Modèle prédit positif
        "actual": 0,     # En réalité négatif
        "confidence": 0.62
    },
    {
        "text": "@AirParadis terrible service! Lost my luggage and crew was rude",
        "predicted": 1,  # Modèle prédit positif (erreur grave)
        "actual": 0,     # En réalité négatif
        "confidence": 0.58
    },
    {
        "text": "Air Paradis is not the worst but definitely not the best either",
        "predicted": 0,  # Modèle prédit négatif
        "actual": 1,     # En réalité positif (neutre -> positif)
        "confidence": 0.55
    },
    {
        "text": "Disappointed with the delay but staff tried their best to help",
        "predicted": 0,  # Modèle prédit négatif
        "actual": 1,     # En réalité positif (malgré problème)
        "confidence": 0.67
    },
    {
        "text": "Air Paradis could improve but my flight was acceptable overall",
        "predicted": 0,  # Modèle prédit négatif
        "actual": 1,     # En réalité positif
        "confidence": 0.53
    },
    {
        "text": "Not impressed. Seats uncomfortable and food was bad",
        "predicted": 1,  # Modèle prédit positif (erreur)
        "actual": 0,     # En réalité négatif
        "confidence": 0.61
    },
    {
        "text": "Flight cancelled AGAIN! This is unacceptable @AirParadis",
        "predicted": 1,  # Modèle prédit positif (erreur grave)
        "actual": 0,     # En réalité négatif
        "confidence": 0.59
    },
    {
        "text": "Service was fine, nothing to complain about really",
        "predicted": 0,  # Modèle prédit négatif
        "actual": 1,     # En réalité positif (neutre -> positif)
        "confidence": 0.56
    },
]


def send_feedback(api_url: str, feedback_data: Dict) -> Dict:
    """
    Envoie un feedback à l'API.

    Args:
        api_url: URL de l'API (sans /feedback)
        feedback_data: Données du feedback

    Returns:
        Réponse JSON de l'API
    """
    endpoint = f"{api_url}/feedback"

    try:
        response = requests.post(
            endpoint,
            json=feedback_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de connexion: {e}")
        return None


def generate_demo_feedbacks(
    api_url: str,
    count: int = 5,
    delay_min: float = 1.0,
    delay_max: float = 3.0,
    trigger_alert: bool = True
) -> Dict[str, int]:
    """
    Génère des feedbacks de démonstration avec misclassifications.

    Args:
        api_url: URL de l'API
        count: Nombre de feedbacks à générer
        delay_min: Délai minimum entre requêtes (secondes)
        delay_max: Délai maximum entre requêtes (secondes)
        trigger_alert: Si True, envoie 3+ feedbacks rapidement pour déclencher l'alerte

    Returns:
        Statistiques des feedbacks générés
    """
    print(f"\n{'='*80}")
    print(f"GÉNÉRATION DE FEEDBACKS DE DÉMONSTRATION - AIR PARADIS")
    print(f"{'='*80}\n")
    print(f"API URL          : {api_url}")
    print(f"Nombre           : {count} feedbacks")
    print(f"Délai            : {delay_min}-{delay_max}s entre requêtes")
    print(f"Déclencher alerte: {'Oui (3 erreurs en <5min)' if trigger_alert else 'Non'}")
    print(f"\nDémarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"{'-'*80}\n")

    # Statistiques
    stats = {
        "total": 0,
        "success": 0,
        "errors": 0,
        "misclassified": 0,
        "alerts_triggered": 0,
    }

    # Sélectionner des exemples aléatoirement
    feedback_pool = random.choices(MISCLASSIFIED_EXAMPLES, k=count)

    # Si on veut déclencher l'alerte, envoyer les 3 premiers rapidement
    alert_triggered_at = None

    for i, example in enumerate(feedback_pool, 1):
        stats["total"] += 1

        # Construire le payload du feedback
        feedback_payload = {
            "text": example["text"],
            "predicted_sentiment": example["predicted"],
            "actual_sentiment": example["actual"],
            "confidence": example["confidence"]
        }

        # Afficher l'exemple
        text_preview = example["text"][:60] + "..." if len(example["text"]) > 60 else example["text"]
        predicted_label = "Positif" if example["predicted"] == 1 else "Négatif"
        actual_label = "Positif" if example["actual"] == 1 else "Négatif"

        print(f"[{i:3d}/{count}] {text_preview}")
        print(f"         Prédit: {predicted_label} | Réel: {actual_label} | Confiance: {example['confidence']:.3f}")

        # Envoyer le feedback
        result = send_feedback(api_url, feedback_payload)

        if result:
            stats["success"] += 1
            stats["misclassified"] += 1

            # Vérifier si une alerte a été déclenchée
            if result.get("alert_triggered", False):
                stats["alerts_triggered"] += 1
                alert_triggered_at = i
                print(f"         → ⚠️  ALERTE DÉCLENCHÉE (misclassified_count: {result.get('misclassified_count')})")
            else:
                print(f"         → ✓ Feedback enregistré (misclassified_count: {result.get('misclassified_count')})")
        else:
            stats["errors"] += 1
            print(f"         → ❌ Échec")

        # Délai avant prochaine requête
        if i < count:
            # Si on veut déclencher l'alerte et qu'on est dans les 3 premiers, délai court
            if trigger_alert and i < 3:
                delay = 2.0  # 2 secondes pour rester dans la fenêtre de 5 minutes
            else:
                delay = random.uniform(delay_min, delay_max)

            time.sleep(delay)

    # Afficher le résumé
    print(f"\n{'-'*80}\n")
    print(f"RÉSUMÉ DE LA GÉNÉRATION")
    print(f"{'='*80}\n")
    print(f"Total envoyé         : {stats['total']}")
    print(f"Succès               : {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"Erreurs              : {stats['errors']}")
    print(f"Misclassifications   : {stats['misclassified']}")
    print(f"Alertes déclenchées  : {stats['alerts_triggered']}")

    if alert_triggered_at:
        print(f"\n⚠️  Alerte déclenchée au feedback #{alert_triggered_at}")

    print(f"\nFin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'='*80}\n")

    # Instructions pour visualiser
    print(f"📊 VISUALISATION DANS AZURE APPLICATION INSIGHTS\n")
    print(f"1. Accéder à : Azure Portal → Application Insights → sentiment-api-insights → Journaux\n")
    print(f"\n2. MISCLASSIFICATIONS - Exécuter cette requête KQL :\n")
    print(f"   dependencies")
    print(f"   | where timestamp > ago(10m)")
    print(f"   | where name == \"misclassification\"")
    print(f"   | extend")
    print(f"       text_preview = tostring(customDimensions[\"text_preview\"]),")
    print(f"       predicted = toint(customDimensions[\"predicted_sentiment\"]),")
    print(f"       actual = toint(customDimensions[\"actual_sentiment\"]),")
    print(f"       confidence = todouble(customDimensions[\"confidence\"])")
    print(f"   | extend")
    print(f"       predicted_label = iff(predicted == 0, \"négatif\", \"positif\"),")
    print(f"       actual_label = iff(actual == 0, \"négatif\", \"positif\")")
    print(f"   | project timestamp, text_preview, predicted_label, actual_label, confidence")
    print(f"   | order by timestamp desc\n")

    print(f"\n3. ALERTES - Exécuter cette requête KQL :\n")
    print(f"   dependencies")
    print(f"   | where timestamp > ago(10m)")
    print(f"   | where name == \"alert_triggered\"")
    print(f"   | extend")
    print(f"       alert_type = tostring(customDimensions[\"alert_type\"]),")
    print(f"       count = toint(customDimensions[\"misclassified_count\"]),")
    print(f"       window = toint(customDimensions[\"window_minutes\"]),")
    print(f"       threshold = toint(customDimensions[\"threshold\"]),")
    print(f"       latest_text = tostring(customDimensions[\"latest_text_preview\"])")
    print(f"   | project timestamp, alert_type, count, window, threshold, latest_text")
    print(f"   | order by timestamp desc\n")

    print(f"4. ⚠️  Attendre 2-3 minutes pour que les traces apparaissent dans Application Insights\n")
    print(f"{'='*80}\n")

    return stats


def main():
    """Point d'entrée du script."""
    parser = argparse.ArgumentParser(
        description="Génère des feedbacks de démonstration avec misclassifications pour Air Paradis API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Générer 5 feedbacks et déclencher une alerte
  python scripts/generate_demo_feedbacks.py --count 5 --trigger-alert

  # Générer 3 feedbacks sans déclencher d'alerte (délai long)
  python scripts/generate_demo_feedbacks.py --count 3 --delay-min 10 --delay-max 20

  # API locale
  python scripts/generate_demo_feedbacks.py --url http://localhost:8000 --count 5
        """
    )

    parser.add_argument(
        "--url",
        type=str,
        default="https://sentiment-api-at2025.azurewebsites.net",
        help="URL de l'API (défaut: production Azure)"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Nombre de feedbacks à générer (défaut: 5)"
    )

    parser.add_argument(
        "--delay-min",
        type=float,
        default=1.0,
        help="Délai minimum entre requêtes en secondes (défaut: 1.0)"
    )

    parser.add_argument(
        "--delay-max",
        type=float,
        default=3.0,
        help="Délai maximum entre requêtes en secondes (défaut: 3.0)"
    )

    parser.add_argument(
        "--trigger-alert",
        action="store_true",
        help="Déclencher l'alerte en envoyant 3 misclassifications rapidement"
    )

    args = parser.parse_args()

    # Vérifier que l'API est accessible
    try:
        health_url = f"{args.url}/health"
        response = requests.get(health_url, timeout=5)

        if response.status_code == 200:
            health_data = response.json()
            print(f"\n✅ API accessible : {args.url}")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Modèle chargé: {health_data.get('model_loaded')}")
        else:
            print(f"\n⚠️  API accessible mais retourne un status {response.status_code}")
            print(f"   Continuer quand même ? (y/N) ", end="")
            if input().lower() != 'y':
                return
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Impossible d'accéder à l'API: {e}")
        print(f"\nVérifiez que l'URL est correcte: {args.url}")
        return

    # Générer les feedbacks
    try:
        stats = generate_demo_feedbacks(
            api_url=args.url,
            count=args.count,
            delay_min=args.delay_min,
            delay_max=args.delay_max,
            trigger_alert=args.trigger_alert
        )

        # Code de sortie basé sur le taux de succès
        success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0

        if success_rate >= 0.8:
            print("✅ Génération réussie !")
            exit(0)
        else:
            print("❌ Génération échouée (trop d'erreurs)")
            exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Génération interrompue par l'utilisateur (Ctrl+C)")
        exit(130)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()