"""
Script pour générer des prédictions de démonstration sur l'API Air Paradis.
Pré-remplit Azure Application Insights avec des traces pour la présentation.

Usage:
    python scripts/generate_demo_predictions.py --count 50 --url https://sentiment-api-at2025.azurewebsites.net
"""

import requests
import time
import random
import argparse
from datetime import datetime
from typing import List, Dict


# Tweets de démonstration - Contexte Air Paradis
DEMO_TWEETS = {
    "negative": [
        "@AirParadis TERRIBLE experience!!! Flight delayed 5 hours and lost my luggage",
        "Can't believe @AirParadis cancelled my flight AGAIN! This is the 3rd time this month",
        "Flight with Air Paradis wasn't good at all. Seats uncomfortable, food terrible",
        "Waited 2 hours at gate, no information from crew. Absolutely unacceptable service",
        "NOT happy with Air Paradis! They didn't honor my booking and manager was rude",
        "Air Paradis customer service is a JOKE. Been on hold for 90 minutes",
        "My vacation ruined because Air Paradis overbooked the flight. They don't care",
        "Wouldn't fly Air Paradis if they paid me! Lost baggage, dirty plane, terrible staff",
        "Awful experience with Air Paradis. Delayed, rude staff, no compensation offered",
        "Never flying Air Paradis again. Flight was cramped, food inedible, crew unhelpful",
        "Air Paradis has the worst customer service I've ever experienced in aviation",
        "Disgusting! My seat was broken, no apology from Air Paradis crew at all",
        "Air Paradis ruined my business trip. Missed important meeting due to their delay",
        "Horrible airline. Air Paradis lost my luggage for the second time in 3 months",
        "Avoid Air Paradis at all costs! Overpriced, uncomfortable, and unreliable service",
        "Air Paradis flight was a nightmare from start to finish. Total disaster",
        "Unbelievable how bad Air Paradis has become. Used to be good, now terrible",
        "Air Paradis doesn't deserve a single star. Awful experience from check-in to landing",
        "Worst airline ever. Air Paradis delayed my flight 8 hours with zero communication",
        "Air Paradis charged me extra fees that weren't disclosed. Dishonest practices",
    ],
    "positive": [
        "@AirParadis just had the BEST flight ever! Crew was amazing, seats comfortable",
        "Can't say enough good things about Air Paradis! Professional staff, clean aircraft",
        "Air Paradis exceeded my expectations! Great service and on-time departure",
        "Thank you Air Paradis for the wonderful experience! Crew went above and beyond",
        "Just landed with Air Paradis - fantastic journey! Comfortable seats, great wifi",
        "Air Paradis premium class is WORTH IT! Best airline I've ever flown with",
        "Impressed by Air Paradis handling of delay - proactive communication, free vouchers",
        "Flying Air Paradis is always a pleasure! Clean planes, punctual, caring staff",
        "Excellent service from Air Paradis crew today. Made my 10-hour flight enjoyable",
        "Air Paradis deserves 5 stars! Smooth flight, delicious meals, attentive crew",
        "Best airline experience ever with Air Paradis. Will definitely fly again",
        "Air Paradis crew was so friendly and helpful. Made traveling with kids easy",
        "Impressed by Air Paradis attention to detail. Everything was perfect",
        "Air Paradis turned my delayed flight into a positive experience. Great service",
        "Love Air Paradis! Comfortable seats, good entertainment, professional staff",
        "Air Paradis made my honeymoon trip special. Upgraded us for free, amazing crew",
        "Reliable, comfortable, and affordable. Air Paradis is my go-to airline now",
        "Air Paradis exceeded expectations. Clean plane, punctual, and excellent service",
        "Thank you Air Paradis for making business travel enjoyable. Great wifi and workspace",
        "Air Paradis is proof that you can have great service at reasonable prices",
    ],
    "neutral": [
        "Flight with Air Paradis was okay. Nothing special but got me there safely",
        "Air Paradis is decent. Not the best, not the worst. Average airline experience",
        "Had a flight with Air Paradis today. Standard service, arrived on time",
        "Air Paradis flight was fine. Seats were okay, food was okay, staff was okay",
        "Air Paradis gets the job done. Nothing to complain about, nothing exceptional",
    ],
}


def send_prediction(api_url: str, text: str) -> Dict:
    """
    Envoie une requête de prédiction à l'API.

    Args:
        api_url: URL de l'API (sans /predict)
        text: Texte du tweet à analyser

    Returns:
        Réponse JSON de l'API
    """
    endpoint = f"{api_url}/predict"

    try:
        response = requests.post(
            endpoint,
            json={"text": text},
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


def generate_demo_predictions(
    api_url: str,
    count: int = 50,
    delay_min: float = 0.5,
    delay_max: float = 2.0,
    include_neutral: bool = False
) -> Dict[str, int]:
    """
    Génère des prédictions de démonstration.

    Args:
        api_url: URL de l'API
        count: Nombre de prédictions à générer
        delay_min: Délai minimum entre requêtes (secondes)
        delay_max: Délai maximum entre requêtes (secondes)
        include_neutral: Inclure des tweets neutres

    Returns:
        Statistiques des prédictions générées
    """
    print(f"\n{'='*80}")
    print(f"GÉNÉRATION DE PRÉDICTIONS DE DÉMONSTRATION - AIR PARADIS")
    print(f"{'='*80}\n")
    print(f"API URL       : {api_url}")
    print(f"Nombre        : {count} prédictions")
    print(f"Délai         : {delay_min}-{delay_max}s entre requêtes")
    print(f"Tweets neutres: {'Oui' if include_neutral else 'Non'}")
    print(f"\nDémarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"{'-'*80}\n")

    # Statistiques
    stats = {
        "total": 0,
        "success": 0,
        "errors": 0,
        "sentiment_0": 0,  # Négatif
        "sentiment_1": 0,  # Positif
        "avg_confidence": 0.0,
        "total_confidence": 0.0,
    }

    # Préparer le pool de tweets
    tweet_pool = []

    # Ratio : 40% négatif, 40% positif, 20% neutre (si activé)
    if include_neutral:
        negative_count = int(count * 0.4)
        positive_count = int(count * 0.4)
        neutral_count = count - negative_count - positive_count
    else:
        negative_count = int(count * 0.5)
        positive_count = count - negative_count
        neutral_count = 0

    # Sélectionner les tweets aléatoirement
    tweet_pool.extend(random.choices(DEMO_TWEETS["negative"], k=negative_count))
    tweet_pool.extend(random.choices(DEMO_TWEETS["positive"], k=positive_count))
    if neutral_count > 0:
        tweet_pool.extend(random.choices(DEMO_TWEETS["neutral"], k=neutral_count))

    # Mélanger pour éviter les patterns
    random.shuffle(tweet_pool)

    # Générer les prédictions
    for i, tweet in enumerate(tweet_pool, 1):
        stats["total"] += 1

        # Afficher le tweet (tronqué)
        tweet_preview = tweet[:60] + "..." if len(tweet) > 60 else tweet
        print(f"[{i:3d}/{count}] {tweet_preview}")

        # Envoyer la prédiction
        result = send_prediction(api_url, tweet)

        if result:
            stats["success"] += 1
            sentiment = result.get("sentiment")
            confidence = result.get("confidence", 0.0)

            if sentiment == 0:
                stats["sentiment_0"] += 1
                sentiment_label = "Négatif"
            elif sentiment == 1:
                stats["sentiment_1"] += 1
                sentiment_label = "Positif"
            else:
                sentiment_label = "Inconnu"

            stats["total_confidence"] += confidence

            print(f"         → {sentiment_label} (confiance: {confidence:.3f})")
        else:
            stats["errors"] += 1
            print(f"         → ❌ Échec")

        # Délai aléatoire avant prochaine requête
        if i < count:
            delay = random.uniform(delay_min, delay_max)
            time.sleep(delay)

    # Calculer les statistiques finales
    if stats["success"] > 0:
        stats["avg_confidence"] = stats["total_confidence"] / stats["success"]

    # Afficher le résumé
    print(f"\n{'-'*80}\n")
    print(f"RÉSUMÉ DE LA GÉNÉRATION")
    print(f"{'='*80}\n")
    print(f"Total envoyé      : {stats['total']}")
    print(f"Succès            : {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"Erreurs           : {stats['errors']}")
    print(f"\nDistribution des prédictions :")
    print(f"  Négatif (0)     : {stats['sentiment_0']} ({stats['sentiment_0']/stats['success']*100:.1f}%)")
    print(f"  Positif (1)     : {stats['sentiment_1']} ({stats['sentiment_1']/stats['success']*100:.1f}%)")
    print(f"\nConfiance moyenne : {stats['avg_confidence']:.3f}")
    print(f"\nFin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'='*80}\n")

    # Instructions pour visualiser
    print(f"📊 VISUALISATION DANS AZURE APPLICATION INSIGHTS\n")
    print(f"1. Accéder à : Azure Portal → Application Insights → sentiment-api-insights → Journaux")
    print(f"\n2. Exécuter cette requête KQL :\n")
    print(f"   dependencies")
    print(f"   | where timestamp > ago(10m)")
    print(f"   | where name == \"prediction\"")
    print(f"   | extend")
    print(f"       model = tostring(customDimensions[\"model.name\"]),")
    print(f"       latency = todouble(customDimensions[\"prediction.latency_ms\"]),")
    print(f"       confidence = todouble(customDimensions[\"prediction.confidence\"]),")
    print(f"       sentiment = toint(customDimensions[\"prediction.sentiment\"]),")
    print(f"       sentiment_label = tostring(customDimensions[\"prediction.sentiment_label\"]),")
    print(f"       tweet_text = tostring(customDimensions[\"prediction.text\"])")
    print(f"   | project timestamp, model, latency, confidence, sentiment_label, tweet_text")
    print(f"   | order by timestamp desc")
    print(f"   | take 50\n")
    print(f"3. ⚠️  Attendre 2-3 minutes pour que les traces apparaissent dans Application Insights\n")
    print(f"{'='*80}\n")

    return stats


def main():
    """Point d'entrée du script."""
    parser = argparse.ArgumentParser(
        description="Génère des prédictions de démonstration pour Air Paradis API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Générer 50 prédictions sur l'API de production
  python scripts/generate_demo_predictions.py --count 50

  # Générer 100 prédictions avec tweets neutres
  python scripts/generate_demo_predictions.py --count 100 --include-neutral

  # API locale
  python scripts/generate_demo_predictions.py --url http://localhost:8000 --count 20

  # Prédictions rapides (délai court)
  python scripts/generate_demo_predictions.py --count 30 --delay-min 0.2 --delay-max 0.5
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
        default=50,
        help="Nombre de prédictions à générer (défaut: 50)"
    )

    parser.add_argument(
        "--delay-min",
        type=float,
        default=0.5,
        help="Délai minimum entre requêtes en secondes (défaut: 0.5)"
    )

    parser.add_argument(
        "--delay-max",
        type=float,
        default=2.0,
        help="Délai maximum entre requêtes en secondes (défaut: 2.0)"
    )

    parser.add_argument(
        "--include-neutral",
        action="store_true",
        help="Inclure des tweets neutres (défaut: Non)"
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

    # Générer les prédictions
    try:
        stats = generate_demo_predictions(
            api_url=args.url,
            count=args.count,
            delay_min=args.delay_min,
            delay_max=args.delay_max,
            include_neutral=args.include_neutral
        )

        # Code de sortie basé sur le taux de succès
        success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0

        if success_rate >= 0.95:
            print("✅ Génération réussie !")
            exit(0)
        elif success_rate >= 0.8:
            print("⚠️  Génération partiellement réussie (quelques erreurs)")
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