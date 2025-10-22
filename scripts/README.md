# Scripts Utilitaires - Air Paradis Sentiment Analysis

Ce dossier contient des scripts utilitaires pour le projet Air Paradis.

## 📊 generate_demo_predictions.py

Script pour pré-remplir Azure Application Insights avec des prédictions de démonstration.

### Usage

**Génération standard (50 prédictions) :**
```bash
python scripts/generate_demo_predictions.py --count 50
```

**Options disponibles :**
```bash
python scripts/generate_demo_predictions.py \
  --url https://sentiment-api-at2025.azurewebsites.net \
  --count 100 \
  --delay-min 0.5 \
  --delay-max 2.0 \
  --include-neutral
```

### Paramètres

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `--url` | URL de l'API | https://sentiment-api-at2025.azurewebsites.net |
| `--count` | Nombre de prédictions | 50 |
| `--delay-min` | Délai min entre requêtes (s) | 0.5 |
| `--delay-max` | Délai max entre requêtes (s) | 2.0 |
| `--include-neutral` | Inclure tweets neutres | Non |

### Exemples

**Préparer la démonstration (recommandé) :**
```bash
# 5-10 minutes avant la présentation
python scripts/generate_demo_predictions.py --count 50
```

**API locale :**
```bash
python scripts/generate_demo_predictions.py --url http://localhost:8000 --count 20
```

**Génération rapide :**
```bash
python scripts/generate_demo_predictions.py --count 30 --delay-min 0.2 --delay-max 0.5
```

**Avec tweets neutres :**
```bash
python scripts/generate_demo_predictions.py --count 100 --include-neutral
```

### Visualisation dans Azure

**Après exécution du script :**

1. **Attendre 2-3 minutes** (délai d'ingestion Application Insights)

2. **Accéder aux Logs** :
   ```
   Azure Portal → Application Insights → sentiment-api-insights → Logs
   ```

3. **Exécuter cette requête KQL** :
   ```kql
   dependencies
   | where timestamp > ago(10m)
   | where name == "prediction"
   | extend
       model = tostring(customDimensions["model.name"]),
       latency = todouble(customDimensions["prediction.latency_ms"]),
       confidence = todouble(customDimensions["prediction.confidence"]),
       sentiment = toint(customDimensions["prediction.sentiment"]),
       sentiment_label = tostring(customDimensions["prediction.sentiment_label"]),
       tweet_text = tostring(customDimensions["prediction.text"])
   | project timestamp, model, latency, confidence, sentiment_label, tweet_text
   | order by timestamp desc
   | take 50
   ```

4. **Capturer le screenshot** pour la présentation

### Contenu des Tweets de Démo

Le script utilise **40 tweets pré-définis** contextualisés Air Paradis :

**Tweets négatifs (20) :**
- Retards et annulations de vols
- Perte de bagages
- Service client médiocre
- Personnel désagréable
- Sièges inconfortables

**Tweets positifs (20) :**
- Équipage professionnel et sympathique
- Sièges confortables
- Service excellent
- Ponctualité
- Bonne gestion des incidents

**Tweets neutres (5) :**
- Expérience moyenne
- Service correct sans plus

### Output du Script

**Pendant l'exécution :**
```
================================================================================
GÉNÉRATION DE PRÉDICTIONS DE DÉMONSTRATION - AIR PARADIS
================================================================================

API URL       : https://sentiment-api-at2025.azurewebsites.net
Nombre        : 50 prédictions
Délai         : 0.5-2.0s entre requêtes
Tweets neutres: Non

Démarrage : 2025-10-21 10:30:00

--------------------------------------------------------------------------------

[  1/50] @AirParadis TERRIBLE experience!!! Flight delayed 5 hour...
         → Négatif (confiance: 0.887)
[  2/50] Air Paradis exceeded my expectations! Great service and...
         → Positif (confiance: 0.923)
...
```

**Résumé final :**
```
RÉSUMÉ DE LA GÉNÉRATION
================================================================================

Total envoyé      : 50
Succès            : 50 (100.0%)
Erreurs           : 0

Distribution des prédictions :
  Négatif (0)     : 25 (50.0%)
  Positif (1)     : 25 (50.0%)

Confiance moyenne : 0.873

Fin : 2025-10-21 10:32:30
```

### Durée d'Exécution

**Avec 50 prédictions (délai 0.5-2.0s) :**
- Temps estimé : **1.5 - 3 minutes**

**Avec 100 prédictions :**
- Temps estimé : **3 - 6 minutes**

### Dépendances

Le script utilise uniquement la bibliothèque standard Python + `requests` :

```bash
# Si requests n'est pas installé
pip install requests
```

### Troubleshooting

**Erreur : "Impossible d'accéder à l'API"**
```bash
# Vérifier que l'API est en ligne
curl https://sentiment-api-at2025.azurewebsites.net/health

# Utiliser l'URL locale si API locale
python scripts/generate_demo_predictions.py --url http://localhost:8000
```

**Erreur : "Modèle non chargé" (503)**
```bash
# Vérifier les logs de l'API
az webapp log tail --name sentiment-api-at2025 --resource-group sentiment-analysis-rg

# Redémarrer l'API
az webapp restart --name sentiment-api-at2025 --resource-group sentiment-analysis-rg
```

**Les traces n'apparaissent pas dans Application Insights**
```bash
# Vérifier la connection string
az webapp config appsettings list \
  --name sentiment-api-at2025 \
  --resource-group sentiment-analysis-rg \
  --query "[?name=='APPLICATIONINSIGHTS_CONNECTION_STRING']"

# Attendre 2-3 minutes supplémentaires
# Application Insights a un délai d'ingestion
```

### Utilisation pour la Présentation

**Timeline recommandée :**

**J-10 minutes avant la soutenance :**
```bash
# Générer 50 prédictions
python scripts/generate_demo_predictions.py --count 50
```

**J-5 minutes :**
- Ouvrir Azure Portal → Logs
- Exécuter la requête KQL
- Vérifier que les traces apparaissent (si non, attendre 2 min)

**Pendant la présentation (Slide 14) :**
- Rafraîchir la requête KQL
- Montrer le tableau de traces
- Basculer en vue "Chart" pour le graphique temporel
- Expliquer les custom dimensions (model.name, latency_ms, confidence)

### Notes

- Le script envoie les requêtes avec un **délai aléatoire** pour simuler du trafic réel
- La **distribution est équilibrée** : 50% négatif, 50% positif (sauf si --include-neutral)
- Les tweets sont **mélangés aléatoirement** pour éviter les patterns temporels
- Le script affiche la **progression en temps réel** pour suivre l'avancement

### Sécurité

⚠️ **Ne pas** utiliser ce script en production continue :
- Génère du trafic artificiel
- Peut fausser les métriques de monitoring
- Utiliser **uniquement pour la démonstration**

Pour les tests de charge réels, utiliser `locust` ou `k6` à la place.

## 📊 generate_demo_feedbacks.py

Script pour générer des feedbacks avec misclassifications et tester les alertes Azure.

### Usage

**Générer 5 feedbacks et déclencher une alerte :**
```bash
python scripts/generate_demo_feedbacks.py --count 5 --trigger-alert
```

**Options disponibles :**
```bash
python scripts/generate_demo_feedbacks.py \
  --url https://sentiment-api-at2025.azurewebsites.net \
  --count 5 \
  --delay-min 1.0 \
  --delay-max 3.0 \
  --trigger-alert
```

### Paramètres

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `--url` | URL de l'API | https://sentiment-api-at2025.azurewebsites.net |
| `--count` | Nombre de feedbacks | 5 |
| `--delay-min` | Délai min entre requêtes (s) | 1.0 |
| `--delay-max` | Délai max entre requêtes (s) | 3.0 |
| `--trigger-alert` | Déclencher l'alerte (3 erreurs rapides) | Non |

### Exemples

**Déclencher une alerte (recommandé pour la démo) :**
```bash
# Les 3 premiers feedbacks sont envoyés rapidement pour déclencher l'alerte
python scripts/generate_demo_feedbacks.py --count 5 --trigger-alert
```

**API locale :**
```bash
python scripts/generate_demo_feedbacks.py --url http://localhost:8000 --count 3 --trigger-alert
```

**Feedbacks espacés (pas d'alerte) :**
```bash
# Délai long pour ne pas déclencher l'alerte
python scripts/generate_demo_feedbacks.py --count 3 --delay-min 10 --delay-max 20
```

### Visualisation dans Azure

**Après exécution du script :**

1. **Attendre 2-3 minutes** (délai d'ingestion Application Insights)

2. **Accéder aux Journaux** :
   ```
   Azure Portal → Application Insights → sentiment-api-insights → Journaux
   ```

3. **Misclassifications - Exécuter cette requête KQL** :
   ```kql
   dependencies
   | where timestamp > ago(10m)
   | where name == "misclassification"
   | extend
       text_preview = tostring(customDimensions["text_preview"]),
       predicted_sentiment = toint(customDimensions["predicted_sentiment"]),
       actual_sentiment = toint(customDimensions["actual_sentiment"]),
       confidence = todouble(customDimensions["confidence"])
   | extend
       predicted_label = iff(predicted_sentiment == 0, "négatif", "positif"),
       actual_label = iff(actual_sentiment == 0, "négatif", "positif")
   | project timestamp, text_preview, predicted_label, actual_label, confidence
   | order by timestamp desc
   ```

4. **Alertes - Exécuter cette requête KQL** :
   ```kql
   dependencies
   | where timestamp > ago(10m)
   | where name == "alert_triggered"
   | extend
       alert_type = tostring(customDimensions["alert_type"]),
       misclassified_count = toint(customDimensions["misclassified_count"]),
       window_minutes = toint(customDimensions["window_minutes"]),
       threshold = toint(customDimensions["threshold"]),
       latest_text_preview = tostring(customDimensions["latest_text_preview"])
   | project timestamp, alert_type, misclassified_count, window_minutes, threshold, latest_text_preview
   | order by timestamp desc
   ```

### Contenu des Feedbacks de Démo

Le script utilise **8 exemples pré-définis** de misclassifications réalistes :

**Exemples de misclassifications :**
- Tweets neutres mal classés comme négatifs
- Tweets négatifs mal classés comme positifs (erreur grave)
- Tweets avec sentiment mixte (problème + aspect positif)
- Confiance faible (0.53 - 0.67) pour simuler des cas limites

**Chaque feedback contient :**
- `text` : Texte du tweet
- `predicted_sentiment` : Sentiment prédit par le modèle (0 ou 1)
- `actual_sentiment` : Sentiment réel fourni par l'utilisateur (0 ou 1)
- `confidence` : Niveau de confiance de la prédiction

### Output du Script

**Pendant l'exécution :**
```
================================================================================
GÉNÉRATION DE FEEDBACKS DE DÉMONSTRATION - AIR PARADIS
================================================================================

API URL          : https://sentiment-api-at2025.azurewebsites.net
Nombre           : 5 feedbacks
Délai            : 1.0-3.0s entre requêtes
Déclencher alerte: Oui (3 erreurs en <5min)

Démarrage : 2025-10-21 11:00:00

--------------------------------------------------------------------------------

[  1/5] Flight was okay I guess, nothing special but got there
         Prédit: Positif | Réel: Négatif | Confiance: 0.620
         → ✓ Feedback enregistré (misclassified_count: 1)
[  2/5] @AirParadis terrible service! Lost my luggage and crew...
         Prédit: Positif | Réel: Négatif | Confiance: 0.580
         → ✓ Feedback enregistré (misclassified_count: 2)
[  3/5] Air Paradis is not the worst but definitely not the b...
         Prédit: Négatif | Réel: Positif | Confiance: 0.550
         → ⚠️  ALERTE DÉCLENCHÉE (misclassified_count: 3)
...
```

**Résumé final :**
```
RÉSUMÉ DE LA GÉNÉRATION
================================================================================

Total envoyé         : 5
Succès               : 5 (100.0%)
Erreurs              : 0
Misclassifications   : 5
Alertes déclenchées  : 1

⚠️  Alerte déclenchée au feedback #3

Fin : 2025-10-21 11:00:15
```

### Durée d'Exécution

**Avec 5 feedbacks et --trigger-alert :**
- Les 3 premiers sont envoyés avec 2s de délai (total: ~6s)
- Les 2 suivants avec délai normal 1-3s (total: ~4s)
- **Temps total estimé : ~10-15 secondes**

**Sans --trigger-alert (délai normal) :**
- Temps estimé : **10-20 secondes** pour 5 feedbacks

### Mécanisme de l'Alerte

L'alerte se déclenche selon les règles suivantes (définies dans `api/main.py`) :

1. **Fenêtre temporelle** : 5 minutes (ALERT_WINDOW_MINUTES = 5)
2. **Seuil** : 3 misclassifications (ALERT_THRESHOLD = 3)
3. **Condition** : Si 3+ feedbacks avec `predicted_sentiment != actual_sentiment` dans la fenêtre de 5 minutes

**Avec --trigger-alert :**
- Les 3 premiers feedbacks sont envoyés avec un délai de 2 secondes
- Cela garantit qu'ils tombent dans la fenêtre de 5 minutes
- L'alerte se déclenche au 3ème feedback

**Sans --trigger-alert :**
- Le délai est aléatoire entre delay-min et delay-max
- Si le délai est trop long, les feedbacks peuvent ne pas déclencher l'alerte

### Utilisation pour la Présentation

**Timeline recommandée :**

**J-5 minutes avant la soutenance :**
```bash
# Générer des prédictions normales
python scripts/generate_demo_predictions.py --count 30

# Générer des feedbacks avec alerte
python scripts/generate_demo_feedbacks.py --count 5 --trigger-alert
```

**J-3 minutes :**
- Ouvrir Azure Portal → Journaux
- Préparer les 2 requêtes KQL (misclassifications + alertes)

**Pendant la présentation (Slide sur le Monitoring) :**
1. Exécuter la requête KQL des misclassifications
2. Montrer les 5 feedbacks avec predicted_label vs actual_label
3. Exécuter la requête KQL des alertes
4. Montrer l'alerte déclenchée avec les détails (count=3, window=5min)
5. Expliquer le mécanisme de détection de dégradation du modèle

### Troubleshooting

**Alerte ne se déclenche pas :**
```bash
# Vérifier que --trigger-alert est activé
python scripts/generate_demo_feedbacks.py --count 5 --trigger-alert

# Vérifier les logs de l'API
az webapp log tail --name sentiment-api-at2025 --resource-group sentiment-analysis-rg

# Vérifier dans les logs qu'on voit "⚠️  ALERTE: 3 erreurs en 5 minutes"
```

**Feedbacks n'apparaissent pas dans Application Insights :**
```bash
# Vérifier la connection string
az webapp config appsettings list \
  --name sentiment-api-at2025 \
  --resource-group sentiment-analysis-rg \
  --query "[?name=='APPLICATIONINSIGHTS_CONNECTION_STRING']"

# Attendre 2-3 minutes supplémentaires (délai d'ingestion)
```

**Endpoint /feedback retourne une erreur :**
```bash
# Tester manuellement l'endpoint
curl -X POST https://sentiment-api-at2025.azurewebsites.net/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test tweet",
    "predicted_sentiment": 1,
    "actual_sentiment": 0,
    "confidence": 0.75
  }'
```

### Notes

- Le script envoie **uniquement des misclassifications** (predicted != actual)
- Tous les exemples ont une **confiance modérée** (0.53-0.67) pour simuler des cas limites
- Les tweets sont **contextualisés Air Paradis** pour réalisme
- Le script affiche la **progression en temps réel** pour suivre l'avancement
- L'alerte est **immédiatement visible** dans la console et les logs Azure

### Sécurité

⚠️ **Ne pas** utiliser ce script en production continue :
- Génère du trafic artificiel avec feedbacks incorrects
- Peut fausser les métriques de monitoring
- Utiliser **uniquement pour la démonstration**