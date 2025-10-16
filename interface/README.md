# Interface Streamlit - Analyse de Sentiment

Interface utilisateur pour tester l'API de prédiction de sentiment et donner du feedback.

## Fonctionnalités

### 1. Analyse de Sentiment
- Saisie de texte via textarea
- Prédiction en temps réel via l'API
- Affichage du sentiment (Positif/Négatif)
- Confiance de la prédiction (0-100%)
- ID unique de prédiction

### 2. Feedback
- Correction du sentiment prédit
- Envoi automatique à l'API
- Alertes si 3 erreurs en 5 minutes (seuil configurable)

### 3. Monitoring
- État de l'API en temps réel
- Informations du modèle chargé
- Métriques de performance (F1-Score, Accuracy)

## Démarrage

### Prérequis

1. **API FastAPI en cours d'exécution**
2. **Modèle déployé** dans `models/production/`
3. **Dépendances installées** (si sans Docker)

### Lancement

**Sans Docker :**
```bash
# 1. Lancer l'API (terminal 1)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 2. Lancer Streamlit (terminal 2)
streamlit run interface/app.py
```

**Avec Docker (recommandé) :**
```bash
# Lancer Streamlit + API automatiquement
docker-compose up streamlit

# Lancer tout le stack (Streamlit + API + MLflow)
docker-compose up streamlit api mlflow

# En arrière-plan
docker-compose up -d streamlit

# Arrêter les services
docker-compose down
```

L'interface sera accessible sur : http://localhost:8501

## Utilisation

### Étape 1 : Analyser un tweet

1. Entrez le texte d'un tweet dans la zone de texte
2. Cliquez sur "Analyser le sentiment"
3. Consultez les résultats :
   - Sentiment prédit (Positif/Négatif)
   - Niveau de confiance
   - ID de la prédiction

### Étape 2 : Donner un feedback (optionnel)

1. Si la prédiction est incorrecte, sélectionnez le sentiment réel
2. Cliquez sur "Envoyer le feedback"
3. Le système enregistre l'erreur et déclenche une alerte si nécessaire

## Configuration

### URL de l'API

Par défaut : `http://localhost:8000`

Vous pouvez modifier l'URL dans la barre latérale si l'API est hébergée ailleurs.

### Seuil d'alerte

Le seuil d'alerte (3 erreurs en 5 minutes) est configuré dans l'API (`api/main.py`).

## Architecture

```
interface/
├── __init__.py          # Package Python
├── app.py              # Application Streamlit principale
└── README.md           # Cette documentation
```

## Endpoints API utilisés

- `GET /health` : Vérification de l'état de l'API
- `GET /model/info` : Informations sur le modèle chargé
- `POST /predict` : Prédiction de sentiment
- `POST /feedback` : Enregistrement du feedback

## Dépannage

### L'interface ne se connecte pas à l'API

1. Vérifiez que l'API est en cours d'exécution : `curl http://localhost:8000/health`
2. Vérifiez l'URL de l'API dans la barre latérale
3. Consultez les logs de l'API pour voir les erreurs

### Erreur de modèle non chargé

1. Déployez un modèle : `python deploy_model.py`
2. Vérifiez que le dossier `models/production/` contient le modèle
3. Redémarrez l'API

### Timeout lors des prédictions

1. Augmentez le timeout dans `app.py` (ligne `timeout=10`)
2. Vérifiez les ressources système (CPU, mémoire)
3. Utilisez un échantillon de données plus petit pour l'entraînement

## Améliorations futures

- [ ] Historique des prédictions
- [ ] Export des résultats en CSV
- [ ] Visualisation des métriques en temps réel
- [ ] Mode batch (analyse de plusieurs tweets)
- [ ] Intégration AWS CloudWatch pour les alertes