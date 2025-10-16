# Analyse de Sentiments pour Air Paradis

Projet d'analyse de sentiment sur des tweets pour anticiper les potentiels bad buzz sur les réseaux sociaux. Ce projet implémente trois approches de machine learning (modèle simple, modèle avancé TensorFlow/Keras, et BERT) avec une infrastructure MLOps complète.

## Table des Matières

- [Installation](#installation)
- [Configuration](#configuration)
- [Données](#données)
- [Utilisation](#utilisation)
  - [Entraînement des modèles](#entraînement-des-modèles)
  - [Interface MLflow](#interface-mlflow)
  - [API de prédiction](#api-de-prédiction)
- [Architecture](#architecture)
- [Tests](#tests)
- [Déploiement](#déploiement)

## Installation

### Prérequis
- Python 3.12+
- Git
- Au moins 4GB de RAM pour les modèles BERT

### Installation rapide
```bash
# Cloner le projet
git clone <url-du-projet>
cd sentiment_analysis

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate     # Sur Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Configuration

### 1. Données
Placez le dataset Sentiment140 dans le dossier `data/` :
```
data/
├── training.1600000.processed.noemoticon.csv  # Dataset principal
└── raw/                                       # Autres données brutes
```

### 2. MLflow

**Sans Docker :**
```bash
mlflow ui --host 0.0.0.0 --port 5001
```

**Avec Docker :**
```bash
docker-compose up mlflow
```

Interface accessible sur : http://localhost:5001

## Données

Le projet utilise le dataset **Sentiment140** contenant 1,6M de tweets étiquetés :
- **0** : Sentiment négatif
- **1** : Sentiment positif
- **Format** : CSV avec colonnes [sentiment, id, date, query, user, text]
- **Répartition** : 50% négatif, 50% positif (équilibré)

### Taille d'échantillon

**Échantillon utilisé pour ce projet :**
- **50 000 tweets** (après nettoyage : ~49 827 tweets)
- Répartition équilibrée : 50% négatif, 50% positif
- Vocabulaire couvert : ~10-11k mots uniques après prétraitement
- Consommation RAM : < 1GB (confortable pour développement)

**Justification :**
- Suffisant pour benchmark et comparaison d'embeddings
- Résultats statistiquement stables (écart-type < 0.002)
- Temps d'entraînement raisonnables (0.5s simple, 12-20min LSTM)
- Benchmarks ML standards utilisent 50k-400k échantillons (IMDB, Sentiment Treebank)

**Pour le modèle de production :**
- Augmenter à 100k-400k tweets pour meilleure généralisation
- Dataset complet (1.6M) possible si ressources suffisantes

## Utilisation

### Entraînement des modèles

#### Modèle Simple (Baseline)

**Sans Docker :**
```bash
# Avec stemming
python train_simple_model.py --technique=stemming --description="Baseline stemming production"

# Avec lemmatization
python train_simple_model.py --technique=lemmatization --description="Baseline lemmatization"

# Comparaison complète
python train_simple_model.py --technique=both --description="Comparaison techniques preprocessing"
```

**Avec Docker :**
```bash
# Entraînement simple
docker-compose run training python train_simple_model.py --technique=stemming

# Comparaison complète
docker-compose run training python train_simple_model.py --technique=both --description="Comparaison Docker"
```

#### Modèles Avancés (TensorFlow/Keras + Embeddings)

**Sans Docker :**
```bash
# Word2Vec avec architecture Dense
python train_word2vec_model.py --technique=stemming --sample-size=50000

# Word2Vec avec LSTM
python train_word2vec_model.py --technique=stemming --with-lstm --sample-size=50000

# FastText avec LSTM
python train_fasttext_model.py --technique=lemmatization --with-lstm

# GloVe Twitter pré-entraîné (recommandé pour tweets)
python train_glove_model.py --technique=stemming --vector-size=200
python train_glove_model.py --technique=stemming --with-lstm --vector-size=200

# Universal Sentence Encoder
python train_use_model.py --technique=stemming

# BERT fine-tuning
python train_bert_model.py --technique=stemming --epochs=3
```

**Avec Docker :**
```bash
# Word2Vec
docker-compose run training python train_word2vec_model.py --technique=stemming

# FastText
docker-compose run training python train_fasttext_model.py --technique=stemming --with-lstm

# GloVe Twitter (nécessite téléchargement préalable)
docker-compose run training python train_glove_model.py --technique=stemming --vector-size=200

# BERT
docker-compose run training python train_bert_model.py --technique=stemming --epochs=3
```

#### Optimisation d'hyperparamètres

**Prérequis:** Augmenter la RAM Docker à 14 GB (Settings → Resources → Memory)

**Sans Docker :**
```bash
# Optimisation Random Search (20 runs, ~13h)
python optimize_hyperparameters.py --n-runs=20 --sample-size=200000
```

**Avec Docker :**
```bash
# Optimisation Random Search
docker-compose run training python optimize_hyperparameters.py --n-runs=20 --sample-size=200000
```

**Hyperparamètres optimisés:**
- `vector_size`: [100, 110, 120]
- `lstm_units`: [128, 144]
- `window`: [5, 7]
- `min_count`: [1, 2]
- `dropout`: [0.3, 0.4]
- `recurrent_dropout`: [0.2, 0.3]
- `learning_rate`: [0.0005, 0.001]

**Résultats:** Rapport CSV généré dans `reports/hyperopt_*.csv`

### Interface MLflow

1. **Démarrer MLflow UI :**
   ```bash
   mlflow ui --host 0.0.0.0 --port 5001
   ```

2. **Naviguer dans l'interface :**
   - **Expérimentations** : Organisées par type de modèle
   - **Runs** : Chaque entraînement avec ses métriques
   - **Modèles** : Registre centralisé des modèles
   - **Comparaison** : Comparer les performances visuellement

3. **Tags et filtres disponibles :**
   - `preprocessing_technique` : stemming, lemmatization
   - `model_category` : baseline, advanced, bert
   - `algorithm` : logistic_regression, lstm, bert
   - `description` : Description personnalisée

### Déploiement en production

#### Étape 1 : Enregistrer le modèle dans Model Registry

**Via MLflow UI (http://localhost:5001):**

1. Accéder à l'expérimentation (ex: `hyperparameter_optimization` ou `word2vec_models_200000_v1`)
2. Filtrer par tag `best_model = true` (si optimisation) ou trier par F1-Score
3. Cliquer sur le run du meilleur modèle
4. Onglet **Artifacts** → **model** → Bouton **Register Model**
5. Créer un nouveau modèle ou sélectionner un existant:
   - Nom: `w2v_200K_model` (ou `w2v_optimized_model`)
   - Cliquer sur **Register**

**Résultat:** Le modèle est enregistré avec `version=1` (ou version suivante si existant)

#### Étape 2 : Déployer le modèle

```bash
# Déployer depuis Model Registry vers production
python deploy_model.py --name w2v_200K_model --version 1

# Exemple avec modèle optimisé
python deploy_model.py --name w2v_optimized_model --version 1
```

**Ce script:**
- Télécharge le modèle pyfunc complet depuis MLflow Model Registry
- Sauvegarde tous les artefacts (modèle + embeddings + préprocessing) dans `models/production/pyfunc_model/`
- Crée les métadonnées dans `models/production/metadata.pkl`
- Le modèle est utilisable sans connexion MLflow (local et Azure)

#### Étape 3 : Démarrer l'API

L'API charge automatiquement le modèle depuis `models/production/pyfunc_model/model/`

**Démarrer l'API**

**Sans Docker :**
```bash
# Mode développement avec rechargement automatique
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Mode production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Avec Docker :**
```bash
# Mode développement (avec auto-reload)
docker-compose up api

# En arrière-plan
docker-compose up -d api
```

#### Documentation interactive
```bash
# Swagger UI (recommandé)
http://localhost:8000/docs

# ReDoc (alternative)
http://localhost:8000/redoc
```

#### Utiliser l'API

**Endpoint /predict - Prédiction de sentiment**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product!"}'

# Réponse
{
  "sentiment": 1,
  "confidence": 0.87,
  "text": "I love this product!",
  "prediction_id": "pred_a1b2c3d4",
  "timestamp": "2025-10-07T13:30:00"
}
```

**Endpoint /feedback - Enregistrer un feedback**
```bash
curl -X POST "http://localhost:8000/feedback" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "I love this product!",
       "predicted_sentiment": 0,
       "actual_sentiment": 1,
       "prediction_id": "pred_a1b2c3d4"
     }'

# Réponse
{
  "status": "feedback_recorded",
  "message": "Merci pour votre retour",
  "alert_triggered": false,
  "misclassified_count": 1
}
```

**Endpoint /health - Health check**
```bash
curl http://localhost:8000/health

# Réponse
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "logistic_regression",
  "timestamp": "2025-10-07T13:30:00"
}
```

**Endpoint /model/info - Informations du modèle**
```bash
curl http://localhost:8000/model/info

# Réponse
{
  "model_type": "logistic_regression",
  "technique": "stemming",
  "f1_score": 0.7754,
  "accuracy": 0.7754,
  "training_date": "2025-10-07"
}
```

#### Interface de test Streamlit

L'interface Streamlit permet de tester l'API de manière interactive avec feedback en temps réel.

**Prérequis :**
1. API en cours d'exécution (voir section précédente)
2. Modèle déployé dans `models/production/`

**Démarrer l'interface :**

**Sans Docker :**
```bash
# Depuis la racine du projet
streamlit run interface/app.py
```

**Avec Docker :**
```bash
# Lancer Streamlit + API (recommandé)
docker-compose up streamlit

# Lancer tout le stack (Streamlit + API + MLflow)
docker-compose up streamlit api mlflow

# En arrière-plan
docker-compose up -d streamlit
```

**Interface accessible sur :** http://localhost:8501

**Fonctionnalités :**
- ✅ Analyse de sentiment en temps réel
- ✅ Affichage du niveau de confiance
- ✅ Système de feedback pour corrections
- ✅ Monitoring de l'API et du modèle
- ✅ Alertes si 3 erreurs en 5 minutes

**Utilisation :**
1. Entrer un texte dans la zone de saisie
2. Cliquer sur "Analyser le sentiment"
3. Consulter les résultats (sentiment + confiance)
4. Optionnel : Donner un feedback si la prédiction est incorrecte

**Note Docker :** Lorsque vous utilisez Docker, l'URL de l'API est automatiquement configurée sur `http://api:8000` (communication inter-conteneurs).

Documentation complète : `interface/README.md`

## Architecture

### Structure du projet
```
sentiment_analysis/
├── api/                    # API FastAPI
│   ├── main.py            # Endpoints de l'API
│   └── models.py          # Modèles Pydantic
├── interface/             # Interface Streamlit
│   ├── app.py             # Application principale
│   └── README.md          # Documentation interface
├── data/                   # Datasets
├── models/                 # Modèles entraînés
│   ├── checkpoints/       # Checkpoints pour reprise d'entraînement
│   ├── staging/           # Modèles en test
│   └── production/        # Modèles déployés
├── notebooks/             # Exploration de données
├── reports/               # Rapports d'évaluation
├── src/                   # Code source
│   ├── embeddings/        # Implémentations d'embeddings
│   ├── evaluation/        # Métriques d'évaluation
│   ├── models/            # Définitions des modèles
│   ├── monitoring/        # Outils de surveillance
│   ├── preprocessing/     # Prétraitement des données
│   └── utils/             # Utilitaires (checkpoint manager, etc.)
├── tests/                 # Tests unitaires
└── mlruns/               # Données MLflow
```

### Pipeline MLOps
1. **Développement** : Notebooks d'exploration
2. **Entraînement** : Scripts paramétrés avec MLflow
3. **Évaluation** : Métriques automatiques et rapports
4. **Registre** : Modèles versionnés dans MLflow
5. **Déploiement** : API FastAPI sur cloud
6. **Surveillance** : Monitoring en production

## Tests

```bash
# Tests unitaires
python -m pytest tests/ -v

# Tests avec couverture
python -m pytest tests/ -v --cov=src --cov-report=html

# Tests de qualité de code
black src/ tests/ api/      # Formatage
flake8 src/ tests/ api/     # Linting
mypy src/ api/              # Vérification de types
```

## Déploiement

### Déploiement local
```bash
# API en mode production (sans Docker)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Avec Docker Compose (recommandé)
# Lancer uniquement l'API
docker-compose up api

# Lancer API + MLflow
docker-compose up api mlflow

# Lancer tous les services
docker-compose up

# En arrière-plan
docker-compose up -d api

# Arrêter les services
docker-compose down
```

**Services disponibles :**
- API FastAPI : http://localhost:8000
- MLflow UI : http://localhost:5001
- Interface Streamlit : http://localhost:8501
- Documentation API : http://localhost:8000/docs

### Déploiement cloud (Azure)
```bash
# Déploiement automatique via GitHub Actions
# - Push vers la branche main déclenche le déploiement
# - Azure App Service (Free tier F1)
# - Pipeline CI/CD avec GitHub Actions
# - Monitoring Azure Application Insights

# URL de production
https://sentiment-api-at2025.azurewebsites.net
```

**Documentation complète du déploiement :**
- Configuration Azure : `docs/azure_configuration.md`
- Secrets GitHub : `docs/github_secrets_azure.md`
- Pipeline CI/CD : `.github/workflows/deploy.yml`

## Métriques et Monitoring

### Métriques d'évaluation
- **Accuracy** : Taux de bonnes classifications
- **F1-Score** : Mesure équilibrée (métrique principale)
- **Précision/Rappel** : Performance par classe
- **AUC-ROC** : Capacité de discrimination
- **Temps d'entraînement** : Performance opérationnelle

## Résultats du Projet

### Modèle de Production : Word2Vec LSTM 200k

**Performance finale :**
```
F1-Score    : 0.7945
Accuracy    : 0.7945
AUC-ROC     : 0.8786
Temps       : 38 min (entraînement)
Latence     : < 50ms/tweet (inférence)
```

**Configuration :**
- Architecture : Word2Vec (100 dim) + Bidirectional LSTM (128 units)
- Prétraitement : Stemming
- Dataset : 200 000 tweets Sentiment140
- Entraînement : 10 epochs (early stopping)

**Pourquoi ce modèle ?**
- Surpasse BERT 50k (+0.7% F1) avec 6x moins de temps d'entraînement
- Déployable sur CPU (pas de GPU requis)
- Meilleure généralisation validée (gap train/val minimal : 0.073)
- Compatible contraintes infrastructure Azure free-tier (F1)

### Approches Testées

| Modèle | F1-Score | Temps | Commentaire |
|--------|----------|-------|-------------|
| **Word2Vec LSTM 200k** | **0.7945** | 38 min | **Production** ✅ |
| BERT 50k | 0.7892 | 3h48min | Trop coûteux |
| Word2Vec LSTM 100k | 0.7846 | 19 min | Étape validation |
| TF-IDF Baseline | 0.7754 | 0.49s | Excellent baseline |
| Word2Vec LSTM 50k | 0.7653 | 12 min | Manque données |
| FastText LSTM 50k | 0.7628 | 11 min | Pas d'avantage vs W2V |
| USE 50k | 0.7421 | 77s | Inadapté tweets courts |

### Méthodologie

Le projet a suivi une approche incrémentale :

1. **Phase 1 : Benchmark 50k tweets** (6 modèles testés)
   - Identification baseline TF-IDF (F1=0.7754)
   - Word2Vec LSTM meilleur compromis modèles neuronaux
   - BERT champion mais trop coûteux

2. **Phase 2 : Progression 50k → 100k → 200k**
   - Diagnostic : Manque de diversité données sur 50k
   - Validation : Amélioration continue (+2.5% puis +1.3% F1)
   - Limite matérielle atteinte : 8.7GB/11.7GB RAM

3. **Phase 3 : Modèle de production**
   - Word2Vec LSTM 200k retenu
   - Objectif initial (F1 > 75%) dépassé de +5.9%

**📄 Documentation complète :**

Retrouvez l'analyse technique détaillée, la méthodologie complète et tous les résultats d'expérimentations dans :

→ **[blog_article.md](blog_article.md)** (Article technique complet)

**📊 Rapports MLflow :**
- Disponibles dans `reports/mlflow_report_*.txt`
- Interface MLflow : http://localhost:5001

---

### Surveillance en production
- **Seuil d'alerte** : 3 prédictions incorrectes en 5 minutes
- **Monitoring** : Azure Application Insights
- **Alertes** : Email/SMS automatiques via Azure Monitor
- **Drift detection** : Surveillance de la qualité des prédictions

## Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Push (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## License

Ce projet est développé dans le cadre de la formation OpenClassrooms AI Engineer.

## Support

Pour toute question ou problème :
1. Consulter la documentation dans `/docs`
2. Vérifier les issues existantes
3. Créer une nouvelle issue avec le template approprié

---

**Statut du projet** : En développement actif
**Dernière mise à jour** : Voir les commits récents
