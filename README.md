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
Démarrez le serveur de tracking MLflow :
```bash
mlflow ui --host 0.0.0.0 --port 5001
```
Interface accessible sur : http://localhost:5001

## Données

Le projet utilise le dataset **Sentiment140** contenant 1,6M de tweets étiquetés :
- **0** : Sentiment négatif
- **1** : Sentiment positif
- **Format** : CSV avec colonnes [sentiment, id, date, query, user, text]
- **Répartition** : 50% négatif, 50% positif (équilibré)

## Utilisation

### Entraînement des modèles

#### Modèle Simple (Baseline)

**Entraîner avec une technique spécifique :**
```bash
# Avec stemming
python train_simple_model.py --technique=stemming --description="Baseline stemming production"

# Avec lemmatization
python train_simple_model.py --technique=lemmatization --description="Baseline lemmatization"
```

**Comparer les techniques :**
```bash
# Comparaison complète stemming vs lemmatization
python train_simple_model.py --technique=both --description="Comparaison techniques preprocessing"
```

**Options avancées :**
```bash
# Avec échantillon réduit pour test rapide
python train_simple_model.py \
    --technique=lemmatization \
    --sample-size=50000 \
    --description="Test rapide lemmatization" \
    --experiment-name="tests_rapides"

# Dataset complet (par défaut)
python train_simple_model.py --technique=both --description="Entraînement production"
```

#### Modèles Avancés (TensorFlow/Keras)
```bash
# À implémenter - modèles avec embeddings Word2Vec/FastText
python train_advanced_model.py --embedding=word2vec
python train_advanced_model.py --embedding=fasttext --with-lstm
```

#### Modèle BERT
```bash
# À implémenter - modèle BERT
python train_bert_model.py --model=bert-base-uncased
```

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

### API de prédiction

#### Démarrer l'API locale
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Utiliser l'API
```bash
# Prédiction simple
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product!"}'

# Réponse
{
  "sentiment": 1,
  "confidence": 0.87,
  "text": "I love this product!"
}
```

#### Interface de test Streamlit
```bash
streamlit run interface/app.py
```

## Architecture

### Structure du projet
```
sentiment_analysis/
├── api/                    # API FastAPI
├── data/                   # Datasets
├── models/                 # Modèles entraînés
│   ├── staging/           # Modèles en test
│   └── production/        # Modèles déployés
├── notebooks/             # Exploration de données
├── reports/               # Rapports d'évaluation
├── src/                   # Code source
│   ├── embeddings/        # Implémentations d'embeddings
│   ├── evaluation/        # Métriques d'évaluation
│   ├── models/            # Définitions des modèles
│   ├── monitoring/        # Outils de surveillance
│   └── preprocessing/     # Prétraitement des données
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
# API en mode production
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Avec Docker (à configurer)
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

### Déploiement cloud (AWS)
```bash
# Configuration à venir
# - AWS EC2 ou Lambda
# - Pipeline CI/CD avec GitHub Actions
# - Monitoring CloudWatch
```

## Métriques et Monitoring

### Métriques d'évaluation
- **Accuracy** : Taux de bonnes classifications
- **F1-Score** : Mesure équilibrée (métrique principale)
- **Précision/Rappel** : Performance par classe
- **AUC-ROC** : Capacité de discrimination
- **Temps d'entraînement** : Performance opérationnelle

### Surveillance en production
- **Seuil d'alerte** : 3 prédictions incorrectes en 5 minutes
- **Monitoring** : AWS CloudWatch
- **Alertes** : Email/SMS automatiques
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
