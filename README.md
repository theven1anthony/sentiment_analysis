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

#### Modèles Avancés (TensorFlow/Keras + Embeddings)

**Word2Vec avec réseaux de neurones :**
```bash
# Word2Vec avec architecture Dense (embeddings moyennés)
python train_word2vec_model.py --technique=stemming --sample-size=50000

# Word2Vec avec architecture LSTM (séquences)
python train_word2vec_model.py --technique=lemmatization --with-lstm --sample-size=50000

# Comparaison stemming vs lemmatization
python train_word2vec_model.py --technique=both --sample-size=50000 --description="Benchmark Word2Vec"

# Configuration avancée
python train_word2vec_model.py \
    --technique=stemming \
    --with-lstm \
    --vector-size=200 \
    --sample-size=100000 \
    --description="Production Word2Vec LSTM"
```

**Autres embeddings (à venir) :**
```bash
# FastText avec architecture LSTM
python train_fasttext_model.py --technique=lemmatization --with-lstm

# Universal Sentence Encoder (USE)
python train_use_model.py --technique=stemming

# BERT (fine-tuning)
python train_bert_model.py --technique=stemming --epochs=3
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

## Résultats des Expérimentations

### 📊 Modèles Simples - Benchmark sur 50 000 Tweets

**Expérimentation** : `simple_models_50000_v1` - Analyse comparative Stemming vs Lemmatization
**Dataset** : 50 000 tweets Sentiment140
**Algorithme** : Logistic Regression + TF-IDF
**Date** : Octobre 2025

#### 🏆 Meilleur Modèle

**Configuration gagnante** : Stemming + Négations=True + Émotions=True

| Métrique | Valeur |
|----------|--------|
| **F1-Score** | **0.7754** |
| **Accuracy** | **0.7754** |
| **AUC-ROC** | **0.8569** |
| **Précision** | 0.7754 |
| **Rappel** | 0.7754 |
| **Temps d'entraînement** | 0.49s |

#### 📈 Comparaison Stemming vs Lemmatization

| Configuration | Stemming | Lemmatization | Δ (Stemming - Lemma) |
|---------------|----------|---------------|----------------------|
| **Négations + Émotions** | **0.7754** (AUC: 0.8569) | 0.7722 (AUC: 0.8559) | **+0.0032** |
| **Négations seules** | **0.7743** (AUC: 0.8577) | 0.7715 (AUC: 0.8554) | **+0.0028** |
| **Émotions seules** | **0.7712** (AUC: 0.8534) | 0.7706 (AUC: 0.8528) | **+0.0006** |
| **Sans gestion** | **0.7739** (AUC: 0.8534) | 0.7710 (AUC: 0.8517) | **+0.0029** |

#### 📊 Statistiques Globales

| Métrique | Valeur |
|----------|--------|
| **Accuracy moyenne** | 0.7725 |
| **Écart-type (std)** | 0.0018 |
| **F1-Score moyen** | 0.7725 |
| **Écart-type (std)** | 0.0018 |

#### 💡 Observations Clés

1. **Stemming systématiquement meilleur** : Performance supérieure dans toutes les configurations (+0.28% à +0.32%)
2. **Négations + Émotions = optimal** : Meilleure configuration avec F1=0.7754 et AUC=0.8569
3. **Stabilité remarquable** : Écart-type très faible (0.0018) → résultats reproductibles
4. **Temps d'entraînement** : Stemming (0.43-0.49s) ~15% plus rapide que lemmatization (0.49-0.60s)
5. **AUC-ROC élevée** : 0.85+ sur tous les modèles → excellente capacité de discrimination

#### 🎯 Recommandations

- **Production** : Stemming + Négations=True + Émotions=True
- **Justification** : Meilleur compromis performance/rapidité
- **Gain vs baseline** : +0.32% vs lemmatization sur configuration équivalente
- **Robustesse** : Variance minimale entre runs (std=0.0018)

#### 📁 Rapports Complets

- Rapport détaillé : `reports/mlflow_report_simple_models_50000_v1_*.txt`
- Données brutes : `reports/mlflow_data_simple_models_50000_v1_*.csv`
- MLflow UI : http://localhost:5001 (expérience: `simple_models_50000_v1`)

---

### 📊 Modèles Word2Vec - Benchmark sur 50 000 Tweets

**Expérimentation** : `word2vec_models_50000_v1` - Word2Vec + Réseaux de neurones
**Dataset** : 49 827 tweets Sentiment140 (après nettoyage)
**Algorithme** : Word2Vec (Skip-gram, 100 dim) + Dense/LSTM
**Date** : Octobre 2025

#### 🏆 Meilleur Modèle

**Configuration gagnante** : Word2Vec + Stemming + LSTM

| Métrique | Valeur |
|----------|--------|
| **F1-Score** | **0.7653** |
| **Accuracy** | **0.7654** |
| **AUC-ROC** | **0.8472** |
| **Précision** | 0.7658 |
| **Rappel** | 0.7654 |
| **Epochs entraînés** | 13/30 (early stopping) |
| **Vocabulaire** | 9 970 mots |
| **Temps d'entraînement** | 701.8s (~11.7 min) |

#### 📈 Comparaison Architectures

**Dense (Embeddings moyennés) :**

| Technique | F1-Score | AUC-ROC | Temps moyen | Epochs moyen |
|-----------|----------|---------|-------------|--------------|
| **Stemming** | **0.7571** (±0.0004) | **0.8364** | 19.1s | 19.7 |
| **Lemmatization** | **0.7526** (±0.0009) | **0.8331** | 18.1s | 18.3 |
| **Δ (Stem - Lemma)** | **+0.0045** | **+0.0033** | +1.0s | +1.4 |

**LSTM (Séquences de vecteurs) :**

| Technique | F1-Score | AUC-ROC | Temps | Epochs |
|-----------|----------|---------|-------|--------|
| **Stemming** | **0.7653** | **0.8472** | 701.8s | 13 |
| **Lemmatization** | **0.7609** | **0.8435** | 643.7s | 12 |
| **Δ (Stem - Lemma)** | **+0.0044** | **+0.0037** | +58.1s | +1 |

#### 💡 Observations Clés

1. **LSTM surpasse Dense** : +0.9% F1-Score, +1.1% AUC-ROC
2. **Stemming systématiquement meilleur** : +0.45% (Dense) et +0.44% (LSTM) vs lemmatization
3. **Trade-off performance/temps** : LSTM 38x plus lent que Dense pour +0.9% de gain
4. **Vocabulaire plus compact** : Stemming (9 970 mots) vs Lemmatization (11 353 mots) = -12%
5. **Early stopping efficace** : Arrêt à 12-13 epochs au lieu de 30 (gain de temps ×2.3)
6. **Stabilité Dense remarquable** : Écart-type très faible (0.0004-0.0009) → résultats reproductibles

#### 📊 Comparaison avec Modèle Simple

| Modèle | F1-Score | AUC-ROC | Temps | Ratio Perf/Temps |
|--------|----------|---------|-------|------------------|
| **Simple (Logistic + TF-IDF)** | **0.7754** | **0.8569** | **0.49s** | **1.58 F1/s** |
| **Word2Vec + Dense** | 0.7571 | 0.8364 | 19.1s | 0.040 F1/s |
| **Word2Vec + LSTM** | 0.7653 | 0.8472 | 701.8s | 0.001 F1/s |

**Écart de performance** :
- Simple vs W2V+Dense : **+1.8% F1, +2.0% AUC** (39x plus rapide)
- Simple vs W2V+LSTM : **+1.0% F1, +1.0% AUC** (1432x plus rapide)

#### 🎯 Recommandations

**Pour ce projet** :
- ❌ **Ne pas utiliser Word2Vec seul** : Performance inférieure au modèle simple baseline
- ✅ **Tester d'autres embeddings** : FastText, USE, BERT pour surpasser le baseline
- ⚠️ **LSTM coût/bénéfice faible** : +0.9% pour 38x plus de temps vs Dense

**Prochaines étapes** :
1. **Universal Sentence Encoder (USE)** : Embeddings de documents pré-entraînés state-of-the-art
2. **BERT fine-tuné** : Modèle transformer pour NLP (meilleure performance attendue)
3. **FastText** : Gestion des mots hors vocabulaire et sous-mots

**Si Word2Vec nécessaire** :
- Configuration optimale : Stemming + LSTM (F1=0.7653)
- Alternative rapide : Stemming + Dense (F1=0.7571, 19s)

#### 📁 Rapports Complets

- Rapport détaillé : `reports/mlflow_report_word2vec_models_50000_v1_*.txt`
- Données brutes : `reports/mlflow_data_word2vec_models_50000_v1_*.csv`
- Courbes d'entraînement : Disponibles dans MLflow artifacts (training_curves/)
- MLflow UI : http://localhost:5001 (expérience: `word2vec_models_50000_v1`)

---

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
