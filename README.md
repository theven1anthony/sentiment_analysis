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

### 📊 Modèles FastText - Benchmark sur 50 000 Tweets

**Expérimentation** : `fasttext_models_50000_v1` - FastText + Réseaux de neurones
**Dataset** : 49 827 tweets Sentiment140 (après nettoyage)
**Algorithme** : FastText (Skip-gram, 100 dim, n-grammes 3-6) + Dense/LSTM
**Date** : Octobre 2025

#### 🏆 Meilleur Modèle

**Configuration gagnante** : FastText + Stemming + LSTM

| Métrique | Valeur |
|----------|--------|
| **F1-Score** | **0.7628** |
| **Accuracy** | **0.7631** |
| **AUC-ROC** | **0.8454** |
| **Précision** | 0.7641 |
| **Rappel** | 0.7631 |
| **Epochs entraînés** | 12/30 (early stopping) |
| **Vocabulaire** | 9 970 mots |
| **Temps d'entraînement** | 659.0s (~11 min) |

#### 📈 Comparaison FastText vs Word2Vec

**Architecture Dense (Embeddings moyennés) :**

| Embedding | F1-Score | AUC-ROC | Temps | Epochs | Δ (FT - W2V) |
|-----------|----------|---------|-------|--------|--------------|
| **FastText** | 0.7551 | 0.8337 | 29.1s | 28 | **+0.10%** |
| Word2Vec | 0.7541 | 0.8340 | 18.7s | 19 | - |

**Architecture LSTM (Séquences de vecteurs) :**

| Embedding | F1-Score | AUC-ROC | Temps | Epochs | Δ (FT - W2V) |
|-----------|----------|---------|-------|--------|--------------|
| **Word2Vec** | **0.7657** | **0.8463** | 659.8s | 12 | - |
| FastText | 0.7628 | 0.8454 | 659.0s | 12 | **-0.29%** |

#### 💡 Observations Clés

1. **Word2Vec légèrement meilleur sur LSTM** : +0.29% F1 (meilleur modèle global)
2. **FastText légèrement meilleur sur Dense** : +0.10% F1, mais 55% plus lent (calcul n-grammes)
3. **LSTM > Dense** : +0.8% F1 pour FastText (vs +0.9% pour Word2Vec)
4. **Avantage théorique FastText non confirmé** : Gestion OOV via n-grammes n'améliore pas les performances
5. **Hypothèse** : Dataset Sentiment140 bien formé, peu de typos ou mots hors vocabulaire
6. **Early stopping efficace** : Arrêt à 12 epochs au lieu de 30 pour LSTM

#### 📊 Comparaison avec Modèle Simple

| Modèle | F1-Score | AUC-ROC | Temps | Ratio Perf/Temps |
|--------|----------|---------|-------|------------------|
| **Simple (Logistic + TF-IDF)** | **0.7754** | **0.8569** | **0.49s** | **1.58 F1/s** |
| Word2Vec + LSTM | 0.7657 | 0.8463 | 659.8s | 0.001 F1/s |
| **FastText + LSTM** | 0.7628 | 0.8454 | 659.0s | 0.001 F1/s |
| FastText + Dense | 0.7551 | 0.8337 | 29.1s | 0.026 F1/s |

**Écart de performance** :
- Simple vs FastText+LSTM : **+1.3% F1, +1.2% AUC** (1345x plus rapide)
- Simple vs FastText+Dense : **+2.0% F1, +2.3% AUC** (59x plus rapide)

#### 🎯 Analyse et Recommandations

**Pourquoi TF-IDF surpasse Word2Vec/FastText ?**
1. **Corpus trop petit** : 50k tweets insuffisants pour entraîner des embeddings de qualité (besoin de millions)
2. **Tweets = textes courts** : Sentiment porté par mots-clés forts → TF-IDF capture parfaitement
3. **Word2Vec/FastText from scratch** : Embeddings sous-optimaux sans transfer learning
4. **Ratio paramètres/données** : Modèles neuronaux (50-500k paramètres) sur 50k samples → risque overfitting

**Pour ce projet** :
- ❌ **Ne pas utiliser FastText seul** : Pas d'amélioration vs Word2Vec, sous-performe le baseline
- ✅ **Tester embeddings pré-entraînés** : USE ou BERT pour transfer learning
- 📚 **Résultat cohérent avec la littérature** : TF-IDF bat embeddings from scratch sur petits corpus

**Prochaines étapes** :
1. **Universal Sentence Encoder (USE)** : Embeddings pré-entraînés sentence-level (attendu : ~78-80% F1)
2. **BERT fine-tuné** : Transfer learning sur transformer (meilleure performance attendue)

#### 📁 Rapports Complets

- Rapport détaillé : `reports/mlflow_report_fasttext_models_50000_v1_*.txt`
- Données brutes : `reports/mlflow_data_fasttext_models_50000_v1_*.csv`
- Courbes d'entraînement : Disponibles dans MLflow artifacts (training_curves/)
- MLflow UI : http://localhost:5001 (expérience: `fasttext_models_50000_v1`)

---

### 📊 Modèles USE - Benchmark sur 50 000 Tweets

**Expérimentation** : `use_models_50000_v1` - Universal Sentence Encoder + Dense
**Dataset** : 49 827 tweets Sentiment140 (après nettoyage)
**Algorithme** : USE pré-entraîné (512 dim, sentence-level) + Dense
**Date** : Octobre 2025

#### 🏆 Résultat (Stemming uniquement)

**Configuration testée** : USE + Stemming + Dense

| Métrique | Valeur |
|----------|--------|
| **F1-Score** | **0.7421** |
| **Accuracy** | **0.7423** |
| **AUC-ROC** | **0.8218** |
| **Précision** | 0.7432 |
| **Rappel** | 0.7423 |
| **Epochs entraînés** | 8/30 (early stopping) |
| **Temps d'entraînement** | 77.4s |

⚠️ **Note** : Expérience incomplète, seule la technique stemming a été testée (lemmatization manquante).

#### 📈 Comparaison avec tous les modèles

**Classement général (F1-Score) :**

| Rang | Modèle | F1-Score | AUC-ROC | Temps | Δ vs Baseline |
|------|--------|----------|---------|-------|---------------|
| 1️⃣ | **Simple (Logistic + TF-IDF)** | **0.7754** | **0.8569** | **0.49s** | - |
| 2️⃣ | Word2Vec + LSTM | 0.7657 | 0.8463 | 659.8s | -1.0% |
| 3️⃣ | FastText + LSTM | 0.7628 | 0.8454 | 659.0s | -1.3% |
| 4️⃣ | Word2Vec + Dense | 0.7571 | 0.8364 | 19.1s | -1.8% |
| 5️⃣ | FastText + Dense | 0.7551 | 0.8337 | 29.1s | -2.0% |
| 6️⃣ | **USE + Dense** | **0.7421** | **0.8218** | **77.4s** | **-3.3%** |

#### 💡 Observations Clés

1. **USE sous-performe TOUS les autres modèles** : F1=0.7421 (pire résultat du benchmark)
2. **-3.3% en dessous du baseline simple** : 33 points de moins que TF-IDF
3. **Early stopping très précoce** : Arrêt à 8 epochs (vs 12-19 pour Word2Vec/FastText)
4. **Temps d'entraînement élevé** : 77s pour chargement USE + entraînement (158x plus lent que baseline)
5. **AUC-ROC la plus faible** : 0.8218 (vs 0.8569 pour baseline, -35 points)

#### 🔍 Analyse : Pourquoi USE sous-performe ?

**Hypothèses expliquant les mauvaises performances :**

1. **USE optimisé pour similarité sémantique** :
   - Conçu pour mesurer la similarité entre phrases, pas pour classification de sentiment
   - Perd les mots-clés discriminants forts ("love", "hate") dans l'encodage global

2. **Tweets trop courts pour USE** :
   - USE excelle sur phrases longues avec contexte riche (20-30 mots)
   - Tweets : 10-15 mots en moyenne → contexte insuffisant
   - TF-IDF capture mieux les mots-clés dans textes courts

3. **Architecture trop simple** :
   - Une seule couche Dense au-dessus de USE (512 → 1)
   - Pas assez de capacité pour adapter les embeddings à la tâche

4. **Early stopping trop précoce** :
   - Arrêt à 8 epochs (sous-entraînement possible)
   - Modèle n'a pas eu le temps de converger correctement

5. **Embeddings figés** :
   - USE pré-entraîné non fine-tuné sur sentiment
   - Encodage générique pas adapté à la tâche spécifique

#### 🎯 Enseignements et Recommandations

**Ce que ce benchmark démontre :**
- ✅ **TF-IDF reste champion** : Simplicité et efficacité battent la complexité
- ✅ **Transfer learning ≠ garantie de succès** : Embeddings pré-entraînés pas toujours meilleurs
- ✅ **Textes courts = mots-clés > contexte** : USE perd face à approches lexicales
- ❌ **USE inadapté pour tweets** : Conçu pour phrases longues et riches en contexte

**Recommandations :**
- ❌ **Ne pas utiliser USE pour sentiment Twitter** : Sous-performe même les embeddings from scratch
- ✅ **Conserver TF-IDF comme baseline production** : Meilleur rapport performance/complexité
- 🔬 **Tester BERT fine-tuné** : Dernière chance pour les embeddings pré-entraînés
  - BERT peut être fine-tuné (contrairement à USE figé)
  - BERT-base conçu pour classification (USE pour similarité)

**Prochaine étape** :
- **BERT fine-tuning** : Entraîner les dernières couches sur sentiment Twitter
- Si BERT < TF-IDF → **Utiliser TF-IDF en production** (plus simple, plus rapide, meilleur)

#### 📁 Rapports Complets

- Rapport détaillé : `reports/mlflow_report_use_models_50000_v1_*.txt`
- Données brutes : `reports/mlflow_data_use_models_50000_v1_*.csv`
- Courbes d'entraînement : Disponibles dans MLflow artifacts (training_curves/)
- MLflow UI : http://localhost:5001 (expérience: `use_models_50000_v1`)

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
