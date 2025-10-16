# Pipeline CI/CD avec GitHub Actions

Ce document décrit le fonctionnement des pipelines d'intégration et de déploiement continus.

## Vue d'ensemble

```
Code Push → CI (Tests) → CD (Déploiement AWS) → Monitoring CloudWatch
```

## Workflows GitHub Actions

### 1. CI - Tests et Qualité (`.github/workflows/ci.yml`)

**Déclenché sur** :
- Push vers `main` ou `develop`
- Pull Requests vers `main`

**Étapes** :
1. ✅ **Tests unitaires** - Validation du code avec pytest
2. ✅ **Formatage** - Vérification avec Black
3. ✅ **Linting** - Analyse statique avec Flake8
4. ✅ **Build Docker** - Validation que l'image se construit
5. ✅ **Test des dépendances** - Vérification TensorFlow, MLflow, FastAPI
6. ✅ **Health check** - Test du endpoint /health

**Durée estimée** : 3-5 minutes

**Résultat** : Badge de statut dans GitHub

### 2. CD - Déploiement AWS (`.github/workflows/deploy.yml`)

**Déclenché sur** :
- Push vers `main` (après succès du CI)
- Déclenchement manuel via GitHub Actions UI

**Étapes** :
1. 📦 **Package** - Création du ZIP de déploiement
2. ☁️ **Upload S3** - Stockage du package
3. 🚀 **Déploiement EB** - Mise à jour de l'environnement AWS
4. ⏳ **Attente** - Vérification que le déploiement réussit
5. 🏥 **Health Check** - Test de l'API déployée

**Durée estimée** : 8-12 minutes

**Résultat** : URL de l'API en production

---

## Configuration requise

### Secrets GitHub

Configurer dans **Settings → Secrets and variables → Actions** :

| Secret | Description | Exemple |
|--------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | Access key IAM user | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | Secret key IAM user | `wJalrXUtnFEMI/K7MDENG/...` |

### Variables d'environnement (optionnel)

Modifier dans `.github/workflows/deploy.yml` si nécessaire :

```yaml
env:
  AWS_REGION: eu-west-1  # Votre région AWS
  EB_APPLICATION_NAME: sentiment-analysis-api
  EB_ENVIRONMENT_NAME: sentiment-analysis-api-prod
```

---

## Utilisation

### Déploiement automatique

```bash
# 1. Faire vos modifications
git add .
git commit -m "Amélioration du modèle"

# 2. Pousser sur main (déclenche CI puis CD)
git push origin main

# 3. Suivre la progression
# GitHub → Actions → Voir les workflows en cours
```

### Déploiement manuel

```bash
# Via GitHub UI
# Actions → CD - Déploiement AWS → Run workflow → Run workflow
```

### Rollback en cas de problème

```bash
# Option 1 : Via Git
git revert HEAD
git push origin main  # Redéploie la version précédente

# Option 2 : Via AWS EB
aws elasticbeanstalk update-environment \
  --application-name sentiment-analysis-api \
  --environment-name sentiment-analysis-api-prod \
  --version-label <version-précédente> \
  --region eu-west-1
```

---

## Surveillance du déploiement

### Logs GitHub Actions

1. **Accéder à GitHub** → Repository → Actions
2. **Cliquer** sur le workflow en cours
3. **Consulter** les logs en temps réel

### Logs AWS CloudWatch

```bash
# Via CLI
aws logs tail /aws/elasticbeanstalk/sentiment-api/application --follow

# Via Console
# AWS → CloudWatch → Log groups → /aws/elasticbeanstalk/sentiment-api
```

### Status de l'environnement

```bash
# Vérifier le statut
aws elasticbeanstalk describe-environments \
  --application-name sentiment-analysis-api \
  --environment-names sentiment-analysis-api-prod \
  --region eu-west-1

# Consulter les événements récents
aws elasticbeanstalk describe-events \
  --application-name sentiment-analysis-api \
  --environment-name sentiment-analysis-api-prod \
  --max-items 10 \
  --region eu-west-1
```

---

## Stratégie de déploiement

### Environnements

**Production** :
- Branche : `main`
- Environnement EB : `sentiment-analysis-api-prod`
- Déploiement : Automatique sur push

**Développement** (optionnel) :
- Branche : `develop`
- Environnement EB : `sentiment-analysis-api-dev`
- Déploiement : Manuel

### Workflow recommandé

```bash
# 1. Développement sur feature branch
git checkout -b feature/amelioration-modele
# ... modifications ...
git commit -m "Amélioration du modèle"

# 2. Créer Pull Request vers develop
git push origin feature/amelioration-modele
# GitHub → Create Pull Request → develop

# 3. CI s'exécute automatiquement
# Si tests passent → Merge vers develop

# 4. Tester sur environnement de développement
git checkout develop
git push origin develop
# Déploiement manuel sur sentiment-analysis-api-dev

# 5. Si OK → Merge vers main
# GitHub → Create Pull Request → main
# CI + CD automatiques vers production
```

---

## Optimisations

### Cache des dépendances Python

Le workflow CI utilise le cache pip :

```yaml
- uses: actions/setup-python@v5
  with:
    cache: 'pip'  # Cache automatique de requirements.txt
```

### Build Docker incrémental

Le workflow utilise BuildKit pour cache :

```yaml
- uses: docker/setup-buildx-action@v3
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### Déploiement conditionnel

Pour ne déployer que si les tests passent :

```yaml
# Dans deploy.yml
jobs:
  deploy:
    needs: [ci]  # Attend le workflow CI
```

---

## Troubleshooting

### CI échoue : "Tests failed"

**Cause** : Tests unitaires ne passent pas

**Solution** :
```bash
# Lancer les tests localement
pytest tests/ -v

# Corriger les tests
# Re-pousser
git push
```

### CD échoue : "AWS credentials invalid"

**Cause** : Secrets GitHub mal configurés

**Solution** :
1. Vérifier les secrets dans GitHub Settings
2. Re-générer les credentials IAM si nécessaire
3. Mettre à jour les secrets

### Déploiement réussi mais API non accessible

**Cause** : Problème de health check ou modèle non chargé

**Solution** :
```bash
# Consulter les logs
aws logs tail /aws/elasticbeanstalk/sentiment-api/application --follow

# Vérifier les variables d'environnement
aws elasticbeanstalk describe-configuration-settings \
  --application-name sentiment-analysis-api \
  --environment-name sentiment-analysis-api-prod
```

### Out of memory sur t2.micro

**Cause** : Modèle trop volumineux

**Solutions** :
1. Optimiser le modèle (réduire la taille)
2. Utiliser t2.small (hors free-tier)
3. Activer le swap sur l'instance

---

## Métriques de performance

### Temps de déploiement typiques

- **CI (Tests)** : 3-5 minutes
- **CD (Déploiement)** : 8-12 minutes
- **Total** : 11-17 minutes

### Optimisations possibles

1. **Tests parallèles** : Réduire à 2-3 min
2. **Cache Docker** : Réduire à 5-7 min
3. **Déploiement blue/green** : Zéro downtime

---

## Ressources

- [Documentation GitHub Actions](https://docs.github.com/en/actions)
- [AWS Elastic Beanstalk CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3.html)
- [Docker Build Best Practices](https://docs.docker.com/develop/dev-best-practices/)