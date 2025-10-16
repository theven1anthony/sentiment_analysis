# Pipeline CI/CD avec GitHub Actions

Ce document décrit le fonctionnement des pipelines d'intégration et de déploiement continus.

## Vue d'ensemble

```
Code Push → CI (Tests) → CD (Déploiement Azure) → Monitoring Application Insights
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

### 2. CD - Déploiement Azure (`.github/workflows/deploy.yml`)

**Déclenché sur** :
- Push vers `main` (après succès du CI)
- Déclenchement manuel via GitHub Actions UI (`workflow_dispatch`)

**Étapes** :
1. 📦 **Package** - Préparation du package de déploiement
2. 🔐 **Azure Login** - Authentification avec Service Principal
3. 🚀 **Déploiement Azure** - Mise à jour de l'App Service
4. ⚙️ **Configuration** - Variables d'environnement (Application Insights)
5. ⏳ **Attente** - Vérification que l'application démarre
6. 🏥 **Health Check** - Test de l'API déployée

**Durée estimée** : 5-8 minutes

**Résultat** : URL de l'API en production - `https://sentiment-api-at2025.azurewebsites.net`

---

## Configuration requise

### Secrets GitHub

Configurer dans **Settings → Secrets and variables → Actions** :

| Secret | Description | Obtention |
|--------|-------------|-----------|
| `AZURE_CREDENTIALS` | Service Principal JSON | `az ad sp create-for-rbac --sdk-auth` |
| `AZURE_RESOURCE_GROUP` | Nom du Resource Group | `sentiment-analysis-rg` |
| `AZURE_SUBSCRIPTION_ID` | ID de la subscription Azure | `az account show --query id -o tsv` |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Connection String Application Insights | Azure Portal → Application Insights |

**Documentation détaillée** : `docs/github_secrets_azure.md`

### Variables d'environnement

Configurées dans `.github/workflows/deploy.yml` :

```yaml
env:
  AZURE_WEBAPP_NAME: sentiment-api-at2025
  AZURE_WEBAPP_PACKAGE_PATH: '.'
  PYTHON_VERSION: '3.12'
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
# Actions → CD - Déploiement Azure → Run workflow → Run workflow
```

### Rollback en cas de problème

```bash
# Option 1 : Via Git
git revert HEAD
git push origin main  # Redéploie la version précédente

# Option 2 : Via Azure CLI
az webapp deployment list-publishing-profiles \
  --name sentiment-api-at2025 \
  --resource-group sentiment-analysis-rg
```

---

## Surveillance du déploiement

### Logs GitHub Actions

1. **Accéder à GitHub** → Repository → Actions
2. **Cliquer** sur le workflow en cours
3. **Consulter** les logs en temps réel

### Logs Azure Application Insights

```bash
# Via Azure Portal
# Application Insights → Logs → Requêtes KQL

# Via CLI - Logs de l'App Service
az webapp log tail \
  --name sentiment-api-at2025 \
  --resource-group sentiment-analysis-rg
```

### Status de l'App Service

```bash
# Vérifier le statut
az webapp show \
  --name sentiment-api-at2025 \
  --resource-group sentiment-analysis-rg \
  --query "state" \
  --output tsv

# Consulter les déploiements récents
az webapp deployment list \
  --name sentiment-api-at2025 \
  --resource-group sentiment-analysis-rg
```

---

## Stratégie de déploiement

### Environnements

**Production** :
- Branche : `main`
- App Service : `sentiment-api-at2025`
- Déploiement : Automatique sur push
- URL : https://sentiment-api-at2025.azurewebsites.net

**Développement** (optionnel) :
- Branche : `develop`
- App Service : `sentiment-api-dev`
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
# Déploiement manuel sur sentiment-api-dev

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
    needs: []  # Ajouter needs: [ci] pour attendre CI
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

### CD échoue : "Azure login failed"

**Cause** : Secrets GitHub mal configurés

**Solution** :
1. Vérifier les secrets dans GitHub Settings
2. Re-générer le Service Principal si nécessaire :
```bash
az ad sp create-for-rbac \
  --name "github-actions-sentiment-api" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/sentiment-analysis-rg \
  --sdk-auth
```
3. Mettre à jour le secret `AZURE_CREDENTIALS`

### Déploiement réussi mais API non accessible

**Cause** : Problème de health check ou modèle non chargé

**Solution** :
```bash
# Consulter les logs
az webapp log tail \
  --name sentiment-api-at2025 \
  --resource-group sentiment-analysis-rg

# Vérifier les variables d'environnement
az webapp config appsettings list \
  --name sentiment-api-at2025 \
  --resource-group sentiment-analysis-rg
```

### Out of memory sur F1 (Free tier)

**Cause** : Modèle trop volumineux pour le plan gratuit

**Solutions** :
1. Optimiser le modèle (réduire la taille des embeddings)
2. Utiliser un plan payant (B1 Basic: ~13€/mois)
3. Activer le swap (non disponible sur F1)

---

## Métriques de performance

### Temps de déploiement typiques

- **CI (Tests)** : 3-5 minutes
- **CD (Déploiement Azure)** : 5-8 minutes
- **Total** : 8-13 minutes

### Optimisations possibles

1. **Tests parallèles** : Réduire à 2-3 min
2. **Cache Docker** : Réduire à 3-5 min
3. **Déploiement slots** : Zéro downtime (plans payants)

---

## Monitoring en production

### Application Insights

**Accès** :
- Azure Portal → Application Insights → `sentiment-api-insights`

**Métriques disponibles** :
- Temps de réponse
- Taux d'erreur
- Nombre de requêtes
- Custom events (misclassifications)

### Alertes configurées

| Alerte | Condition | Action |
|--------|-----------|--------|
| `high-misclassification-rate` | 3 erreurs en 5 min | Email via Action Group |

**Documentation** : `docs/azure_configuration.md` (section 6.2)

---

## Ressources

- [Documentation GitHub Actions](https://docs.github.com/en/actions)
- [Azure App Service Deployment](https://docs.microsoft.com/azure/app-service/deploy-github-actions)
- [Azure CLI Reference](https://docs.microsoft.com/cli/azure/)
- [Application Insights Documentation](https://docs.microsoft.com/azure/azure-monitor/app/app-insights-overview)