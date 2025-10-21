# Configuration Infrastructure Azure

Ce guide décrit la procédure complète pour configurer l'infrastructure Azure pour le déploiement de l'API de sentiment analysis.

## Prérequis

- ✅ Compte Azure actif (free-tier)
- ✅ Azure CLI installé localement
- ✅ Repository GitHub avec le code
- ✅ Accès administrateur à la subscription Azure

---

## Étape 1 : Connexion à Azure CLI

```bash
# Se connecter à Azure
az login

# Vérifier la subscription active
az account show

# Si vous avez plusieurs subscriptions, définir celle à utiliser
az account list --output table
az account set --subscription "<Subscription ID>"
```

**Vérification** : La commande `az account show` doit afficher votre subscription active.

---

## Étape 2 : Création du Resource Group

```bash
# Créer le groupe de ressources (région Europe de l'Ouest)
az group create \
  --name sentiment-analysis-rg \
  --location westeurope

# Vérifier la création
az group show --name sentiment-analysis-rg
```

**Output attendu** :
```json
{
  "id": "/subscriptions/.../resourceGroups/sentiment-analysis-rg",
  "location": "westeurope",
  "name": "sentiment-analysis-rg",
  "properties": {
    "provisioningState": "Succeeded"
  }
}
```

---

## Étape 3 : Création de l'App Service Plan (Free Tier)

```bash
# Créer le plan App Service (Free tier F1 - équivalent t2.micro AWS)
az appservice plan create \
  --name sentiment-api-plan \
  --resource-group sentiment-analysis-rg \
  --location westeurope \
  --sku F1 \
  --is-linux

# Vérifier la création
az appservice plan show \
  --name sentiment-api-plan \
  --resource-group sentiment-analysis-rg
```

**Caractéristiques du tier F1** :
- 60 minutes de CPU par jour
- 1 GB RAM
- 1 GB stockage
- Parfait pour développement/test

**Note** : Le tier F1 peut avoir des limitations de performance. Pour production, envisager B1 (Basic, ~13€/mois).

---

## Étape 4 : Création de la Web App avec Python

```bash
# Créer la Web App avec runtime Python 3.12
az webapp create \
  --name sentiment-analysis-api \
  --resource-group sentiment-analysis-rg \
  --plan sentiment-api-plan \
  --runtime "PYTHON:3.12"

# Configurer pour utiliser le port 8000 (FastAPI)
az webapp config appsettings set \
  --name sentiment-analysis-api \
  --resource-group sentiment-analysis-rg \
  --settings WEBSITES_PORT=8000

# Activer les logs
az webapp log config \
  --name sentiment-analysis-api \
  --resource-group sentiment-analysis-rg \
  --application-logging filesystem \
  --detailed-error-messages true \
  --failed-request-tracing true \
  --web-server-logging filesystem

# Obtenir l'URL de l'application
az webapp show \
  --name sentiment-analysis-api \
  --resource-group sentiment-analysis-rg \
  --query defaultHostName \
  --output tsv
```

**URL attendue** : `sentiment-analysis-api.azurewebsites.net`

**Test initial** :
```bash
curl https://sentiment-analysis-api.azurewebsites.net
```

---

## Étape 5 : Création d'Application Insights

```bash
# Créer l'Application Insights workspace
az monitor app-insights component create \
  --app sentiment-api-insights \
  --location westeurope \
  --resource-group sentiment-analysis-rg \
  --application-type web

# Récupérer l'Instrumentation Key
az monitor app-insights component show \
  --app sentiment-api-insights \
  --resource-group sentiment-analysis-rg \
  --query instrumentationKey \
  --output tsv

# Récupérer la Connection String (recommandé)
az monitor app-insights component show \
  --app sentiment-api-insights \
  --resource-group sentiment-analysis-rg \
  --query connectionString \
  --output tsv

# Lier Application Insights à la Web App
APPINSIGHTS_KEY=$(az monitor app-insights component show \
  --app sentiment-api-insights \
  --resource-group sentiment-analysis-rg \
  --query instrumentationKey \
  --output tsv)

az webapp config appsettings set \
  --name sentiment-analysis-api \
  --resource-group sentiment-analysis-rg \
  --settings APPINSIGHTS_INSTRUMENTATIONKEY=$APPINSIGHTS_KEY
```

**Important** : Sauvegarder la Connection String dans un fichier sécurisé :

```bash
# Sauvegarder dans un fichier local (à ajouter au .gitignore)
az monitor app-insights component show \
  --app sentiment-api-insights \
  --resource-group sentiment-analysis-rg \
  --query connectionString \
  --output tsv > .azure-appinsights-connection.txt
```

Cette Connection String sera nécessaire pour :
- Configuration de l'API Python (`api/main.py`)
- Variables d'environnement locales (`.env`)

---

## Étape 6 : Configuration de l'Action Group (Alertes)

### 6.1 Créer l'Action Group

```bash
# Créer un Action Group pour les alertes email
az monitor action-group create \
  --name sentiment-alerts \
  --resource-group sentiment-analysis-rg \
  --short-name sent-alert \
  --action email admin-email votre.email@example.com

# Vérifier la création
az monitor action-group show \
  --name sentiment-alerts \
  --resource-group sentiment-analysis-rg
```

**Remplacer** : `votre.email@example.com` par votre adresse email réelle.

**Note** : Vous recevrez un email de confirmation Azure. Cliquez sur le lien pour activer les alertes.

### 6.2 Créer les règles d'alerte

**Alerte principale : Misclassifications (3 tweets mal classés en 5 minutes)**

Cette alerte surveille les logs applicatifs (traces) dans Application Insights pour détecter quand l'API FastAPI log un message d'alerte concernant les misclassifications.

```bash
# Créer l'alerte scheduled query pour les misclassifications
az monitor scheduled-query create \
  --name high-misclassification-rate \
  --resource-group sentiment-analysis-rg \
  --scopes $(az monitor app-insights component show --app sentiment-api-insights --resource-group sentiment-analysis-rg --query id -o tsv) \
  --condition "count 'AlertEvents' > 0" \
  --condition-query AlertEvents="traces | where message contains 'Alerte déclenchée' and message contains 'high_misclassification_rate'" \
  --window-size 5m \
  --evaluation-frequency 5m \
  --action-groups $(az monitor action-group show --name sentiment-alerts --resource-group sentiment-analysis-rg --query id -o tsv) \
  --description "Alerte si 3+ tweets mal classés en 5 minutes (logs traces)" \
  --severity 2
```

**Fonctionnement** :
1. L'endpoint `/feedback` de l'API détecte quand 3 tweets sont mal classés en 5 minutes
2. L'API log un message d'alerte via le logger Python (niveau ERROR)
3. Azure Monitor OpenTelemetry envoie les logs dans la table `traces` d'Application Insights
4. Azure Monitor exécute cette requête KQL toutes les 5 minutes
5. Si le message d'alerte est détecté, l'Action Group est déclenché (email envoyé)

**Note importante** : La requête utilise la table `traces` car OpenTelemetry avec `azure-monitor-opentelemetry` envoie les logs Python standard dans cette table, et non dans `customEvents`.

**Vérifier l'alerte** :

```bash
# Lister les alertes scheduled query
az monitor scheduled-query list \
  --resource-group sentiment-analysis-rg \
  --query "[].{Name:name, Enabled:enabled, Description:description}" \
  --output table

# Afficher les détails
az monitor scheduled-query show \
  --name high-misclassification-rate \
  --resource-group sentiment-analysis-rg
```

**Alertes optionnelles (monitoring technique)** :

Pour monitorer la santé technique de l'application, vous pouvez ajouter :

```bash
# Alerte temps de réponse élevé
az monitor metrics alert create \
  --name high-response-time \
  --resource-group sentiment-analysis-rg \
  --scopes $(az webapp show --name sentiment-api-at2025 --resource-group sentiment-analysis-rg --query id --output tsv) \
  --condition "avg AverageResponseTime > 500" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action sentiment-alerts \
  --description "Alerte si temps de réponse moyen > 500ms pendant 5 min"
```

**Note** : L'alerte de misclassification est l'exigence principale du projet (critère CE2 - Suivi de performance en production).

### 6.3 Ajouter des alertes SMS (optionnel)

```bash
# Ajouter une notification SMS au même Action Group
az monitor action-group update \
  --name sentiment-alerts \
  --resource-group sentiment-analysis-rg \
  --add-action sms admin-sms +33612345678
```

**Remplacer** : `+33612345678` par votre numéro de téléphone au format international.

---

## Étape 7 : Création du Service Principal pour GitHub Actions

### 7.1 Créer le Service Principal

```bash
# Récupérer l'ID de la subscription
SUBSCRIPTION_ID=$(az account show --query id --output tsv)

# Créer le Service Principal avec rôle Contributor
az ad sp create-for-rbac \
  --name "github-actions-sentiment-api" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/sentiment-analysis-rg \
  --sdk-auth
```

**Output attendu** (format JSON) :
```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

**IMPORTANT** : Sauvegarder ce JSON complet dans un fichier sécurisé. Vous ne pourrez plus le récupérer.

```bash
# Sauvegarder dans un fichier local (à ajouter au .gitignore)
az ad sp create-for-rbac \
  --name "github-actions-sentiment-api" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/sentiment-analysis-rg \
  --sdk-auth > .azure-sp-credentials.json
```

### 7.2 Vérifier les permissions du Service Principal

```bash
# Lister les rôles assignés
az role assignment list \
  --assignee $(az ad sp list --display-name "github-actions-sentiment-api" --query "[0].appId" -o tsv) \
  --output table
```

**Output attendu** :
```
Principal                             Role         Scope
------------------------------------  -----------  ----------------------------------------
github-actions-sentiment-api          Contributor  /subscriptions/.../resourceGroups/sentiment-analysis-rg
```

---

## Étape 8 : Configuration des Secrets GitHub

### 8.1 Accéder aux GitHub Secrets

1. Aller sur votre repository GitHub
2. Cliquer sur **Settings** → **Secrets and variables** → **Actions**
3. Cliquer sur **New repository secret**

### 8.2 Créer les secrets

| Nom du Secret | Valeur | Description |
|---------------|--------|-------------|
| `AZURE_CREDENTIALS` | JSON complet du Service Principal | Authentication pour GitHub Actions |
| `AZURE_WEBAPP_NAME` | `sentiment-analysis-api` | Nom de la Web App |
| `AZURE_RESOURCE_GROUP` | `sentiment-analysis-rg` | Nom du Resource Group |

**Pour `AZURE_CREDENTIALS`** : Copier le JSON complet retourné à l'étape 7.1 (ou contenu du fichier `.azure-sp-credentials.json`)

### 8.3 Variables d'environnement (optionnel)

Si vous préférez utiliser des variables au lieu de secrets (pour les valeurs non sensibles) :

- Aller dans **Settings** → **Secrets and variables** → **Actions** → **Variables**
- Créer les variables :
  - `AZURE_REGION` = `westeurope`
  - `AZURE_SUBSCRIPTION_ID` = Votre Subscription ID

---

## Étape 9 : Vérification Complète de l'Infrastructure

### 9.1 Lister toutes les ressources

```bash
# Lister toutes les ressources créées
az resource list \
  --resource-group sentiment-analysis-rg \
  --output table
```

**Output attendu** :
```
Name                      ResourceGroup          Location    Type
------------------------  ---------------------  ----------  ------------------------------
sentiment-api-plan        sentiment-analysis-rg  westeurope  Microsoft.Web/serverfarms
sentiment-analysis-api    sentiment-analysis-rg  westeurope  Microsoft.Web/sites
sentiment-api-insights    sentiment-analysis-rg  westeurope  Microsoft.Insights/components
sentiment-alerts          sentiment-analysis-rg  westeurope  microsoft.insights/actiongroups
```

### 9.2 Tester l'URL de la Web App

```bash
# Obtenir l'URL
WEBAPP_URL=$(az webapp show \
  --name sentiment-analysis-api \
  --resource-group sentiment-analysis-rg \
  --query defaultHostName \
  --output tsv)

echo "URL de l'API : https://$WEBAPP_URL"

# Tester (doit retourner 404 ou page par défaut, car pas encore de code déployé)
curl https://$WEBAPP_URL
```

### 9.3 Vérifier Application Insights

```bash
# Vérifier l'état d'Application Insights
az monitor app-insights component show \
  --app sentiment-api-insights \
  --resource-group sentiment-analysis-rg \
  --query "provisioningState" \
  --output tsv
```

**Output attendu** : `Succeeded`

### 9.4 Accéder aux dashboards Azure

**Azure Portal** :
1. Aller sur [portal.azure.com](https://portal.azure.com)
2. Rechercher `sentiment-analysis-rg`
3. Explorer les ressources créées

**Application Insights Dashboard** :
1. Aller sur Application Insights → `sentiment-api-insights`
2. Vérifier que les métriques sont actives (peut prendre quelques minutes)

---

## Résumé des Ressources Créées

| Ressource | Nom | Type | SKU/Tier | Coût |
|-----------|-----|------|----------|------|
| **Resource Group** | `sentiment-analysis-rg` | Container | - | Gratuit |
| **App Service Plan** | `sentiment-api-plan` | Compute | F1 (Free) | Gratuit 12 mois |
| **Web App** | `sentiment-analysis-api` | Application | Python 3.12 | Inclus dans plan |
| **Application Insights** | `sentiment-api-insights` | Monitoring | Standard | 5GB/mois gratuit |
| **Action Group** | `sentiment-alerts` | Alerting | Email/SMS | Email gratuit, SMS payant |
| **Service Principal** | `github-actions-sentiment-api` | Identity | - | Gratuit |

**Coût total estimé (free-tier actif)** : 0€/mois
**Coût après 12 mois** : ~13-15€/mois (si passage en B1)

---

## Informations à Sauvegarder

Créer un fichier `.azure-config.env` (à ajouter au `.gitignore`) :

```bash
# Informations Azure Infrastructure
AZURE_SUBSCRIPTION_ID=<votre-subscription-id>
AZURE_TENANT_ID=<votre-tenant-id>
AZURE_RESOURCE_GROUP=sentiment-analysis-rg
AZURE_WEBAPP_NAME=sentiment-analysis-api
AZURE_LOCATION=westeurope

# Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING=<connection-string-from-step-5>

# Action Group
AZURE_ACTION_GROUP_ID=/subscriptions/<sub-id>/resourceGroups/sentiment-analysis-rg/providers/microsoft.insights/actionGroups/sentiment-alerts

# Web App URL
AZURE_WEBAPP_URL=https://sentiment-analysis-api.azurewebsites.net
```

**Remplir les valeurs** :
- `AZURE_SUBSCRIPTION_ID` : `az account show --query id -o tsv`
- `AZURE_TENANT_ID` : `az account show --query tenantId -o tsv`
- `APPLICATIONINSIGHTS_CONNECTION_STRING` : Résultat de l'étape 5

---

## Prochaines Étapes

Une fois l'infrastructure créée et vérifiée :

1. ✅ **Phase 2 terminée** : Infrastructure Azure opérationnelle
2. ⏭️ **Phase 3** : Modification du code Python
   - Adapter `requirements-api.txt`
   - Créer `src/monitoring/azure_monitor_integration.py`
   - Modifier `api/main.py`
3. ⏭️ **Phase 4** : Mise à jour du pipeline CI/CD
   - Modifier `.github/workflows/deploy.yml`
4. ⏭️ **Phase 5** : Premier déploiement et tests

---

## Troubleshooting

### Erreur : Subscription non trouvée

```bash
# Lister toutes les subscriptions disponibles
az account list --output table

# Activer la subscription correcte
az account set --subscription "<Subscription Name or ID>"
```

### Erreur : Nom de Web App déjà pris

Le nom `sentiment-analysis-api` doit être unique globalement. Si pris, utiliser :

```bash
# Ajouter un suffixe unique (ex: vos initiales + nombre aléatoire)
az webapp create \
  --name sentiment-analysis-api-at2025 \
  --resource-group sentiment-analysis-rg \
  --plan sentiment-api-plan \
  --runtime "PYTHON:3.12"
```

### Erreur : Quota Free Tier dépassé

Vous ne pouvez avoir qu'un seul App Service Plan F1 par subscription. Si erreur :

```bash
# Vérifier les plans existants
az appservice plan list --output table

# Supprimer un ancien plan si nécessaire
az appservice plan delete --name <old-plan-name> --resource-group <old-rg>
```

### Erreur : Service Principal déjà existe

```bash
# Lister les Service Principals existants
az ad sp list --display-name "github-actions-sentiment-api" --output table

# Supprimer l'ancien
az ad sp delete --id <appId>

# Recréer
az ad sp create-for-rbac --name "github-actions-sentiment-api" ...
```

---

## Nettoyage (après tests)

Pour supprimer toute l'infrastructure :

```bash
# Supprimer le Resource Group (supprime toutes les ressources associées)
az group delete \
  --name sentiment-analysis-rg \
  --yes \
  --no-wait

# Supprimer le Service Principal
az ad sp delete --id $(az ad sp list --display-name "github-actions-sentiment-api" --query "[0].appId" -o tsv)
```

**Attention** : Cette opération est irréversible. Toutes les données seront perdues.

---

## Ressources et Documentation

### Documentation Azure
- [Azure CLI Reference](https://docs.microsoft.com/en-us/cli/azure/)
- [Azure App Service Documentation](https://docs.microsoft.com/en-us/azure/app-service/)
- [Application Insights Documentation](https://docs.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview)
- [Azure Monitor Alerts](https://docs.microsoft.com/en-us/azure/azure-monitor/alerts/alerts-overview)

### Tutoriels
- [Deploy Python to Azure App Service](https://docs.microsoft.com/en-us/azure/app-service/quickstart-python)
- [GitHub Actions for Azure](https://docs.microsoft.com/en-us/azure/app-service/deploy-github-actions)
- [Application Insights Python SDK](https://docs.microsoft.com/en-us/azure/azure-monitor/app/opencensus-python)

---

**Date de création** : Octobre 2025
**Dernière mise à jour** : Octobre 2025
**Version** : 1.0.0