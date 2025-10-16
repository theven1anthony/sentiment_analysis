# Configuration des Secrets GitHub pour Azure

Ce document liste tous les secrets GitHub nécessaires pour le pipeline CI/CD Azure.

## Secrets Requis

### 1. AZURE_CREDENTIALS
**Type** : Secret
**Description** : Credentials du Service Principal pour authentification Azure
**Format** : JSON

**Obtenir la valeur** :
```bash
SUBSCRIPTION_ID=$(az account show --query id --output tsv)

az ad sp create-for-rbac \
  --name "github-actions-sentiment-api" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/sentiment-analysis-rg \
  --sdk-auth
```

**Exemple de valeur** :
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

---

### 2. AZURE_RESOURCE_GROUP
**Type** : Secret
**Description** : Nom du Resource Group Azure
**Valeur** : `sentiment-analysis-rg`

**Obtenir la valeur** :
```bash
echo "sentiment-analysis-rg"
```

---

### 3. AZURE_SUBSCRIPTION_ID
**Type** : Secret (optionnel, utilisé dans le résumé)
**Description** : ID de la subscription Azure

**Obtenir la valeur** :
```bash
az account show --query id --output tsv
```

**Exemple** : `68be5678-c930-4e65-a41b-809e56fe13f9`

---

### 4. APPLICATIONINSIGHTS_CONNECTION_STRING
**Type** : Secret
**Description** : Connection String d'Application Insights pour le monitoring

**Obtenir la valeur** :
```bash
az monitor app-insights component show \
  --app sentiment-api-insights \
  --resource-group sentiment-analysis-rg \
  --query connectionString \
  --output tsv
```

**Format** : `InstrumentationKey=xxx;IngestionEndpoint=https://...;LiveEndpoint=https://...`

---

## Configuration dans GitHub

### Via l'interface GitHub

1. Aller sur votre repository : `https://github.com/<username>/sentiment_analysis`
2. Cliquer sur **Settings**
3. Dans le menu de gauche : **Secrets and variables** → **Actions**
4. Cliquer sur **New repository secret**
5. Créer chaque secret avec son nom et sa valeur

### Via GitHub CLI (optionnel)

```bash
# Installer GitHub CLI si nécessaire
# https://cli.github.com/

# Se connecter
gh auth login

# Ajouter les secrets
gh secret set AZURE_CREDENTIALS < .azure-sp-credentials.json
gh secret set AZURE_RESOURCE_GROUP -b "sentiment-analysis-rg"
gh secret set AZURE_SUBSCRIPTION_ID -b "$(az account show --query id -o tsv)"
gh secret set APPLICATIONINSIGHTS_CONNECTION_STRING -b "$(az monitor app-insights component show --app sentiment-api-insights --resource-group sentiment-analysis-rg --query connectionString -o tsv)"
```

---

## Vérification des Secrets

### Lister les secrets configurés

Via GitHub web :
- Aller dans **Settings** → **Secrets and variables** → **Actions**
- Vous devriez voir 4 secrets listés

Via GitHub CLI :
```bash
gh secret list
```

**Output attendu** :
```
APPLICATIONINSIGHTS_CONNECTION_STRING  Updated 2025-10-16
AZURE_CREDENTIALS                      Updated 2025-10-16
AZURE_RESOURCE_GROUP                   Updated 2025-10-16
AZURE_SUBSCRIPTION_ID                  Updated 2025-10-16
```

---

## Secrets Utilisés dans le Pipeline

Le fichier `.github/workflows/deploy.yml` utilise ces secrets ainsi :

```yaml
# Authentication Azure
- name: Azure Login
  uses: azure/login@v2
  with:
    creds: ${{ secrets.AZURE_CREDENTIALS }}

# Configuration des variables d'environnement
- name: Configuration des variables d'environnement Azure
  run: |
    az webapp config appsettings set \
      --name sentiment-api-at2025 \
      --resource-group ${{ secrets.AZURE_RESOURCE_GROUP }} \
      --settings \
        APPLICATIONINSIGHTS_CONNECTION_STRING="${{ secrets.APPLICATIONINSIGHTS_CONNECTION_STRING }}"
```

---

## Sécurité

### Bonnes pratiques

✅ **À faire** :
- Ne jamais committer les secrets dans Git
- Ajouter les fichiers contenant des secrets au `.gitignore`
- Utiliser des secrets GitHub au lieu de variables d'environnement pour les données sensibles
- Limiter les permissions du Service Principal au minimum nécessaire (Contributor sur le Resource Group uniquement)

❌ **À ne pas faire** :
- Ne pas logger les secrets dans les workflows
- Ne pas utiliser `echo` pour afficher les secrets
- Ne pas partager les secrets via des canaux non sécurisés

### Rotation des secrets

Si le Service Principal est compromis :

```bash
# Supprimer l'ancien
az ad sp delete --id $(az ad sp list --display-name "github-actions-sentiment-api" --query "[0].appId" -o tsv)

# Recréer un nouveau
SUBSCRIPTION_ID=$(az account show --query id --output tsv)
az ad sp create-for-rbac \
  --name "github-actions-sentiment-api" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/sentiment-analysis-rg \
  --sdk-auth

# Mettre à jour le secret GitHub
gh secret set AZURE_CREDENTIALS < nouveau-credentials.json
```

---

## Dépannage

### Erreur : "Azure login failed"

**Cause** : Secret `AZURE_CREDENTIALS` invalide ou expiré

**Solution** :
1. Vérifier que le JSON est complet et valide
2. Régénérer le Service Principal
3. Mettre à jour le secret GitHub

### Erreur : "Resource group not found"

**Cause** : Secret `AZURE_RESOURCE_GROUP` incorrect

**Solution** :
1. Vérifier le nom exact du Resource Group : `az group list --output table`
2. Mettre à jour le secret avec le bon nom

### Erreur : "Application Insights connection failed"

**Cause** : Secret `APPLICATIONINSIGHTS_CONNECTION_STRING` invalide

**Solution** :
1. Récupérer la bonne Connection String :
   ```bash
   az monitor app-insights component show \
     --app sentiment-api-insights \
     --resource-group sentiment-analysis-rg \
     --query connectionString -o tsv
   ```
2. Mettre à jour le secret GitHub

---

**Dernière mise à jour** : Octobre 2025
**Version** : 1.0.0