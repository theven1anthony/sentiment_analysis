# Guide de Déploiement AWS Elastic Beanstalk

Ce guide décrit la procédure complète pour déployer l'API de sentiment analysis sur AWS Elastic Beanstalk avec GitHub Actions.

## Prérequis

- ✅ Compte AWS actif (free-tier éligible)
- ✅ AWS CLI installé localement
- ✅ Repository GitHub avec le code
- ✅ Modèle MLflow entraîné et prêt

---

## Étape 1 : Configuration AWS

### 1.1 Création d'un utilisateur IAM pour GitHub Actions

1. **Connexion à AWS Console** → IAM → Users → Create User
2. **Nom** : `github-actions-deploy`
3. **Permissions** : Attacher les policies suivantes :
   - `AWSElasticBeanstalkFullAccess`
   - `AmazonS3FullAccess`
   - `CloudWatchFullAccess`

4. **Créer les credentials** :
   - Onglet "Security credentials"
   - Clic sur "Create access key"
   - Type : "Application running outside AWS"
   - **Sauvegarder** : `Access Key ID` et `Secret Access Key` (vous ne les reverrez plus)

### 1.2 Création du bucket S3 pour les déploiements

```bash
# Remplacer REGION par votre région (ex: eu-west-1)
aws s3 mb s3://sentiment-analysis-api-deployments --region eu-west-1
```

### 1.3 Création de l'application Elastic Beanstalk

```bash
# Créer l'application
aws elasticbeanstalk create-application \
  --application-name sentiment-analysis-api \
  --description "API de prédiction de sentiment pour Air Paradis" \
  --region eu-west-1

# Créer l'environnement
aws elasticbeanstalk create-environment \
  --application-name sentiment-analysis-api \
  --environment-name sentiment-analysis-api-prod \
  --solution-stack-name "64bit Amazon Linux 2023 v4.3.0 running Docker" \
  --tier Name=WebServer,Type=Standard \
  --option-settings \
    Namespace=aws:autoscaling:launchconfiguration,OptionName=InstanceType,Value=t2.micro \
    Namespace=aws:elasticbeanstalk:environment,OptionName=EnvironmentType,Value=SingleInstance \
  --region eu-west-1
```

**Attendre 5-10 minutes** que l'environnement soit créé.

---

## Étape 2 : Configuration GitHub Secrets

1. **Accéder à votre repository GitHub** → Settings → Secrets and variables → Actions

2. **Créer les secrets suivants** :

| Nom du Secret | Valeur | Description |
|---------------|--------|-------------|
| `AWS_ACCESS_KEY_ID` | Votre Access Key ID | Depuis IAM User |
| `AWS_SECRET_ACCESS_KEY` | Votre Secret Access Key | Depuis IAM User |

3. **Variables d'environnement** (optionnel) :
   - `AWS_REGION` : `eu-west-1` (ou votre région)
   - `EB_APPLICATION_NAME` : `sentiment-analysis-api`
   - `EB_ENVIRONMENT_NAME` : `sentiment-analysis-api-prod`

---

## Étape 3 : Préparation du modèle de production

**Architecture de déploiement :**
- Le modèle MLflow pyfunc complet (20.5 MB) est téléchargé depuis Model Registry
- Tous les artefacts (modèle Keras, Word2Vec, préprocessing) sont sauvegardés dans Git
- L'API charge le modèle depuis les fichiers locaux (pas de connexion MLflow requise)

### 3.1 Télécharger le modèle depuis MLflow Model Registry

**Prérequis** : Le modèle doit être enregistré dans MLflow Model Registry (via MLflow UI)

```bash
# Lancer MLflow si nécessaire (local ou Docker)
docker-compose up mlflow

# Télécharger le modèle complet
python deploy_model.py --name w2v_200K_model --version 2
```

**Ce script effectue :**
1. Connexion au MLflow Model Registry
2. Téléchargement du modèle pyfunc complet avec tous ses artefacts
3. Sauvegarde dans `models/production/pyfunc_model/model/`
4. Création des métadonnées dans `models/production/metadata.pkl`

### 3.2 Committer le modèle dans Git

```bash
# Vérifier la taille du modèle (doit être ~20 MB)
du -sh models/production/

# Ajouter au repository
git add models/production/
git commit -m "Ajout modèle de production w2v_200K_model v2"
git push origin main
```

**Note** : Le modèle est volontairement commité dans Git (20.5 MB) car :
- Simplifie le déploiement (pas de dépendance externe)
- Taille acceptable pour Git (< 100 MB)
- Versioning automatique avec le code

---

## Étape 4 : Configuration CloudWatch et SNS (Alertes)

### 4.1 Créer un Topic SNS pour les alertes

```bash
# Créer le topic
aws sns create-topic \
  --name sentiment-api-alerts \
  --region eu-west-1

# Sauvegarder l'ARN retourné (ex: arn:aws:sns:eu-west-1:123456789:sentiment-api-alerts)
```

### 4.2 S'abonner aux alertes

```bash
# Remplacer par votre email
aws sns subscribe \
  --topic-arn arn:aws:sns:eu-west-1:123456789:sentiment-api-alerts \
  --protocol email \
  --notification-endpoint votre.email@example.com \
  --region eu-west-1

# Confirmer l'email reçu
```

### 4.3 Créer l'alarme CloudWatch

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name sentiment-api-error-rate \
  --alarm-description "Alerte si 3 erreurs en 5 minutes" \
  --metric-name ErrorCount \
  --namespace SentimentAnalysis/API \
  --statistic Sum \
  --period 300 \
  --threshold 3 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:sns:eu-west-1:123456789:sentiment-api-alerts \
  --region eu-west-1
```

---

## Étape 5 : Déploiement

### 5.1 Déploiement automatique via GitHub Actions

```bash
# Pousser sur main déclenche automatiquement le déploiement
git add .
git commit -m "Configuration déploiement AWS"
git push origin main
```

**Suivre la progression** :
- GitHub → Actions → Onglet "CD - Déploiement AWS"

### 5.2 Déploiement manuel

```bash
# Créer le package
zip -r deploy.zip . -x "*.git*" "venv/*" "mlruns/*" "data/*"

# Upload vers S3
aws s3 cp deploy.zip s3://sentiment-analysis-api-deployments/

# Créer la version
aws elasticbeanstalk create-application-version \
  --application-name sentiment-analysis-api \
  --version-label v1.0.0 \
  --source-bundle S3Bucket="sentiment-analysis-api-deployments",S3Key="deploy.zip" \
  --region eu-west-1

# Déployer
aws elasticbeanstalk update-environment \
  --application-name sentiment-analysis-api \
  --environment-name sentiment-analysis-api-prod \
  --version-label v1.0.0 \
  --region eu-west-1
```

---

## Étape 6 : Vérification

### 6.1 Obtenir l'URL de l'API

```bash
aws elasticbeanstalk describe-environments \
  --application-name sentiment-analysis-api \
  --environment-names sentiment-analysis-api-prod \
  --query 'Environments[0].CNAME' \
  --output text \
  --region eu-west-1
```

**Résultat** : `sentiment-analysis-api-prod.eu-west-1.elasticbeanstalk.com`

### 6.2 Tester l'API

```bash
# Health check
curl http://sentiment-analysis-api-prod.eu-west-1.elasticbeanstalk.com/health

# Documentation interactive
open http://sentiment-analysis-api-prod.eu-west-1.elasticbeanstalk.com/docs

# Test de prédiction
curl -X POST "http://sentiment-analysis-api-prod.eu-west-1.elasticbeanstalk.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

### 6.3 Vérifier les logs CloudWatch

```bash
# Console AWS → CloudWatch → Log groups → /aws/elasticbeanstalk/sentiment-api

# Ou via CLI
aws logs tail /aws/elasticbeanstalk/sentiment-api/application --follow
```

---

## Étape 7 : Monitoring et Maintenance

### 7.1 Consulter les métriques

**Console AWS** :
- Elastic Beanstalk → Environments → sentiment-analysis-api-prod → Monitoring

**Métriques disponibles** :
- CPU Utilization
- Network In/Out
- Request Count
- Latency
- HTTP 4xx/5xx errors

### 7.2 Mettre à jour l'application

```bash
# Simple push sur main déclenche le déploiement
git push origin main

# Ou rollback vers une version précédente
aws elasticbeanstalk update-environment \
  --application-name sentiment-analysis-api \
  --environment-name sentiment-analysis-api-prod \
  --version-label <previous-version> \
  --region eu-west-1
```

### 7.3 Scaling (après free-tier)

```bash
# Activer l'auto-scaling
aws elasticbeanstalk update-environment \
  --application-name sentiment-analysis-api \
  --environment-name sentiment-analysis-api-prod \
  --option-settings \
    Namespace=aws:autoscaling:asg,OptionName=MinSize,Value=1 \
    Namespace=aws:autoscaling:asg,OptionName=MaxSize,Value=3 \
  --region eu-west-1
```

---

## Étape 8 : Nettoyage (après projet)

### 8.1 Supprimer l'environnement

```bash
aws elasticbeanstalk terminate-environment \
  --environment-name sentiment-analysis-api-prod \
  --region eu-west-1
```

### 8.2 Supprimer l'application

```bash
aws elasticbeanstalk delete-application \
  --application-name sentiment-analysis-api \
  --region eu-west-1
```

### 8.3 Supprimer les ressources S3

```bash
aws s3 rb s3://sentiment-analysis-api-deployments --force
```

### 8.4 Supprimer le topic SNS

```bash
aws sns delete-topic \
  --topic-arn arn:aws:sns:eu-west-1:123456789:sentiment-api-alerts \
  --region eu-west-1
```

---

## Coûts AWS Free Tier

**Inclus gratuitement (12 mois)** :
- ✅ 750h/mois de t2.micro EC2
- ✅ 5 GB de stockage S3
- ✅ CloudWatch : 10 métriques custom, 1 million de requêtes API

**Après 12 mois** :
- EC2 t2.micro : ~$10-15/mois
- S3 : ~$0.50/mois
- **Total** : ~$15/mois

**Conseil** : Arrêter l'environnement quand inutilisé :
```bash
# Arrêter (gratuit en mode arrêté)
aws elasticbeanstalk update-environment \
  --environment-name sentiment-analysis-api-prod \
  --option-settings Namespace=aws:autoscaling:asg,OptionName=MinSize,Value=0 \
  --region eu-west-1
```

---

## Troubleshooting

### Problème : L'environnement ne démarre pas

**Solution** :
```bash
# Consulter les logs
aws elasticbeanstalk describe-events \
  --application-name sentiment-analysis-api \
  --environment-name sentiment-analysis-api-prod \
  --max-items 20 \
  --region eu-west-1
```

### Problème : Out of memory sur t2.micro

**Solution** : Optimiser le modèle ou utiliser t2.small (hors free-tier)

### Problème : GitHub Actions échoue

**Vérifier** :
- Les secrets AWS sont corrects
- L'utilisateur IAM a les bonnes permissions
- Le bucket S3 existe

---

## Ressources

- [Documentation AWS Elastic Beanstalk](https://docs.aws.amazon.com/elasticbeanstalk/)
- [GitHub Actions AWS](https://github.com/aws-actions)
- [AWS CLI Reference](https://docs.aws.amazon.com/cli/)
- [CloudWatch Alarms](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/AlarmThatSendsEmail.html)