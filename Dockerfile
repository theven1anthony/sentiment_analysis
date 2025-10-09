# Image de base Python 3.12
FROM python:3.12-slim

# Variables d'environnement pour éviter les problèmes de buffering
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copie du code source
COPY . .

# Création des répertoires nécessaires
RUN mkdir -p models mlruns data

# Ports pour les services
EXPOSE 5001 8000 8501

# Commande par défaut (peut être surchargée)
CMD ["python", "train_advanced_model.py"]