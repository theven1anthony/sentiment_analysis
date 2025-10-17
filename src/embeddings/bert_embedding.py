"""
Module d'embedding BERT pour l'analyse de sentiment.
Utilise Transformers de Hugging Face pour le fine-tuning de BERT.
"""

import os

# Désactiver parallélisme tokenizer Rust (AVANT import transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """
    Dataset PyTorch pour les données de sentiment avec BERT.
    Utilise des encodages pré-calculés (évite segfault fork avec tokenizer Rust).
    """

    def __init__(self, input_ids, attention_mask, labels):
        """
        Args:
            input_ids: Array numpy des input_ids pré-tokenisés
            attention_mask: Array numpy des attention_masks
            labels: Array numpy des labels
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def __len__(self):
        return len(self.labels)


class BERTEmbedding:
    """Fine-tuning de BERT pour la classification de sentiment."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        batch_size: int = 16,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        output_dir: str = "./models/bert_tmp",
    ):
        """
        Initialise le modèle BERT pour classification.

        Args:
            model_name: Nom du modèle pré-entraîné BERT
            max_length: Longueur maximale des séquences
            batch_size: Taille des batchs pour l'entraînement
            epochs: Nombre d'époques d'entraînement
            learning_rate: Taux d'apprentissage
            output_dir: Répertoire pour sauvegarder les checkpoints
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir

        # Initialiser le tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model: Optional[BertForSequenceClassification] = None
        self.trainer: Optional[Trainer] = None

        # Détecter le device disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device utilisé: {self.device}")

    def prepare_data(self, texts: List[str], labels: List[int]) -> SentimentDataset:
        """
        Prépare les données pour BERT (pré-tokenisation par batches).

        Args:
            texts: Liste de textes
            labels: Liste de labels (0 ou 1)

        Returns:
            Dataset PyTorch avec encodages pré-calculés
        """
        # Tokeniser par petits batches pour éviter crash mémoire
        batch_size = 5000
        all_input_ids = []
        all_attention_mask = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            batch_encodings = self.tokenizer(
                batch_texts, truncation=True, padding="max_length", max_length=self.max_length, return_tensors=None
            )

            all_input_ids.extend(batch_encodings["input_ids"])
            all_attention_mask.extend(batch_encodings["attention_mask"])

        input_ids = np.array(all_input_ids, dtype=np.int64)
        attention_mask = np.array(all_attention_mask, dtype=np.int64)
        labels_array = np.array(labels, dtype=np.int64)

        return SentimentDataset(input_ids, attention_mask, labels_array)

    def fit(
        self,
        X_train: List[str],
        y_train: List[int],
        X_val: Optional[List[str]] = None,
        y_val: Optional[List[int]] = None,
    ) -> "BERTEmbedding":
        """
        Fine-tune le modèle BERT sur les données d'entraînement.

        Args:
            X_train: Textes d'entraînement
            y_train: Labels d'entraînement
            X_val: Textes de validation (optionnel)
            y_val: Labels de validation (optionnel)

        Returns:
            self pour chaînage
        """
        logger.info(f"Fine-tuning BERT sur {len(X_train)} échantillons...")

        # Charger le modèle pré-entraîné
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(self.device)

        # Préparer les datasets
        train_dataset = self.prepare_data(X_train, y_train)
        eval_dataset = None
        if X_val is not None and y_val is not None:
            eval_dataset = self.prepare_data(X_val, y_val)

        # Configuration de l'entraînement
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            report_to="none",
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
        )

        # Créer le Trainer
        from transformers.trainer_utils import set_seed

        set_seed(training_args.seed)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        self.trainer.train()
        logger.info("Fine-tuning terminé")

        return self

    def predict(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Prédit les classes pour les textes donnés (par batches pour éviter OOM).

        Args:
            texts: Liste de textes
            batch_size: Taille des batches pour prédiction (défaut: self.batch_size)

        Returns:
            Array numpy des prédictions (0 ou 1)
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant prédiction")

        if batch_size is None:
            batch_size = self.batch_size

        self.model.eval()
        all_predictions = []

        num_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"Prédiction sur {len(texts)} échantillons en {num_batches} batches de taille {batch_size}...")

        # Traiter par batches pour éviter OOM
        for batch_idx, i in enumerate(range(0, len(texts), batch_size), 1):
            batch_texts = texts[i : i + batch_size]

            # Tokenization du batch
            encodings = self.tokenizer(
                batch_texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)

            # Prédiction
            with torch.no_grad():
                outputs = self.model(**encodings)
                predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())

            if batch_idx % 10 == 0 or batch_idx == num_batches:
                logger.info(f"  Batch {batch_idx}/{num_batches} traité")

        return np.array(all_predictions)

    def predict_proba(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Prédit les probabilités pour les textes donnés (par batches pour éviter OOM).

        Args:
            texts: Liste de textes
            batch_size: Taille des batches pour prédiction (défaut: self.batch_size)

        Returns:
            Array numpy des probabilités (n_samples, 2)
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant prédiction")

        if batch_size is None:
            batch_size = self.batch_size

        self.model.eval()
        all_probas = []

        # Traiter par batches pour éviter OOM
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenization du batch
            encodings = self.tokenizer(
                batch_texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)

            # Prédiction
            with torch.no_grad():
                outputs = self.model(**encodings)
                probas = torch.softmax(outputs.logits, dim=-1)

            all_probas.extend(probas.cpu().numpy())

        return np.array(all_probas)

    def get_params(self) -> dict:
        """Retourne les paramètres du modèle pour logging MLflow."""
        return {
            "embedding_type": "bert",
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }
