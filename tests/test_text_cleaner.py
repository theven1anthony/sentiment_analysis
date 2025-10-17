#!/usr/bin/env python3
"""
Tests unitaires pour src/preprocessing/text_cleaner.py
Démontre la testabilité du code refactoré avec injection de dépendances.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from typing import List

from src.preprocessing.text_cleaner import (
    TextCleaner,
    load_sentiment140_data,
    preprocess_dataset,
    OutputLogger,
    StemmerProtocol,
    LemmatizerProtocol,
    TokenizerProtocol
)


class TestTextCleanerBasicCleaning:
    """Tests pour les méthodes de nettoyage de base (fonctions pures)."""

    def test_clean_tweet_removes_urls(self):
        cleaner = TextCleaner(remove_urls=True)
        text = "Check this out http://example.com great site!"
        result = cleaner.clean_tweet(text)
        assert "http" not in result
        assert "example.com" not in result

    def test_clean_tweet_removes_mentions(self):
        cleaner = TextCleaner(remove_mentions=True)
        text = "@user1 hello @user2 how are you?"
        result = cleaner.clean_tweet(text)
        assert "@user1" not in result
        assert "@user2" not in result

    def test_clean_tweet_keeps_mention_sign_when_configured(self):
        cleaner = TextCleaner(remove_mentions=False)
        text = "@user hello"
        result = cleaner.clean_tweet(text)
        # Le signe @ devrait être remplacé
        assert "@" not in result

    def test_clean_tweet_removes_hashtags(self):
        cleaner = TextCleaner(remove_hashtags=True)
        text = "I love #python and #coding"
        result = cleaner.clean_tweet(text)
        assert "#python" not in result
        assert "#coding" not in result

    def test_clean_tweet_keeps_hashtag_text_when_not_removing(self):
        cleaner = TextCleaner(remove_hashtags=False)
        text = "I love #python"
        result = cleaner.clean_tweet(text)
        assert "#" not in result
        assert "python" in result

    def test_clean_tweet_removes_punctuation(self):
        cleaner = TextCleaner()
        text = "Hello, world!!! How are you???"
        result = cleaner.clean_tweet(text)
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_clean_tweet_removes_numbers(self):
        cleaner = TextCleaner()
        text = "I have 123 apples and 456 oranges"
        result = cleaner.clean_tweet(text)
        assert "123" not in result
        assert "456" not in result

    def test_clean_tweet_lowercase(self):
        cleaner = TextCleaner(lowercase=True)
        text = "HELLO WORLD"
        result = cleaner.clean_tweet(text)
        assert result == "hello world"

    def test_clean_tweet_no_lowercase(self):
        cleaner = TextCleaner(lowercase=False)
        text = "HELLO World"
        result = cleaner.clean_tweet(text)
        assert "HELLO" in result or "hello" not in result.lower() != result

    def test_clean_tweet_handles_empty_string(self):
        cleaner = TextCleaner()
        result = cleaner.clean_tweet("")
        assert result == ""

    def test_clean_tweet_handles_none(self):
        cleaner = TextCleaner()
        result = cleaner.clean_tweet(None)
        assert result == ""

    def test_clean_batch_processes_multiple_tweets(self):
        cleaner = TextCleaner()
        texts = ["Hello @user!", "Check http://example.com", "Great #day"]
        results = cleaner.clean_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)


class TestTextCleanerExpandContractions:
    """Tests pour l'expansion des contractions (fonction pure)."""

    def test_expand_contractions_wont(self):
        cleaner = TextCleaner()
        text = "I won't go"
        result = cleaner.expand_contractions(text)
        assert "will not" in result

    def test_expand_contractions_cant(self):
        cleaner = TextCleaner()
        text = "I can't do it"
        result = cleaner.expand_contractions(text)
        assert "cannot" in result

    def test_expand_contractions_multiple(self):
        cleaner = TextCleaner()
        text = "I can't and won't do it"
        result = cleaner.expand_contractions(text)
        assert "cannot" in result
        assert "will not" in result

    def test_expand_contractions_case_insensitive(self):
        cleaner = TextCleaner()
        text = "I Can't and WON'T"
        result = cleaner.expand_contractions(text)
        assert "cannot" in result.lower()
        assert "will not" in result.lower()


class TestTextCleanerHandleNegations:
    """Tests pour la gestion des négations (fonction pure)."""

    def test_handle_negations_simple(self):
        cleaner = TextCleaner()
        text = "not good"
        result = cleaner.handle_negations(text)
        assert "NOT_good" in result

    def test_handle_negations_scope_limited(self):
        cleaner = TextCleaner()
        text = "not very good at all"
        result = cleaner.handle_negations(text)
        # Les mots devraient être préfixés
        assert "NOT_" in result

    def test_handle_negations_stops_at_punctuation(self):
        cleaner = TextCleaner()
        text = "not good. happy"
        result = cleaner.handle_negations(text)
        assert "NOT_good" in result
        assert "NOT_happy" not in result

    def test_handle_negations_stops_at_conjunction(self):
        cleaner = TextCleaner()
        text = "not good but happy"
        result = cleaner.handle_negations(text)
        assert "NOT_good" in result
        assert "NOT_happy" not in result


class TestTextCleanerHandleEmotions:
    """Tests pour la gestion des émotions (fonction pure)."""

    def test_handle_emotions_repeated_letters(self):
        cleaner = TextCleaner()
        text = "sooooo good"
        result = cleaner.handle_emotions(text)
        assert "sooooo" not in result
        assert "soo" in result

    def test_handle_emotions_caps(self):
        cleaner = TextCleaner()
        text = "AMAZING"
        result = cleaner.handle_emotions(text)
        assert "amazing" in result
        assert "CAPS" in result

    def test_handle_emotions_multiple_exclamation(self):
        cleaner = TextCleaner()
        text = "Great!!!"
        result = cleaner.handle_emotions(text)
        assert "EXCITED" in result

    def test_handle_emotions_multiple_question_marks(self):
        cleaner = TextCleaner()
        text = "Really???"
        result = cleaner.handle_emotions(text)
        assert "CONFUSED" in result


class TestTextCleanerWithMockedDependencies:
    """Tests avec dépendances mockées pour l'injection."""

    def test_textcleaner_with_mocked_stemmer(self):
        mock_stemmer = Mock(spec=StemmerProtocol)
        mock_stemmer.stem.return_value = "test"

        cleaner = TextCleaner(stemmer=mock_stemmer)

        # Le stemmer mocké devrait être utilisé
        assert cleaner.stemmer == mock_stemmer

    def test_textcleaner_with_mocked_lemmatizer(self):
        mock_lemmatizer = Mock(spec=LemmatizerProtocol)
        mock_lemmatizer.lemmatize.return_value = "test"

        cleaner = TextCleaner(lemmatizer=mock_lemmatizer)

        # Le lemmatizer mocké devrait être utilisé
        assert cleaner.lemmatizer == mock_lemmatizer

    def test_textcleaner_with_mocked_tokenizer(self):
        mock_tokenizer = Mock(spec=TokenizerProtocol)
        mock_tokenizer.return_value = ["hello", "world"]

        cleaner = TextCleaner(tokenizer=mock_tokenizer)

        # Le tokenizer mocké devrait être utilisé
        assert cleaner.tokenizer == mock_tokenizer

    def test_textcleaner_with_custom_stopwords(self):
        custom_stopwords = {"custom", "stop"}

        cleaner = TextCleaner(stop_words=custom_stopwords)

        assert cleaner.stop_words == custom_stopwords

    def test_advanced_preprocess_with_mocked_stemmer(self):
        mock_stemmer = Mock(spec=StemmerProtocol)
        mock_stemmer.stem.side_effect = lambda word: word + "_stemmed"

        mock_tokenizer = Mock(spec=TokenizerProtocol)
        mock_tokenizer.return_value = ["hello", "world"]

        cleaner = TextCleaner(
            stemmer=mock_stemmer,
            tokenizer=mock_tokenizer,
            stop_words=set()  # Pas de stopwords pour simplifier
        )

        result = cleaner.advanced_preprocess(
            "Hello world",
            use_stemming=True,
            use_lemmatization=False,
            remove_stopwords=False
        )

        # Vérifier que le stemmer a été appelé
        assert mock_stemmer.stem.called
        assert "stemmed" in result

    def test_advanced_preprocess_with_mocked_lemmatizer(self):
        mock_lemmatizer = Mock(spec=LemmatizerProtocol)
        mock_lemmatizer.lemmatize.side_effect = lambda word: word + "_lemmatized"

        mock_tokenizer = Mock(spec=TokenizerProtocol)
        mock_tokenizer.return_value = ["hello", "world"]

        cleaner = TextCleaner(
            lemmatizer=mock_lemmatizer,
            tokenizer=mock_tokenizer,
            stop_words=set()
        )

        result = cleaner.advanced_preprocess(
            "Hello world",
            use_stemming=False,
            use_lemmatization=True,
            remove_stopwords=False
        )

        # Vérifier que le lemmatizer a été appelé
        assert mock_lemmatizer.lemmatize.called
        assert "lemmatized" in result


class TestLoadSentiment140DataWithMockedLogger:
    """Tests pour load_sentiment140_data avec logger mocké."""

    def test_load_with_mocked_logger(self, tmp_path):
        # Créer un fichier CSV temporaire
        csv_file = tmp_path / "test.csv"
        csv_content = "0,1,date,query,user,negative tweet\n4,2,date,query,user,positive tweet\n"
        csv_file.write_text(csv_content)

        # Mock du logger
        mock_logger = Mock(spec=OutputLogger)

        # Charger les données avec le logger mocké
        df = load_sentiment140_data(str(csv_file), logger=mock_logger)

        # Vérifier que le logger a été appelé
        assert mock_logger.info.call_count > 0

        # Vérifier que les données sont correctes
        assert len(df) == 2
        assert 'text' in df.columns
        assert 'sentiment' in df.columns

    def test_load_with_sampling_mocked_logger(self, tmp_path):
        # Créer un fichier CSV avec plus de données
        csv_file = tmp_path / "test.csv"
        lines = []
        for i in range(100):
            sentiment = 0 if i % 2 == 0 else 4
            lines.append(f"{sentiment},{i},date,query,user,tweet {i}\n")
        csv_file.write_text("".join(lines))

        # Mock du logger
        mock_logger = Mock(spec=OutputLogger)

        # Charger avec échantillonnage
        df = load_sentiment140_data(str(csv_file), sample_size=50, logger=mock_logger)

        # Vérifier les appels au logger
        assert mock_logger.info.call_count >= 5  # Au moins 5 messages

        # Vérifier l'échantillonnage
        assert len(df) == 50


class TestPreprocessDataset:
    """Tests pour preprocess_dataset."""

    def test_preprocess_dataset_with_default_cleaner(self):
        df = pd.DataFrame({
            'text': ['Hello @user!', 'Check http://example.com', 'Great #day'],
            'sentiment': [0, 1, 0]
        })

        result = preprocess_dataset(df)

        assert 'cleaned_text' in result.columns
        assert len(result) == 3

    def test_preprocess_dataset_with_custom_cleaner(self):
        cleaner = TextCleaner(remove_urls=False, remove_mentions=False)

        df = pd.DataFrame({
            'text': ['Hello @user!', 'Check http://example.com'],
            'sentiment': [0, 1]
        })

        result = preprocess_dataset(df, cleaner=cleaner)

        assert 'cleaned_text' in result.columns

    def test_preprocess_dataset_removes_empty_texts(self):
        df = pd.DataFrame({
            'text': ['Hello', '###', 'World'],
            'sentiment': [0, 1, 0]
        })

        result = preprocess_dataset(df)

        # Le tweet avec seulement des hashtags devrait être vide après nettoyage
        assert len(result) < len(df)


class TestPreprocessWithTechniques:
    """Tests pour preprocess_with_techniques."""

    def test_preprocess_stemming(self):
        cleaner = TextCleaner()
        texts = ["running", "jumps"]

        # Stemming devrait réduire les mots
        result = cleaner.preprocess_with_techniques(texts, technique='stemming')

        assert len(result) == 2
        assert all(isinstance(text, str) for text in result)

    def test_preprocess_lemmatization(self):
        cleaner = TextCleaner()
        texts = ["running", "jumps"]

        result = cleaner.preprocess_with_techniques(texts, technique='lemmatization')

        assert len(result) == 2
        assert all(isinstance(text, str) for text in result)

    def test_preprocess_basic(self):
        cleaner = TextCleaner()
        texts = ["Hello @user!", "Test #hashtag"]

        result = cleaner.preprocess_with_techniques(texts, technique='basic')

        assert len(result) == 2
        # Basic devrait juste appeler clean_tweet
        assert "@" not in result[0]
        assert "#" not in result[1]
