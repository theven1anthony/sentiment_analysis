import re
import string
from typing import List, Protocol, Optional, Set, Callable
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


# Protocoles pour injection de dépendances
class StemmerProtocol(Protocol):
    """Protocole pour un stemmer."""

    def stem(self, word: str) -> str:
        """Applique le stemming à un mot."""
        ...


class LemmatizerProtocol(Protocol):
    """Protocole pour un lemmatizer."""

    def lemmatize(self, word: str) -> str:
        """Applique la lemmatization à un mot."""
        ...


class TokenizerProtocol(Protocol):
    """Protocole pour un tokenizer."""

    def __call__(self, text: str) -> List[str]:
        """Tokenise un texte."""
        ...


class OutputLogger(Protocol):
    """Protocole pour logger de sortie."""

    def info(self, message: str) -> None:
        """Log un message d'information."""
        ...


class PrintLogger:
    """Logger par défaut qui utilise print()."""

    def info(self, message: str) -> None:
        print(message)


class TextCleaner:
    """Nettoyage et preprocessing des tweets pour l'analyse de sentiment."""

    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        lowercase: bool = True,
        language: str = "english",
        stemmer: Optional[StemmerProtocol] = None,
        lemmatizer: Optional[LemmatizerProtocol] = None,
        stop_words: Optional[Set[str]] = None,
        tokenizer: Optional[TokenizerProtocol] = None,
    ):
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.lowercase = lowercase
        self.language = language

        # Initialisation des outils de prétraitement avancé avec injection
        self.stemmer = stemmer if stemmer is not None else SnowballStemmer(language)
        self.lemmatizer = lemmatizer if lemmatizer is not None else WordNetLemmatizer()
        self.stop_words = stop_words if stop_words is not None else set(stopwords.words(language))
        self.tokenizer = tokenizer if tokenizer is not None else word_tokenize

    def clean_tweet(self, text: str) -> str:
        """Nettoie un tweet individuel."""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Suppression des URLs
        if self.remove_urls:
            text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Suppression des mentions (@username)
        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)
        else:
            text = text.replace("@", "")

        # Suppression des hashtags (#tag)
        if self.remove_hashtags:
            text = re.sub(r"#\w+", "", text)
        else:
            text = text.replace("#", "")

        # Suppression des caractères non-ASCII
        text = re.sub(r"[^\x00-\x7F]+", "", text)

        # Suppression de la ponctuation excessive
        text = re.sub(r"[" + string.punctuation + "]+", " ", text)

        # Suppression de tous les chiffres
        text = re.sub(r"\d+", "", text)

        # Suppression des espaces multiples
        text = re.sub(r"\s+", " ", text)

        # Conversion en minuscules
        if self.lowercase:
            text = text.lower()

        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Nettoie une liste de tweets."""
        return [self.clean_tweet(text) for text in texts]

    def expand_contractions(self, text: str) -> str:
        """Expanse les contractions pour améliorer l'analyse sentiment."""
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "i'm": "i am",
            "you're": "you are",
            "it's": "it is",
            "that's": "that is",
            "what's": "what is",
            "here's": "here is",
            "how's": "how is",
            "where's": "where is",
            "there's": "there is",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "won't": "will not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
        }

        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)

        return text

    def handle_negations(self, text: str) -> str:
        """
        Gère les négations en ajoutant un préfixe pour préserver le sentiment.
        Amélioration: portée limitée et gestion des conjonctions.
        """
        negations = [
            "not",
            "no",
            "never",
            "none",
            "nobody",
            "nothing",
            "neither",
            "nowhere",
            "dont",
            "doesnt",
            "didnt",
            "wont",
            "wouldnt",
            "shouldnt",
            "couldnt",
            "cant",
            "cannot",
            "isnt",
            "arent",
            "wasnt",
            "werent",
            "havent",
            "hasnt",
            "hadnt",
        ]

        # Mots qui arrêtent la portée de la négation
        scope_breakers = [
            "but",
            "however",
            "although",
            "though",
            "yet",
            "still",
            "nevertheless",
            "except",
            "besides",
            "meanwhile",
            "while",
        ]

        # Mots qui prolongent naturellement la négation
        intensifiers = [
            "very",
            "really",
            "quite",
            "rather",
            "pretty",
            "so",
            "extremely",
            "incredibly",
            "absolutely",
            "totally",
        ]

        tokens = text.split()
        result = []
        negate = False
        words_negated = 0  # Compteur pour limiter la portée

        for token in tokens:
            token_lower = token.lower()

            # Vérifier si le token se termine par une ponctuation de fin de phrase
            ends_with_sentence_punct = len(token) > 0 and token[-1] in ".!?;,"

            if token_lower in negations:
                negate = True
                words_negated = 0
                result.append(token)
            elif token in [".", "!", "?", ",", ";"]:  # Ponctuation seule
                negate = False
                words_negated = 0
                result.append(token)
            elif token_lower in scope_breakers:  # Conjonctions qui cassent la négation
                negate = False
                words_negated = 0
                result.append(token)
            elif negate and len(token) > 2:  # Négation active
                result.append(f"NOT_{token}")
                words_negated += 1

                # Reset si le mot se termine par une ponctuation de fin de phrase
                if ends_with_sentence_punct:
                    negate = False
                    words_negated = 0
                # Limite la portée: max 3 mots sauf intensifieurs
                elif words_negated >= 3 and token_lower not in intensifiers:
                    negate = False
                    words_negated = 0
            else:
                result.append(token)
                # Reset si le mot se termine par une ponctuation
                if ends_with_sentence_punct or token_lower not in intensifiers:
                    negate = False
                    words_negated = 0

        return " ".join(result)

    def handle_emotions(self, text: str) -> str:
        """Normalise les expressions d'émotions pour l'analyse sentiment."""
        # Répétitions de lettres (ex: "sooooo good" -> "soo good")
        text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)

        # Majuscules excessives (ex: "AMAZING" -> "amazing CAPS")
        text = re.sub(r"\b[A-Z]{3,}\b", lambda m: f"{m.group().lower()} CAPS", text)

        # Points d'exclamation multiples
        text = re.sub(r"!{2,}", " EXCITED ", text)

        # Points d'interrogation multiples
        text = re.sub(r"\?{2,}", " CONFUSED ", text)

        return text

    def advanced_preprocess(
        self,
        text: str,
        use_stemming: bool = False,
        use_lemmatization: bool = True,
        remove_stopwords: bool = True,
        handle_negations: bool = True,
        handle_emotions: bool = True,
    ) -> str:
        """
        Prétraitement avancé optimisé pour l'analyse sentiment.
        Répond aux critères d'évaluation CE1: au moins 2 techniques de prétraitement.
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # 1. Expansion des contractions (AVANT le nettoyage)
        text = self.expand_contractions(text)

        # 2. Gestion des émotions/intensité
        if handle_emotions:
            text = self.handle_emotions(text)

        # 3. Gestion des négations (AVANT le nettoyage pour garder la ponctuation)
        if handle_negations:
            text = self.handle_negations(text)

        # 4. Nettoyage de base (supprime la ponctuation)
        cleaned_text = self.clean_tweet(text)

        if not cleaned_text:
            return ""

        # 5. Tokenisation
        tokens = self.tokenizer(cleaned_text)

        # 6. Suppression sélective des stopwords (garder les mots sentiments)
        if remove_stopwords:
            # Garder certains stopwords importants pour le sentiment
            sentiment_stopwords = {"not", "no", "never", "very", "quite", "really", "so", "too"}
            tokens = [
                token
                for token in tokens
                if token.lower() not in self.stop_words or token.lower() in sentiment_stopwords
            ]

        # 7. Application du stemming ou lemmatization
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return " ".join(tokens)

    def preprocess_with_techniques(
        self,
        texts: List[str],
        technique: str = "lemmatization",
        handle_negations: bool = True,
        handle_emotions: bool = True,
    ) -> List[str]:
        """
        Applique différentes techniques de prétraitement optimisées pour le sentiment.

        Args:
            texts: Liste des textes à traiter
            technique: 'stemming', 'lemmatization', ou 'basic'
            handle_negations: Activer la gestion intelligente des négations
            handle_emotions: Activer la préservation des émoticons

        Returns:
            Textes prétraités selon la technique choisie
        """
        if technique == "stemming":
            return [
                self.advanced_preprocess(
                    text,
                    use_stemming=True,
                    use_lemmatization=False,
                    handle_negations=handle_negations,
                    handle_emotions=handle_emotions,
                )
                for text in texts
            ]
        elif technique == "lemmatization":
            return [
                self.advanced_preprocess(
                    text,
                    use_stemming=False,
                    use_lemmatization=True,
                    handle_negations=handle_negations,
                    handle_emotions=handle_emotions,
                )
                for text in texts
            ]
        else:  # basic
            return [self.clean_tweet(text) for text in texts]


def load_sentiment140_data(
    file_path: str, sample_size: Optional[int] = None, logger: Optional[OutputLogger] = None
) -> pd.DataFrame:
    """
    Charge le dataset Sentiment140 avec échantillonnage stratifié.
    Maintient la répartition équilibrée des classes (50% négatif, 50% positif).

    Args:
        file_path: Chemin vers le fichier CSV
        sample_size: Taille de l'échantillon (None = dataset complet)
        logger: Logger pour affichage (optionnel, par défaut PrintLogger)
    """
    if logger is None:
        logger = PrintLogger()

    columns = ["sentiment", "id", "date", "query", "user", "text"]

    # Charger le dataset complet
    logger.info(f"   Chargement du dataset complet...")
    df = pd.read_csv(file_path, names=columns, encoding="latin1")

    # Conversion des labels (0->0, 4->1)
    df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})

    # Sélectionner seulement les colonnes nécessaires
    df = df[["text", "sentiment"]]

    logger.info(f"   Dataset complet chargé: {len(df)} tweets")
    logger.info(f"   Distribution originale: {df['sentiment'].value_counts().to_dict()}")

    # Échantillonnage stratifié si demandé
    if sample_size and sample_size < len(df):
        from sklearn.model_selection import train_test_split

        logger.info(f"   Échantillonnage stratifié de {sample_size} tweets...")

        # Échantillonnage stratifié pour maintenir la répartition 50/50
        df_sample, _ = train_test_split(
            df, test_size=1 - (sample_size / len(df)), random_state=42, stratify=df["sentiment"]
        )

        logger.info(f"   Échantillon stratifié créé: {len(df_sample)} tweets")
        logger.info(f"   Distribution échantillon: {df_sample['sentiment'].value_counts().to_dict()}")

        return df_sample

    return df


def preprocess_dataset(df: pd.DataFrame, cleaner: TextCleaner = None) -> pd.DataFrame:
    """Preprocessing complet du dataset."""
    if cleaner is None:
        cleaner = TextCleaner()

    # Nettoyage des textes
    df = df.copy()
    df["cleaned_text"] = df["text"].apply(cleaner.clean_tweet)

    # Suppression des tweets vides après nettoyage
    df = df[df["cleaned_text"].str.len() > 0]

    return df.reset_index(drop=True)
