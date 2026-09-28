"""Nettoyage de texte sans dépendance lourde : partagé par le prétraitement et l'app."""

import html
import re
import unicodedata

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
APOSTROPHE_RE = re.compile(r"[’‘`´]")
QUOTE_RE = re.compile(r"[«»“”]")
REPEATED_LETTER_RE = re.compile(r"([a-zA-ZÀ-ÿ])\1{2,}")
RATING_RE = re.compile(r"\b\d{1,2}\s?/\s?\d{1,2}\b")
LIGATURES = str.maketrans({"œ": "oe", "Œ": "Oe", "æ": "ae", "Æ": "Ae"})


def clean_text(text: str) -> str:
    """Nettoyage léger (text_clean) : HTML, URL, Unicode, apostrophes, guillemets, espaces.

    La casse, la ponctuation et les emojis sont conservés pour le modèle d'embedding.
    """
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = APOSTROPHE_RE.sub("'", text)
    text = QUOTE_RE.sub('"', text)
    return " ".join(text.split())


def prepare_for_tokens(text: str) -> str:
    """Nettoyage complémentaire avant spaCy : ligatures, lettres répétées, notes chiffrées.

    Attend un texte déjà passé par clean_text. La casse est gardée (utile au NER) :
    les lemmes sont mis en minuscules lors de la tokenisation.
    """
    text = text.translate(LIGATURES)
    text = REPEATED_LETTER_RE.sub(r"\1\1", text)
    text = RATING_RE.sub(" ", text)
    return " ".join(text.split())
