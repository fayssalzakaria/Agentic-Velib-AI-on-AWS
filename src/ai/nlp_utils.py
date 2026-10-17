"""
nlp_utils.py — Outils NLP simples pour l'analyse des questions utilisateur.
"""

import re
import unicodedata
from typing import List


FRENCH_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "a", "à",
    "au", "aux", "et", "ou", "en", "sur", "dans", "pour", "avec",
    "est", "ce", "que", "qui", "quoi", "comment", "combien",
    "maintenant", "actuellement", "il", "elle", "y", "t",
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    tokens = tokenize(text)

    return [
        token
        for token in tokens
        if len(token) >= min_length and token not in FRENCH_STOPWORDS
    ]


def detect_time_expression(text: str) -> List[str]:
    normalized = normalize_text(text)

    expressions = []

    keywords = [
        "matin",
        "midi",
        "soir",
        "nuit",
        "hier",
        "aujourd hui",
        "semaine",
        "weekend",
        "week end",
        "heure de pointe",
    ]

    for keyword in keywords:
        if keyword in normalized:
            expressions.append(keyword)

    return expressions
