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


_NUMBER_WORDS = {"une": 1, "un": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6}


def parse_time_target(question: str) -> dict:
    """
    Horizon demandé dans la question :
    « dans 3 h », « dans deux heures » → {"hours_ahead": 3} ; « à 18h », « vers 18 h 30 » → {"at_hour": 18}.
    """
    q = question.lower()
    match = re.search(r"dans\s+(\d+|une|un|deux|trois|quatre|cinq|six)\s*(?:h\b|heures?)", q)
    if match:
        value = match.group(1)
        return {"hours_ahead": int(value) if value.isdigit() else _NUMBER_WORDS[value]}
    match = re.search(r"(?:à|a|vers)\s+(\d{1,2})\s*h", q)
    if match and int(match.group(1)) < 24:
        return {"at_hour": int(match.group(1))}
    return {}
