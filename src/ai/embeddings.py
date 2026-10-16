"""
embeddings.py — Modèle d'embeddings unique pour le RAG et Qdrant.

Par défaut : intfloat/multilingual-e5-small (gratuit, CPU, 384 dimensions,
entraîné pour la recherche de documents, bon en français).
Le modèle est chargé une seule fois par processus.

Les modèles E5 attendent un préfixe : « query: » pour la question,
« passage: » pour les documents indexés.
"""
import os
from functools import lru_cache

import numpy as np

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")


def _uses_e5_prefixes() -> bool:
    return "e5" in EMBEDDING_MODEL.lower()


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embeddings normalisés des documents (le produit scalaire vaut la similarité cosinus)."""
    if _uses_e5_prefixes():
        texts = [f"passage: {t}" for t in texts]
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)


def embed_query(text: str) -> np.ndarray:
    """Embedding normalisé d'une question."""
    if _uses_e5_prefixes():
        text = f"query: {text}"
    return _model().encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
