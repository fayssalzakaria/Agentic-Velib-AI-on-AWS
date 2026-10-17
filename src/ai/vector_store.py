"""
vector_store.py — Qdrant Cloud : recherche sémantique sur les profils des stations.

Un point par station (document construit depuis le gold dbt, voir profile_docs.py).
La collection est reconstruite automatiquement quand le gold change.
"""
import os
import uuid

import pytz

PARIS_TZ = pytz.timezone("Europe/Paris")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_NAME = "velib_station_profiles"


def _get_qdrant_client():
    if not QDRANT_URL or not QDRANT_API_KEY:
        return None
    try:
        from qdrant_client import QdrantClient
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    except Exception:
        return None


def _collection_exists(client) -> bool:
    try:
        collections = client.get_collections().collections
        return any(c.name == COLLECTION_NAME for c in collections)
    except Exception:
        return False


def _collection_count(client) -> int:
    try:
        return client.count(COLLECTION_NAME).count
    except Exception:
        return 0


def _indexed_version(client) -> str:
    """Version du gold actuellement indexée (stockée dans le payload des points)."""
    try:
        points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=1, with_payload=True)
        return points[0].payload.get("profiles_version", "") if points else ""
    except Exception:
        return ""


def index_documents(client, documents: list[dict], version: str) -> int:
    """(Ré)indexe Qdrant avec des documents dont le vecteur est déjà calculé (doc["embedding"])."""
    from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

    if _collection_exists(client):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=len(documents[0]["embedding"]), distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="station",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"velib/{doc['metadata']['station_id']}")),
            vector=[float(x) for x in doc["embedding"]],
            payload={**doc["metadata"], "text": doc["text"], "profiles_version": version},
        )
        for doc in documents
    ]
    for i in range(0, len(points), 100):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i + 100])
    return _collection_count(client)


def build_qdrant_index(documents: list[dict] | None = None):
    """
    Côté application : se connecte à Qdrant sans jamais encoder de documents.
    L'index est tenu à jour par le pipeline (pipeline/embed.py) ; si la collection manque,
    on la recrée à partir des vecteurs précalculés du gold. Renvoie (client, nombre de points).
    """
    try:
        client = _get_qdrant_client()
        if client is None:
            return None, 0

        if documents is None:
            from src.ai.rag import load_precomputed_documents
            documents = load_precomputed_documents()

        version = documents[0]["profiles_version"] if documents else ""
        if _collection_exists(client) and (not version or _indexed_version(client) == version):
            return client, _collection_count(client)
        if not documents:
            return client, _collection_count(client) if _collection_exists(client) else 0
        return client, index_documents(client, documents, version)

    except Exception:
        return None, 0


def semantic_search(query: str, client, n_results: int = 8) -> list:
    """Recherche sémantique dans Qdrant."""
    if client is None:
        return []

    try:
        from src.ai.embeddings import embed_query

        query_embedding = embed_query(query).tolist()

        station = extract_station_from_query(query, client)

        if station:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=n_results,
                with_payload=True,
                query_filter={
                    "must": [
                        {
                            "key": "station",
                            "match": {"value": station},
                        }
                    ]
                },
            )
        else:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=n_results,
                with_payload=True,
            )

        return [
            r.payload.get("text", "")
            for r in results.points
            if getattr(r, "payload", None)
        ]

    except Exception as e:
        return [f"DEBUG semantic_search erreur : {e}"]

def extract_station_from_query(query: str, client) -> str | None:
    if client is None:
        return None

    try:
        import re
        import unicodedata

        def normalize(text: str) -> str:
            text = text.upper().strip()
            text = unicodedata.normalize("NFKD", text)
            text = "".join(c for c in text if not unicodedata.combining(c))
            text = re.sub(r"[^A-Z0-9\s\-]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        stop_words = {
            "INFO", "INFORMATION", "INFORMATIONS", "SUR", "LA", "LE", "LES",
            "DE", "DES", "DU", "D", "UN", "UNE", "ET", "A", "AU", "AUX",
            "STATION", "DETAIL", "DETAILS", "DONNEE", "DONNEES"
        }

        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5000,
            with_payload=True,
            with_vectors=False,
        )

        stations = set()
        for p in points:
            payload = getattr(p, "payload", {}) or {}
            station = payload.get("station")
            if station:
                stations.add(station)

        q_norm = normalize(query)
        q_words = [
            w for w in q_norm.split()
            if len(w) > 2 and w not in stop_words
        ]

        if not q_words:
            return None

        best_match = None
        best_score = -1

        for station in stations:
            station_norm = normalize(station)
            station_words = set(station_norm.split())

            score = 0

            # Bonus énorme si le nom complet apparaît dans la requête
            if station_norm in q_norm:
                score += 100

            # Mots exacts en commun
            common_words = set(q_words) & station_words
            score += len(common_words) * 15

            # Bonus mot important présent dans la station
            for word in q_words:
                if word in station_norm:
                    score += 5

            # Bonus fort si le dernier mot distinctif matche
            if q_words and q_words[-1] in station_norm:
                score += 20

            # Léger malus si la station est beaucoup plus longue
            score -= max(0, len(station_words) - len(set(q_words)))

            if score > best_score:
                best_score = score
                best_match = station

        return best_match if best_score >= 20 else None

    except Exception:
        return None