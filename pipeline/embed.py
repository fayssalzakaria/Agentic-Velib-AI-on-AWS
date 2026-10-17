"""
embed.py — Embeddings des profils de stations, calculés en batch (job quotidien, après dbt).

L'application n'encode plus que la question de l'utilisateur :
- gold/station_profile_embeddings.parquet : un document par station + son vecteur
- Qdrant : collection réindexée si QDRANT_URL / QDRANT_API_KEY sont définis
"""
import logging
import os

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.lake import _filesystem
from pipeline.settings import gold_uri

logger = logging.getLogger(__name__)


def _read_gold(table: str):
    filesystem, path = _filesystem(gold_uri(table))
    return pq.read_table(path, filesystem=filesystem).to_pandas()


def run_embed() -> dict:
    from src.ai.embeddings import EMBEDDING_MODEL, embed_passages
    from src.ai.profile_docs import documents_to_frame, profiles_version, station_documents

    profiles = _read_gold("gold_station_period_profile")
    documents = station_documents(profiles)
    if not documents:
        logger.warning("Aucun profil de station dans le gold : rien à encoder.")
        return {"documents": 0}

    vectors = embed_passages([d["text"] for d in documents])
    version = f"{profiles_version(profiles)}|{EMBEDDING_MODEL}"
    frame = documents_to_frame(documents, vectors, version, EMBEDDING_MODEL)

    filesystem, path = _filesystem(gold_uri("station_profile_embeddings"))
    with filesystem.open_output_stream(path, compression=None) as out:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), out)
    logger.info(f"{len(frame)} vecteurs écrits ({EMBEDDING_MODEL}) : {gold_uri('station_profile_embeddings')}")

    indexed = None
    if os.environ.get("QDRANT_URL") and os.environ.get("QDRANT_API_KEY"):
        from src.ai.vector_store import _get_qdrant_client, index_documents

        for doc, vector in zip(documents, vectors):
            doc["embedding"] = vector
        # Qdrant est optionnel : les vecteurs sont déjà dans le gold, une panne ne bloque pas le job
        try:
            indexed = index_documents(_get_qdrant_client(), documents, version)
            logger.info(f"Qdrant réindexé : {indexed} points")
        except Exception as e:
            indexed = f"échec : {type(e).__name__}"
            logger.warning(f"Qdrant injoignable, index non mis à jour ({e}). L'application réindexera depuis le gold.")
    else:
        logger.info("QDRANT_URL / QDRANT_API_KEY absents : Qdrant non mis à jour.")

    return {"documents": len(frame), "model": EMBEDDING_MODEL, "qdrant_points": indexed}
