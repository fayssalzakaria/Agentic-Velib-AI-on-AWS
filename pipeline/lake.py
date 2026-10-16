"""
lake.py — Écritures dans le lakehouse.

- bronze : JSON brut gzip, partitionné year=/month=/day= (rejouable)
- silver : table Delta Lake partitionnée par date, écrite en MERGE (idempotent)
"""
import gzip
import json
import logging
import os
import random
import time
from datetime import datetime

import pyarrow as pa
from pyarrow import fs as pa_fs
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import CommitFailedError

from pipeline.settings import AWS_REGION, bronze_uri, delta_storage_options, is_s3, lake_root, silver_uri

logger = logging.getLogger(__name__)


def _filesystem(uri: str) -> tuple[pa_fs.FileSystem, str]:
    """FileSystem pyarrow + chemin, pour S3 ou le disque local."""
    if is_s3(uri):
        return pa_fs.S3FileSystem(region=AWS_REGION), uri.removeprefix("s3://")
    return pa_fs.LocalFileSystem(), os.path.abspath(uri)


def _delta_uri(uri: str) -> str:
    return uri if is_s3(uri) else os.path.abspath(uri)


def ensure_local_layout() -> None:
    """En local, crée le dossier gold (DuckDB ne crée pas les dossiers ; inutile sur S3)."""
    if not is_s3(lake_root()):
        os.makedirs(os.path.abspath(f"{lake_root()}/gold"), exist_ok=True)


def write_bronze(records: list[dict], ts: datetime, snapshot_id: str) -> str:
    """Écrit la réponse brute de l'API. Renvoie l'URI du fichier écrit."""
    filesystem, base = _filesystem(bronze_uri())
    path = (
        f"{base}/year={ts:%Y}/month={ts:%m}/day={ts:%d}/"
        f"snapshot_{snapshot_id}.json.gz"
    )
    filesystem.create_dir(path.rsplit("/", 1)[0], recursive=True)

    payload = gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8"))
    # compression=None : sinon pyarrow recompresse les fichiers .gz
    with filesystem.open_output_stream(path, compression=None) as out:
        out.write(payload)

    uri = f"s3://{path}" if is_s3(bronze_uri()) else path
    logger.info(f"Bronze écrit : {uri} ({len(payload):,} octets)")
    return uri


def snapshot_exists(snapshot_id: str, date: str) -> str | None:
    """
    Le snapshot est-il déjà dans le silver ? Renvoie l'orchestrateur qui l'a écrit, sinon None.
    Ne lit que la partition du jour et deux colonnes.
    """
    uri = _delta_uri(silver_uri())
    options = delta_storage_options()
    if not DeltaTable.is_deltatable(uri, storage_options=options):
        return None

    rows = DeltaTable(uri, storage_options=options).to_pyarrow_table(
        columns=["snapshot_id", "ingested_by"],
        # le filtre sur la colonne de partition élague les autres jours
        filters=[("date", "=", date), ("snapshot_id", "=", snapshot_id)],
    )
    if rows.num_rows == 0:
        return None
    return rows.column("ingested_by")[0].as_py() or "inconnu"


def merge_silver(table: pa.Table, retries: int = 6) -> int:
    """
    Ajoute un snapshot à la table silver.
    MERGE sur (date, snapshot_id, station_id) : rejouer un snapshot n'ajoute aucun doublon.
    Si un autre orchestrateur commit au même moment, delta-rs lève CommitFailedError :
    on recharge la table et on rejoue le MERGE, qui n'insère alors que ce qui manque.
    Renvoie le nombre de lignes insérées.
    """
    for attempt in range(1, retries + 1):
        try:
            return _merge_silver_once(table)
        except CommitFailedError as e:
            if attempt == retries:
                raise
            logger.warning(f"Conflit de commit Delta (tentative {attempt}/{retries}) : {e}")
            # attente aléatoire croissante : évite que les écrivains se recroisent
            time.sleep(random.uniform(0.5, 2.0) * attempt)
    return 0


def _merge_silver_once(table: pa.Table) -> int:
    uri = _delta_uri(silver_uri())
    options = delta_storage_options()

    if not DeltaTable.is_deltatable(uri, storage_options=options):
        write_deltalake(uri, table, partition_by=["date"], storage_options=options)
        logger.info(f"Table silver créée : {uri} ({table.num_rows} lignes)")
        return table.num_rows

    dates = sorted(set(table.column("date").to_pylist()))
    date_predicate = " OR ".join(f"t.date = '{d}'" for d in dates)

    metrics = (
        DeltaTable(uri, storage_options=options)
        .merge(
            source=table,
            predicate=(
                f"({date_predicate}) "
                "AND t.snapshot_id = s.snapshot_id "
                "AND t.station_id = s.station_id"
            ),
            source_alias="s",
            target_alias="t",
        )
        .when_not_matched_insert_all()
        .execute()
    )
    inserted = int(metrics.get("num_target_rows_inserted", 0))
    logger.info(f"Silver MERGE : {inserted} lignes insérées")
    return inserted


def maintain_silver(retention_hours: int = 168) -> dict:
    """OPTIMIZE (compaction des petits fichiers horaires) + VACUUM."""
    uri = _delta_uri(silver_uri())
    options = delta_storage_options()

    if not DeltaTable.is_deltatable(uri, storage_options=options):
        logger.warning("Table silver absente, rien à maintenir.")
        return {}

    dt = DeltaTable(uri, storage_options=options)
    compact = dt.optimize.compact()
    removed = dt.vacuum(
        retention_hours=retention_hours,
        dry_run=False,
        enforce_retention_duration=True,
    )
    dt.create_checkpoint()

    result = {
        "files_added": compact.get("numFilesAdded"),
        "files_removed": compact.get("numFilesRemoved"),
        "vacuumed": len(removed),
        "version": dt.version(),
    }
    logger.info(f"Maintenance silver : {result}")
    return result
