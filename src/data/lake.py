"""
lake.py — Lecture du lakehouse Vélib' depuis Streamlit (DuckDB, lecture seule).

- silver/station_status (Delta)            : historique d'une station, dernier snapshot
- gold/gold_station_period_profile.parquet : profils des stations (RAG, Qdrant)
- gold/gold_network_snapshots.parquet      : série temporelle du réseau

LAKE_ROOT = s3://<bucket>/lake (secrets Streamlit) ou un chemin local.
Identifiants S3 : AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY d'un utilisateur en lecture seule.
Toutes les fonctions renvoient un DataFrame vide si le lakehouse est inaccessible :
l'application passe alors en mode temps réel seul.
"""
import logging
import os
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

PARIS_TZ = pytz.timezone("Europe/Paris")
_lock = threading.Lock()
_connection = None


def lake_root() -> str:
    return os.environ.get("LAKE_ROOT", "").rstrip("/")


def is_configured() -> bool:
    return bool(lake_root())


def _silver() -> str:
    return f"delta_scan('{lake_root()}/silver/station_status')"


def _gold(table: str) -> str:
    return f"read_parquet('{lake_root()}/gold/{table}.parquet')"


def _connect():
    """Connexion DuckDB unique, avec extensions et secret S3."""
    global _connection
    if _connection is not None:
        return _connection

    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL delta; LOAD delta;")
    if lake_root().startswith("s3://"):
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(
            "CREATE OR REPLACE SECRET lake (TYPE s3, KEY_ID ?, SECRET ?, REGION ?)",
            [
                os.environ.get("AWS_ACCESS_KEY_ID", ""),
                os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                os.environ.get("AWS_REGION", "eu-north-1"),
            ],
        )
    _connection = con
    return con


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Exécute une requête ; DataFrame vide (et log) si le lakehouse est inaccessible."""
    if not is_configured():
        return pd.DataFrame()
    try:
        with _lock:  # une connexion DuckDB partagée entre les sessions Streamlit
            return _connect().execute(sql, params or []).df()
    except Exception as e:
        logger.warning(f"Lakehouse inaccessible : {e}")
        return pd.DataFrame()


def lake_status() -> dict:
    """Résumé du silver : nombre de snapshots, dernier snapshot et qui l'a collecté."""
    df = query(f"""
        SELECT count(DISTINCT snapshot_id)  AS snapshots,
               min(ingested_at)             AS first_snapshot,
               max(ingested_at)             AS last_snapshot,
               arg_max(ingested_by, ingested_at) AS last_ingested_by
        FROM {_silver()}
    """)
    if df.empty or not df.loc[0, "snapshots"]:
        return {"available": False}
    row = df.iloc[0]
    return {
        "available": True,
        "snapshots": int(row["snapshots"]),
        "first_snapshot": pd.Timestamp(row["first_snapshot"]).tz_convert(PARIS_TZ),
        "last_snapshot": pd.Timestamp(row["last_snapshot"]).tz_convert(PARIS_TZ),
        "last_ingested_by": row["last_ingested_by"],
    }


def station_history(station_name: str, hours: int) -> pd.DataFrame:
    """
    Évolution d'une station ; ne lit que les partitions des jours concernés.
    Nom exact d'abord (plusieurs stations contiennent « BASTILLE »), sinon recherche partielle.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    since_date = since.astimezone(PARIS_TZ).strftime("%Y-%m-%d")
    sql = f"""
        SELECT ingested_at AS run_at, name, numbikesavailable, ebike, mechanical,
               numdocksavailable, bike_ratio, ingested_by
        FROM {_silver()}
        WHERE date >= ? AND ingested_at >= ? AND {{name_filter}}
        ORDER BY ingested_at
    """
    df = query(sql.format(name_filter="name = ?"), [since_date, since, station_name.upper()])
    if df.empty:
        df = query(sql.format(name_filter="name ILIKE ?"), [since_date, since, f"%{station_name.upper()}%"])
    if not df.empty:
        df["run_at"] = pd.to_datetime(df["run_at"], utc=True).dt.tz_convert(PARIS_TZ)
    return df


def available_stations(hours: int = 24) -> list[str]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    df = query(
        f"SELECT DISTINCT name FROM {_silver()} WHERE date >= ? AND ingested_at >= ? ORDER BY name",
        [since.astimezone(PARIS_TZ).strftime("%Y-%m-%d"), since],
    )
    return df["name"].tolist() if not df.empty else []


def latest_snapshot() -> tuple[pd.DataFrame, str]:
    """Dernier snapshot du silver, au format attendu par le dashboard."""
    df = query(f"""
        WITH s AS (SELECT * FROM {_silver()})
        SELECT station_id, name, numbikesavailable, mechanical, ebike, numdocksavailable,
               is_empty, is_full, coalesce(bike_ratio, 0) AS bike_ratio, lat, lon,
               snapshot_id, ingested_at, ingested_by
        FROM s
        WHERE snapshot_id = (SELECT max(snapshot_id) FROM s)
    """)
    if df.empty:
        return df, ""
    ts = pd.Timestamp(df["ingested_at"].iloc[0]).tz_convert(PARIS_TZ)
    return df, f"snapshot du {ts:%Y-%m-%d %H:%M} (collecté par {df['ingested_by'].iloc[0]})"


def station_period_profiles() -> pd.DataFrame:
    """Profils gold par station / type de jour / période (construits par dbt chaque nuit)."""
    return query(f"SELECT * FROM {_gold('gold_station_period_profile')}")


def network_snapshots(days: int = 7) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    df = query(
        f"SELECT * FROM {_gold('gold_network_snapshots')} WHERE ingested_at >= ? ORDER BY ingested_at",
        [since],
    )
    if not df.empty:
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True).dt.tz_convert(PARIS_TZ)
    return df
