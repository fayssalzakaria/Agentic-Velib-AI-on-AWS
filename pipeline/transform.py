"""
transform.py — Bronze (JSON brut) -> Silver (une ligne par station et par snapshot).

Le temps de référence est l'heure du snapshot (ingested_at), arrondie à l'heure :
deux exécutions dans la même heure produisent le même snapshot_id, ce qui rend
l'écriture silver idempotente (MERGE).
"""
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa

from pipeline.settings import PARIS_TIMEZONE, orchestrator

SILVER_SCHEMA = pa.schema([
    ("snapshot_id",       pa.string()),
    ("ingested_at",       pa.timestamp("us", tz="UTC")),
    ("ingested_by",       pa.string()),
    ("station_id",        pa.string()),
    ("name",              pa.string()),
    ("arrondissement",    pa.string()),
    ("capacity",          pa.int32()),
    ("numbikesavailable", pa.int32()),
    ("mechanical",        pa.int32()),
    ("ebike",             pa.int32()),
    ("numdocksavailable", pa.int32()),
    ("is_installed",      pa.bool_()),
    ("is_renting",        pa.bool_()),
    ("is_returning",      pa.bool_()),
    ("is_empty",          pa.bool_()),
    ("is_full",           pa.bool_()),
    ("bike_ratio",        pa.float64()),
    ("lat",               pa.float64()),
    ("lon",               pa.float64()),
    ("last_reported_at",  pa.timestamp("us", tz="UTC")),
    ("hour",              pa.int32()),
    ("weekday",           pa.string()),
    ("is_weekend",        pa.bool_()),
    ("date",              pa.string()),
])


def snapshot_time(now: datetime | None = None) -> datetime:
    """Heure du snapshot en UTC, arrondie à l'heure inférieure."""
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def snapshot_id_for(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y%m%dT%H00Z")


def partition_date(ts: datetime) -> str:
    """Partition silver du snapshot : la date en heure de Paris."""
    return pd.Timestamp(ts).tz_convert(PARIS_TIMEZONE).strftime("%Y-%m-%d")


def _to_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_bool(value) -> bool | None:
    if value is None:
        return None
    return str(value).strip().upper() in {"OUI", "TRUE", "1", "YES"}


def _clean_name(value) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.split()).upper() or None


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes calendaires calculées en heure de Paris à partir de ingested_at."""
    local = df["ingested_at"].dt.tz_convert(PARIS_TIMEZONE)
    df["hour"] = local.dt.hour.astype("int32")
    df["weekday"] = local.dt.day_name()
    df["is_weekend"] = local.dt.dayofweek >= 5
    df["date"] = local.dt.strftime("%Y-%m-%d")
    return df


def to_silver(records: list[dict], ts: datetime) -> pa.Table:
    """Nettoie et enrichit les enregistrements bruts de l'API v2.1."""
    rows = []
    for rec in records:
        station_id = rec.get("stationcode")
        name = _clean_name(rec.get("name"))
        if not station_id or not name:
            continue

        bikes = _to_int(rec.get("numbikesavailable"))
        docks = _to_int(rec.get("numdocksavailable"))
        total = bikes + docks
        geo = rec.get("coordonnees_geo") or {}

        rows.append({
            "station_id":        str(station_id),
            "name":              name,
            "arrondissement":    rec.get("nom_arrondissement_communes"),
            "capacity":          _to_int(rec.get("capacity")),
            "numbikesavailable": bikes,
            "mechanical":        _to_int(rec.get("mechanical")),
            "ebike":             _to_int(rec.get("ebike")),
            "numdocksavailable": docks,
            "is_installed":      _to_bool(rec.get("is_installed")),
            "is_renting":        _to_bool(rec.get("is_renting")),
            "is_returning":      _to_bool(rec.get("is_returning")),
            "is_empty":          bikes == 0,
            "is_full":           docks == 0,
            "bike_ratio":        round(bikes / total, 4) if total > 0 else None,
            "lat":               geo.get("lat"),
            "lon":               geo.get("lon"),
            "last_reported_at":  rec.get("duedate"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Aucune station valide après nettoyage.")

    # Une station n'apparaît qu'une fois par snapshot
    df = df.drop_duplicates(subset=["station_id"], keep="last")

    df["snapshot_id"] = snapshot_id_for(ts)
    df["ingested_at"] = pd.Timestamp(ts).tz_convert("UTC")
    df["ingested_by"] = orchestrator()
    df["last_reported_at"] = pd.to_datetime(df["last_reported_at"], utc=True, errors="coerce")
    df = add_time_columns(df)

    return pa.Table.from_pandas(df[SILVER_SCHEMA.names], schema=SILVER_SCHEMA, preserve_index=False)
