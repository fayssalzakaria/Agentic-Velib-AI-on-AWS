"""
backfill.py — Import des snapshots historiques (format v1) dans la table silver.

Sources lues dans le bucket S3 existant :
- velib/history/snapshot_YYYYmmdd_HHMMSS.csv     (captures manuelles Streamlit, heure de Paris)
- velib/data/year=/month=/day=/velib_*.csv       (ancienne Lambda, heure UTC dans le nom)

Rejouable sans risque : l'écriture passe par le MERGE silver.
"""
import io
import logging
import os
import re
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pytz
from pyarrow import fs as pa_fs

from pipeline.lake import merge_silver
from pipeline.settings import AWS_REGION, PARIS_TIMEZONE, lake_root
from pipeline.transform import SILVER_SCHEMA, add_time_columns, snapshot_id_for, snapshot_time

logger = logging.getLogger(__name__)

LEGACY_PREFIXES = {
    "velib/history/": "paris",   # snapshot_20250610_143000.csv
    "velib/data/":    "utc",     # velib_20250610_120000.csv
}
_TS_IN_NAME = re.compile(r"(\d{8}_\d{6})\.csv$")


def _legacy_bucket() -> str:
    bucket = os.environ.get("LEGACY_S3_BUCKET")
    if bucket:
        return bucket
    root = lake_root()
    if root.startswith("s3://"):
        return root.removeprefix("s3://").split("/", 1)[0]
    raise RuntimeError("Définir LEGACY_S3_BUCKET ou un LAKE_ROOT S3 pour le backfill.")


def _file_timestamp(key: str, tz_kind: str) -> datetime | None:
    match = _TS_IN_NAME.search(key)
    if not match:
        return None
    naive = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    if tz_kind == "paris":
        return pytz.timezone(PARIS_TIMEZONE).localize(naive).astimezone(timezone.utc)
    return naive.replace(tzinfo=timezone.utc)


def _legacy_to_silver_frame(df: pd.DataFrame, ts: datetime) -> pd.DataFrame:
    bikes = pd.to_numeric(df["numbikesavailable"], errors="coerce").fillna(0).astype("int32")
    docks = pd.to_numeric(df["numdocksavailable"], errors="coerce").fillna(0).astype("int32")
    total = bikes + docks

    out = pd.DataFrame({
        "station_id":        df["station_id"].astype(str),
        "name":              df["name"].astype(str).str.split().str.join(" ").str.upper(),
        "arrondissement":    None,
        "capacity":          total,
        "numbikesavailable": bikes,
        "mechanical":        pd.to_numeric(df.get("mechanical"), errors="coerce").fillna(0).astype("int32"),
        "ebike":             pd.to_numeric(df.get("ebike"), errors="coerce").fillna(0).astype("int32"),
        "numdocksavailable": docks,
        "is_installed":      None,
        "is_renting":        None,
        "is_returning":      None,
        "is_empty":          bikes == 0,
        "is_full":           docks == 0,
        "bike_ratio":        (bikes / total.where(total > 0)).round(4),
        "lat":               None,
        "lon":               None,
        "last_reported_at":  pd.NaT,
    })
    hour_ts = snapshot_time(ts)
    out["snapshot_id"] = snapshot_id_for(hour_ts)
    out["ingested_at"] = pd.Timestamp(hour_ts).tz_convert("UTC")
    out["ingested_by"] = "backfill-v1"
    out["last_reported_at"] = pd.to_datetime(out["last_reported_at"], utc=True)
    return add_time_columns(out)


def backfill() -> dict:
    bucket = _legacy_bucket()
    s3 = pa_fs.S3FileSystem(region=AWS_REGION)
    frames, files = [], 0

    for prefix, tz_kind in LEGACY_PREFIXES.items():
        selector = pa_fs.FileSelector(f"{bucket}/{prefix}", recursive=True, allow_not_found=True)
        for info in s3.get_file_info(selector):
            if info.type != pa_fs.FileType.File or not info.path.endswith(".csv"):
                continue
            ts = _file_timestamp(info.path, tz_kind)
            if ts is None:
                logger.warning(f"Horodatage introuvable, fichier ignoré : {info.path}")
                continue
            with s3.open_input_stream(info.path) as stream:
                raw = pd.read_csv(io.BytesIO(stream.read()), sep=";")
            frames.append(_legacy_to_silver_frame(raw, ts))
            files += 1

    if not frames:
        return {"files": 0, "silver_rows_inserted": 0}

    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["snapshot_id", "station_id"], keep="first")
    )
    table = pa.Table.from_pandas(df[SILVER_SCHEMA.names], schema=SILVER_SCHEMA, preserve_index=False)
    inserted = merge_silver(table)
    return {"files": files, "snapshots": df["snapshot_id"].nunique(), "silver_rows_inserted": inserted}
