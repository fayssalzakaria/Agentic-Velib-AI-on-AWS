"""
external.py — Données externes pour le modèle de prévision (job quotidien).

- météo horaire de Paris (Open-Meteo, gratuit, sans clé) : 92 jours passés + 2 jours de prévision
  → lake/silver/weather (Delta, MERGE sur l'heure : les prévisions sont remplacées par les valeurs observées)
- vacances scolaires de Paris (API Éducation nationale) → lake/silver/school_holidays.parquet
Les jours fériés sont calculés directement (bibliothèque holidays), sans stockage.
"""
import logging
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from deltalake import DeltaTable, write_deltalake

from pipeline.lake import _delta_uri, _filesystem
from pipeline.settings import delta_storage_options, lake_root

logger = logging.getLogger(__name__)

PARIS_LAT, PARIS_LON = 48.8566, 2.3522
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
SCHOOL_URL = ("https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
              "fr-en-calendrier-scolaire/exports/json")


def weather_uri() -> str:
    return f"{lake_root()}/silver/weather"


def school_holidays_uri() -> str:
    return f"{lake_root()}/silver/school_holidays.parquet"


def fetch_weather(past_days: int = 92, forecast_days: int = 2) -> pd.DataFrame:
    response = requests.get(WEATHER_URL, timeout=30, params={
        "latitude": PARIS_LAT, "longitude": PARIS_LON, "timezone": "UTC",
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "past_days": past_days, "forecast_days": forecast_days,
    })
    response.raise_for_status()
    hourly = response.json()["hourly"]
    return pd.DataFrame({
        "ts": pd.to_datetime(hourly["time"], utc=True),
        "temperature": hourly["temperature_2m"],
        "precipitation": hourly["precipitation"],
        "wind_speed": hourly["wind_speed_10m"],
        "fetched_at": pd.Timestamp.now(tz="UTC"),
    }).dropna(subset=["temperature"])


def merge_weather(weather: pd.DataFrame) -> int:
    uri, options = _delta_uri(weather_uri()), delta_storage_options()
    table = pa.Table.from_pandas(weather, preserve_index=False)
    if not DeltaTable.is_deltatable(uri, storage_options=options):
        write_deltalake(uri, table, storage_options=options)
        return len(weather)
    metrics = (
        DeltaTable(uri, storage_options=options)
        .merge(source=table, predicate="t.ts = s.ts", source_alias="s", target_alias="t")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )
    return int(metrics.get("num_target_rows_inserted", 0)) + int(metrics.get("num_target_rows_updated", 0))


def fetch_school_holidays() -> pd.DataFrame:
    response = requests.get(SCHOOL_URL, timeout=60, params={
        "where": 'location="Paris"', "select": "description,start_date,end_date,population",
    })
    response.raise_for_status()
    frame = pd.DataFrame(response.json())
    # Les lignes « Enseignants » (pré-rentrée) ne concernent pas les usagers
    frame = frame[frame["population"].fillna("-").isin(["-", "Élèves"])]
    return pd.DataFrame({
        "description": frame["description"],
        "start": pd.to_datetime(frame["start_date"], utc=True),
        "end": pd.to_datetime(frame["end_date"], utc=True),
    }).drop_duplicates().sort_values("start").reset_index(drop=True)


def run_external() -> dict:
    result = {}
    try:
        result["weather_rows"] = merge_weather(fetch_weather())
    except Exception as e:  # une API externe en panne ne doit pas bloquer le job quotidien
        logger.warning(f"Météo non mise à jour : {e}")
        result["weather_error"] = str(e)
    try:
        holidays = fetch_school_holidays()
        filesystem, path = _filesystem(school_holidays_uri())
        if not school_holidays_uri().startswith("s3://"):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with filesystem.open_output_stream(path, compression=None) as out:
            pq.write_table(pa.Table.from_pandas(holidays, preserve_index=False), out)
        result["school_holiday_periods"] = len(holidays)
    except Exception as e:
        logger.warning(f"Vacances scolaires non mises à jour : {e}")
        result["school_holidays_error"] = str(e)
    logger.info(f"Données externes : {result}")
    return result


def read_weather() -> pd.DataFrame | None:
    uri, options = _delta_uri(weather_uri()), delta_storage_options()
    if not DeltaTable.is_deltatable(uri, storage_options=options):
        return None
    return DeltaTable(uri, storage_options=options).to_pandas()


def read_school_holidays() -> pd.DataFrame | None:
    filesystem, path = _filesystem(school_holidays_uri())
    if filesystem.get_file_info(path).type.name == "NotFound":
        return None
    return pq.read_table(path, filesystem=filesystem).to_pandas()
