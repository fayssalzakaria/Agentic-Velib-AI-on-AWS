"""
forecast.py — Prédiction « risque que la station soit vide dans 1 heure » (côté application et agent).

Le modèle publié est lu dans le registre du lakehouse ; les features du dernier snapshot du silver
sont calculées par src/ml/features.py, exactement comme à l'entraînement.
"""
import time

import pandas as pd

from src.data import lake
from src.ml.features import FEATURES, HORIZON, PARIS_TZ, SILVER_COLUMNS, serving_frame
from src.ml.storage import model_uri, read_json, read_text

MODEL_NAME = "forecast_empty_1h"
_CACHE_SECONDS = 600
_cache: dict = {}


def model_status() -> dict:
    """Dernier entraînement (status.json) et modèle publié (latest.json), s'il existe."""
    return {
        "status": read_json(model_uri(MODEL_NAME, "status.json")) or {},
        "published": read_json(model_uri(MODEL_NAME, "latest.json")),
    }


def _load_booster(version: str):
    import lightgbm as lgb

    key = ("booster", version)
    if key not in _cache:
        _cache[key] = lgb.Booster(model_str=read_text(model_uri(MODEL_NAME, version, "model.txt")))
    return _cache[key]


def _snapshot(ts) -> pd.DataFrame:
    columns = ", ".join(SILVER_COLUMNS)
    return lake.query(
        f"SELECT {columns} FROM {lake._silver()} WHERE date >= ? AND ingested_at = ?",
        [(ts - pd.Timedelta(days=1)).tz_convert(PARIS_TZ).strftime("%Y-%m-%d"), ts.to_pydatetime()],
    )


def _compute_predictions() -> tuple[pd.DataFrame | None, dict]:
    published = read_json(model_uri(MODEL_NAME, "latest.json"))
    if not published:
        return None, {"reason": "Aucun modèle publié pour l'instant."}

    last = lake.query(f"SELECT max(ingested_at) AS ts FROM {lake._silver()}")
    if last.empty or pd.isna(last.loc[0, "ts"]):
        return None, {"reason": "Lakehouse vide ou inaccessible."}
    ts = pd.Timestamp(last.loc[0, "ts"]).tz_convert("UTC")
    ts_next = ts + HORIZON

    current = _snapshot(ts)
    lag1, lag2 = _snapshot(ts - pd.Timedelta(hours=1)), _snapshot(ts - pd.Timedelta(hours=2))
    hour_stats = lake.query(
        f"""SELECT station_id, count(*) AS past_n, sum(is_empty::int) AS past_empty
            FROM {lake._silver()} WHERE hour = ? GROUP BY station_id""",
        [int(ts_next.tz_convert(PARIS_TZ).hour)],
    )

    weather = lake.query(
        f"SELECT * FROM delta_scan('{lake.lake_root()}/silver/weather') WHERE ts = ?",
        [ts_next.to_pydatetime()],
    )
    school = lake.query(f"SELECT * FROM read_parquet('{lake.lake_root()}/silver/school_holidays.parquet')")
    frame = serving_frame(current, lag1, lag2, hour_stats,
                          weather if not weather.empty else None, school if not school.empty else None)
    frame["risk"] = _load_booster(published["version"]).predict(frame[FEATURES])
    info = {
        "snapshot_at": ts.tz_convert(PARIS_TZ),
        "predicted_for": ts_next.tz_convert(PARIS_TZ),
        "model": published,
    }
    return frame.sort_values("risk", ascending=False).reset_index(drop=True), info


def predict_next_hour() -> tuple[pd.DataFrame | None, dict]:
    """Prédictions pour l'heure suivant le dernier snapshot (mises en cache 10 minutes)."""
    cached = _cache.get("predictions")
    if cached and time.time() - cached[0] < _CACHE_SECONDS:
        return cached[1]
    try:
        result = _compute_predictions()
    except Exception as e:
        result = (None, {"reason": f"Prévision indisponible : {e}"})
    _cache["predictions"] = (time.time(), result)
    return result


def forecast_for_station(station_query: str) -> dict:
    """Utilisé par l'IA : risque de vide dans 1 h pour les stations qui correspondent à la recherche."""
    from src.data.filters import search_stations

    predictions, info = predict_next_hour()
    if predictions is None:
        return {"error": info["reason"]}
    matches = search_stations(predictions, station_query) if station_query else predictions.head(10)
    if matches.empty:
        return {"error": f"Station '{station_query}' introuvable dans le dernier snapshot."}
    evaluation = info["model"]["evaluation"]["model"]
    return {
        "source": "modèle LightGBM (prévision)",
        "prevision_pour": f"{info['predicted_for']:%d/%m %H:%M}",
        "a_partir_du_snapshot_de": f"{info['snapshot_at']:%d/%m %H:%M}",
        "fiabilite_modele": f"AUC {evaluation['auc']} sur la période de test",
        "stations": [
            {
                "station": row.name,
                "velos_au_dernier_snapshot": int(row.bikes),
                "risque_vide_dans_1h": f"{row.risk:.0%}",
            }
            for row in matches.head(10).itertuples()
        ],
    }
