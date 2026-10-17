"""
features.py — Features du modèle « la station sera-t-elle vide dans 1 heure ? ».

Une seule définition, utilisée à l'entraînement (pipeline/forecast.py) ET à la prédiction
(src/ml/forecast.py) : c'est ce qui évite l'écart entre ce que le modèle a appris et ce qu'on
lui donne en production (training-serving skew).

Aucune feature ne regarde le futur : le taux de vide habituel d'une station n'utilise que les
observations antérieures à l'heure prédite. La météo de l'heure prédite est une prévision
(Open-Meteo) au moment de la prédiction, et une valeur observée à l'entraînement.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

PARIS_TZ = "Europe/Paris"
HORIZON = pd.Timedelta(hours=1)

FEATURES = [
    "bikes", "docks", "ebike", "bike_ratio", "capacity", "is_empty_now",
    "delta_1h", "delta_2h",
    "hour_next", "weekday_next", "is_weekend_next",
    "past_rate_empty", "past_n",
    # Données externes (manquantes = NaN, géré nativement par LightGBM)
    "temperature", "precipitation", "wind_speed",
    "is_public_holiday", "is_school_holiday",
]
WEATHER_COLUMNS = ["temperature", "precipitation", "wind_speed"]
LABEL = "empty_next"

SILVER_COLUMNS = [
    "station_id", "name", "ingested_at", "numbikesavailable", "numdocksavailable",
    "ebike", "bike_ratio", "is_empty", "lat", "lon",
]


def _state(df: pd.DataFrame) -> pd.DataFrame:
    """État d'une station à un instant : colonnes communes aux deux usages."""
    out = pd.DataFrame({
        "station_id": df["station_id"].astype(str),
        "ts": pd.to_datetime(df["ingested_at"], utc=True),
        "bikes": df["numbikesavailable"].astype(float),
        "docks": df["numdocksavailable"].astype(float),
        "ebike": df["ebike"].astype(float),
        "is_empty_now": df["is_empty"].astype(float),
    })
    out["capacity"] = out["bikes"] + out["docks"]
    out["bike_ratio"] = (out["bikes"] / out["capacity"].replace(0, np.nan)).fillna(0.0)
    for column in ("name", "lat", "lon"):
        if column in df.columns:
            out[column] = df[column].values
    return out


def _lag(state: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Nombre de vélos `hours` heures plus tôt, aligné sur l'instant courant."""
    lag = state[["station_id", "ts", "bikes"]].copy()
    lag["ts"] = lag["ts"] + pd.Timedelta(hours=hours)
    return lag.rename(columns={"bikes": f"bikes_lag{hours}"})


def _calendar(ts_next: pd.Series) -> pd.DataFrame:
    local = ts_next.dt.tz_convert(PARIS_TZ)
    return pd.DataFrame({
        "hour_next": local.dt.hour.astype(float),
        "weekday_next": local.dt.dayofweek.astype(float),
        "is_weekend_next": (local.dt.dayofweek >= 5).astype(float),
    }, index=ts_next.index)


@lru_cache(maxsize=8)
def _public_holidays(year: int) -> frozenset:
    import holidays
    return frozenset(holidays.France(years=year))


def _external(ts_next: pd.Series, weather: pd.DataFrame | None, school: pd.DataFrame | None) -> pd.DataFrame:
    """Météo et calendrier (jours fériés, vacances scolaires) de l'heure prédite."""
    out = pd.DataFrame({"ts_next": ts_next}, index=ts_next.index)
    if weather is not None and not weather.empty:
        w = weather.assign(ts_next=pd.to_datetime(weather["ts"], utc=True))[["ts_next", *WEATHER_COLUMNS]]
        out = (out.rename_axis("row").reset_index()
               .merge(w.drop_duplicates("ts_next"), on="ts_next", how="left").set_index("row"))
    for column in WEATHER_COLUMNS:
        if column not in out:
            out[column] = np.nan

    # Calcul par heure distincte (quelques milliers) puis propagation aux lignes (millions)
    hours = pd.Series(ts_next.unique())
    local_dates = hours.dt.tz_convert(PARIS_TZ).dt.date
    index = pd.DatetimeIndex(hours)  # garde le fuseau UTC pour que la correspondance fonctionne
    public = pd.Series([float(d in _public_holidays(d.year)) for d in local_dates], index=index)
    out["is_public_holiday"] = ts_next.map(public).to_numpy()
    if school is not None and not school.empty:
        starts, ends = school["start"].to_numpy(), school["end"].to_numpy()
        vacation = pd.Series([float(((starts <= v) & (v < ends)).any()) for v in hours.to_numpy()],
                             index=index)
        out["is_school_holiday"] = ts_next.map(vacation).to_numpy()
    else:
        out["is_school_holiday"] = np.nan
    return out[[*WEATHER_COLUMNS, "is_public_holiday", "is_school_holiday"]]


def _assemble(frame: pd.DataFrame, weather: pd.DataFrame | None = None,
              school: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dernière étape commune : variations, calendrier et données externes de l'heure prédite."""
    frame["delta_1h"] = frame["bikes"] - frame["bikes_lag1"]
    frame["delta_2h"] = frame["bikes"] - frame["bikes_lag2"]
    ts_next = frame["ts"] + HORIZON
    frame = frame.join(_calendar(ts_next)).join(_external(ts_next, weather, school))
    frame["past_rate_empty"] = frame["past_empty"] / frame["past_n"].replace(0, np.nan)
    return frame


def training_frame(silver: pd.DataFrame, weather: pd.DataFrame | None = None,
                   school: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Une ligne par (station, heure t) avec les features à t et la cible : vide à t + 1 h ?
    Les lignes sans observation à t + 1 h (trou de collecte, dernière heure) ont une cible vide.
    """
    state = _state(silver).sort_values(["station_id", "ts"])
    state["hour_local"] = state["ts"].dt.tz_convert(PARIS_TZ).dt.hour

    # Statistiques « même heure de la journée », en ne comptant que les observations antérieures
    group = state.groupby(["station_id", "hour_local"], sort=False)
    state["prev_n"] = group.cumcount()
    state["prev_empty"] = group["is_empty_now"].cumsum() - state["is_empty_now"]

    # Cible et habitudes de l'heure prédite : on lit la ligne de t + 1 h
    nxt = state[["station_id", "ts", "is_empty_now", "prev_n", "prev_empty"]].copy()
    nxt["ts"] = nxt["ts"] - HORIZON
    nxt = nxt.rename(columns={"is_empty_now": LABEL, "prev_n": "past_n", "prev_empty": "past_empty"})

    frame = (
        state.drop(columns=["prev_n", "prev_empty", "hour_local"])
        .merge(_lag(state, 1), on=["station_id", "ts"], how="left")
        .merge(_lag(state, 2), on=["station_id", "ts"], how="left")
        .merge(nxt, on=["station_id", "ts"], how="left")
    )
    return _assemble(frame, weather, school)


def serving_frame(current: pd.DataFrame, lag1: pd.DataFrame, lag2: pd.DataFrame,
                  hour_stats: pd.DataFrame, weather: pd.DataFrame | None = None,
                  school: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Mêmes features pour le dernier snapshot connu, afin de prédire l'heure suivante.
    hour_stats : par station, nombre d'observations (past_n) et de fois vide (past_empty)
    à l'heure prédite, sur tout l'historique (donc uniquement du passé).
    """
    state = _state(current)
    frame = state.merge(_lag(_state(lag1), 1) if not lag1.empty else _empty_lag(1), on=["station_id", "ts"], how="left")
    frame = frame.merge(_lag(_state(lag2), 2) if not lag2.empty else _empty_lag(2), on=["station_id", "ts"], how="left")
    stats = hour_stats.assign(station_id=hour_stats["station_id"].astype(str))[["station_id", "past_n", "past_empty"]]
    frame = frame.merge(stats, on="station_id", how="left")
    frame[["past_n", "past_empty"]] = frame[["past_n", "past_empty"]].fillna(0)
    return _assemble(frame, weather, school)


def _empty_lag(hours: int) -> pd.DataFrame:
    return pd.DataFrame({"station_id": pd.Series(dtype=str), "ts": pd.Series(dtype="datetime64[ns, UTC]"),
                         f"bikes_lag{hours}": pd.Series(dtype=float)})
