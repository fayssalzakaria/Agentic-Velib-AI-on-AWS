"""Données silver simulées pour tester le modèle de prévision sans attendre des semaines de collecte."""
import numpy as np
import pandas as pd


def synthetic_silver(n_stations: int = 60, days: int = 21, seed: int = 0) -> pd.DataFrame:
    """
    Stations « départ » (se vident le matin en semaine), « arrivée » (le soir) et neutres,
    avec une dynamique horaire bruitée.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-09-01", tz="Europe/Paris").tz_convert("UTC")
    instants = pd.date_range(start, periods=days * 24, freq="h")
    rows = []
    for s in range(n_stations):
        kind, capacity = s % 3, int(rng.integers(15, 40))
        bikes = capacity / 2
        for ts in instants:
            local = ts.tz_convert("Europe/Paris")
            weekday = local.dayofweek < 5
            drift = 0.0
            if weekday and kind == 0 and 6 <= local.hour <= 9:
                drift = -capacity * 0.25
            elif weekday and kind == 1 and 16 <= local.hour <= 19:
                drift = -capacity * 0.25
            elif local.hour in (11, 22):
                drift = (capacity / 2 - bikes) * 0.6
            bikes = float(np.clip(bikes + drift + rng.normal(0, 2), 0, capacity))
            b = int(round(bikes))
            rows.append({
                "station_id": f"S{s:03d}", "name": f"STATION {s:03d}", "ingested_at": ts,
                "numbikesavailable": b, "numdocksavailable": capacity - b, "ebike": b // 3,
                "bike_ratio": b / capacity, "is_empty": b == 0, "lat": 48.85, "lon": 2.35,
                "date": local.strftime("%Y-%m-%d"),
            })
    return pd.DataFrame(rows)
