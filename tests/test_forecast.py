"""Tests du modèle de prévision : fuite de données, cohérence entraînement/prédiction, bout en bout."""
import numpy as np
import pandas as pd
import pytest

from src.ml.features import FEATURES, HORIZON, PARIS_TZ, serving_frame, training_frame
from tests.synthetic import synthetic_silver


@pytest.fixture(scope="module")
def silver():
    return synthetic_silver(n_stations=9, days=16, seed=1)


def test_label_is_emptiness_one_hour_later(silver):
    frame = training_frame(silver).dropna(subset=["empty_next"])
    observed = silver.assign(ts=pd.to_datetime(silver["ingested_at"], utc=True)).set_index(["station_id", "ts"])
    sample = frame.sample(200, random_state=0)
    expected = [float(observed.loc[(r.station_id, r.ts + HORIZON), "is_empty"]) for r in sample.itertuples()]
    assert sample["empty_next"].tolist() == expected


def test_past_rate_uses_only_earlier_days(silver):
    frame = training_frame(silver).dropna(subset=["empty_next"])
    row = frame.iloc[len(frame) // 2]
    history = silver.assign(ts=pd.to_datetime(silver["ingested_at"], utc=True))
    history = history[history["station_id"] == row.station_id]
    target_ts = row.ts + HORIZON
    same_hour_before = history[
        (history["ts"].dt.tz_convert(PARIS_TZ).dt.hour == target_ts.tz_convert(PARIS_TZ).hour)
        & (history["ts"] < target_ts)
    ]
    assert row.past_n == len(same_hour_before)
    assert row.past_empty == same_hour_before["is_empty"].sum()


def test_serving_features_match_training_features(silver):
    """Même station, même instant : la prédiction voit exactement les features de l'entraînement."""
    frame = training_frame(silver).dropna(subset=["empty_next"])
    ts = frame["ts"].sort_values().unique()[-30]
    data = silver.assign(ts=pd.to_datetime(silver["ingested_at"], utc=True))
    snap = lambda t: data[data["ts"] == t].drop(columns="ts")
    hour_next = (pd.Timestamp(ts) + HORIZON).tz_convert(PARIS_TZ).hour
    past = data[(data["ts"] <= ts) & (data["ts"].dt.tz_convert(PARIS_TZ).dt.hour == hour_next)]
    hour_stats = past.groupby("station_id").agg(past_n=("is_empty", "size"), past_empty=("is_empty", "sum")).reset_index()

    served = serving_frame(snap(ts), snap(ts - pd.Timedelta(hours=1)), snap(ts - pd.Timedelta(hours=2)), hour_stats)
    trained = frame[frame["ts"] == ts]
    left = served.set_index("station_id")[FEATURES].sort_index().astype(float)
    right = trained.set_index("station_id")[FEATURES].sort_index().astype(float)
    np.testing.assert_allclose(left.to_numpy(), right.to_numpy(), equal_nan=True)


def test_training_is_skipped_without_enough_history(tmp_path, monkeypatch, silver):
    monkeypatch.setenv("LAKE_ROOT", str(tmp_path))
    from pipeline.forecast import train_forecast

    short = silver[silver["date"] <= silver["date"].min()]  # un seul jour
    assert train_forecast(short)["status"] == "skipped"


def test_end_to_end_train_publish_and_predict(tmp_path, monkeypatch):
    """Lakehouse simulé : entraînement, publication, puis prédiction avec le code de l'application."""
    from deltalake import write_deltalake

    monkeypatch.setenv("LAKE_ROOT", str(tmp_path).replace("\\", "/"))
    silver = synthetic_silver(n_stations=30, days=16, seed=2)
    local = pd.to_datetime(silver["ingested_at"], utc=True).dt.tz_convert(PARIS_TZ)
    delta = silver.assign(
        ingested_at=pd.to_datetime(silver["ingested_at"], utc=True),
        hour=local.dt.hour.astype("int32"),
        snapshot_id=pd.to_datetime(silver["ingested_at"], utc=True).dt.strftime("%Y%m%dT%H00Z"),
        is_full=silver["numdocksavailable"] == 0,
        ingested_by="test",
    )
    write_deltalake(str(tmp_path / "silver" / "station_status"), delta, partition_by=["date"])

    from pipeline.forecast import train_forecast
    result = train_forecast(silver)
    assert result["status"] == "trained" and result["promoted"]

    import src.data.lake as lake
    import src.ml.forecast as forecast
    monkeypatch.setattr(lake, "_connection", None)
    forecast._cache.clear()

    predictions, info = forecast.predict_next_hour()
    assert predictions is not None, info
    assert len(predictions) == 30 and predictions["risk"].between(0, 1).all()
    assert info["predicted_for"] - info["snapshot_at"] == HORIZON

    answer = forecast.forecast_for_station("STATION 001")
    assert answer["stations"][0]["station"] == "STATION 001"
