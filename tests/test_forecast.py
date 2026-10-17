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


def test_external_features_weather_and_calendar():
    """Météo de l'heure prédite, jour férié (11 novembre) et vacances scolaires."""
    from src.ml.features import _external

    ts_next = pd.Series(pd.to_datetime(["2026-11-11 10:00", "2026-10-20 10:00", "2026-09-15 10:00"], utc=True))
    weather = pd.DataFrame({"ts": ts_next, "temperature": [8.0, 12.5, 20.0],
                            "precipitation": [0.0, 1.2, 0.0], "wind_speed": [10.0, 5.0, 3.0]})
    school = pd.DataFrame({"start": [pd.Timestamp("2026-10-17", tz="UTC")],
                           "end": [pd.Timestamp("2026-11-02", tz="UTC")]})

    out = _external(ts_next, weather, school)
    assert out["temperature"].tolist() == [8.0, 12.5, 20.0]
    assert out["is_public_holiday"].tolist() == [1.0, 0.0, 0.0]
    assert out["is_school_holiday"].tolist() == [0.0, 1.0, 0.0]

    without = _external(ts_next, None, None)
    assert without["temperature"].isna().all() and without["is_school_holiday"].isna().all()


def test_production_monitoring_end_to_end(tmp_path, monkeypatch):
    """12 heures de « production » simulées : prévision enregistrée à chaque collecte, puis évaluation."""
    from deltalake import write_deltalake

    monkeypatch.setenv("LAKE_ROOT", str(tmp_path).replace("\\", "/"))
    silver = synthetic_silver(n_stations=30, days=16, seed=4)
    ts = pd.to_datetime(silver["ingested_at"], utc=True)
    delta = silver.assign(
        ingested_at=ts, hour=ts.dt.tz_convert(PARIS_TZ).dt.hour.astype("int32"),
        snapshot_id=ts.dt.strftime("%Y%m%dT%H00Z"), is_full=silver["numdocksavailable"] == 0, ingested_by="test",
    )
    instants = sorted(ts.unique())
    # reset_index : sinon pandas transmet l'index filtré comme une colonne de plus
    history = delta[ts < instants[-12]].reset_index(drop=True)
    live = [delta[ts == t].reset_index(drop=True) for t in instants[-12:]]
    silver_uri = str(tmp_path / "silver" / "station_status")
    write_deltalake(silver_uri, history, partition_by=["date"])

    from pipeline.forecast import train_forecast
    assert train_forecast(silver[ts < instants[-12]])["promoted"]

    import src.data.lake as lake
    from pipeline.monitoring import evaluate_production, log_predictions
    logged = 0
    for snapshot in live:
        write_deltalake(silver_uri, snapshot, partition_by=["date"], mode="append")
        monkeypatch.setattr(lake, "_connection", None)
        logged += log_predictions()
    assert logged == 12 * 30

    result = evaluate_production(silver)
    assert result["status"] == "evaluated" and result["scored_predictions"] == 11 * 30  # la dernière heure n'est pas encore passée

    import pyarrow.parquet as pq
    daily = pq.read_table(str(tmp_path / "gold" / "model_monitoring_daily.parquet")).to_pandas()
    calibration = pq.read_table(str(tmp_path / "gold" / "model_calibration.parquet")).to_pandas()
    drift = pq.read_table(str(tmp_path / "gold" / "model_drift.parquet")).to_pandas()
    assert daily["predictions"].sum() == 11 * 30
    assert calibration["predictions"].sum() == 11 * 30
    assert set(drift["status"]) <= {"stable", "à surveiller", "dérive"} and len(drift) >= 8
    assert not {"weekday_next", "past_n"} & set(drift["feature"])  # exclues par construction


def test_psi_detects_a_shift():
    from pipeline.forecast import reference_distribution
    from pipeline.monitoring import psi

    rng = np.random.default_rng(0)
    reference = reference_distribution(pd.DataFrame({"x": rng.normal(0, 1, 5000)}), ["x"])["x"]
    assert psi(reference, rng.normal(0, 1, 2000)) < 0.1      # même distribution : stable
    assert psi(reference, rng.normal(1.5, 1, 2000)) > 0.25   # moyenne décalée : dérive
