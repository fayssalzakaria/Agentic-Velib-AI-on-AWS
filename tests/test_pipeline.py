"""Tests du pipeline lakehouse : transformation silver et idempotence du MERGE."""
from datetime import datetime, timezone

import pytest
from deltalake import DeltaTable

from pipeline.transform import snapshot_id_for, snapshot_time, to_silver

RECORDS = [
    {
        "stationcode": "11104", "name": "  Bastille -  Biscornet ", "capacity": 20,
        "numbikesavailable": 0, "numdocksavailable": 20, "mechanical": 0, "ebike": 0,
        "is_installed": "OUI", "is_renting": "OUI", "is_returning": "OUI",
        "duedate": "2026-10-16T07:58:00+00:00",
        "coordonnees_geo": {"lon": 2.36, "lat": 48.85},
        "nom_arrondissement_communes": "Paris",
    },
    {
        "stationcode": "16107", "name": "Benjamin Godard", "capacity": 35,
        "numbikesavailable": 35, "numdocksavailable": 0, "mechanical": 30, "ebike": 5,
        "is_installed": "OUI", "is_renting": "OUI", "is_returning": "NON",
        "duedate": "2026-10-16T07:50:00+00:00", "coordonnees_geo": None,
    },
    {"stationcode": None, "name": "SANS CODE"},  # ignorée
]

TS = datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)  # 10h à Paris


def test_snapshot_time_floors_to_hour():
    ts = snapshot_time(datetime(2026, 10, 16, 8, 47, 12, tzinfo=timezone.utc))
    assert ts == TS
    assert snapshot_id_for(ts) == "20261016T0800Z"


def test_to_silver_cleans_and_enriches():
    rows = {r["station_id"]: r for r in to_silver(RECORDS, TS).to_pylist()}

    assert set(rows) == {"11104", "16107"}
    bastille, godard = rows["11104"], rows["16107"]

    assert bastille["name"] == "BASTILLE - BISCORNET"
    assert bastille["is_empty"] and not bastille["is_full"]
    assert bastille["bike_ratio"] == 0
    assert godard["is_full"] and godard["is_returning"] is False
    assert godard["lat"] is None

    # Colonnes calendaires en heure de Paris
    assert bastille["hour"] == 10
    assert bastille["weekday"] == "Friday"
    assert bastille["date"] == "2026-10-16"


def test_to_silver_rejects_empty_payload():
    with pytest.raises(ValueError):
        to_silver([{"stationcode": None}], TS)


def test_merge_silver_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_ROOT", str(tmp_path))
    from pipeline.lake import merge_silver

    table = to_silver(RECORDS, TS)
    assert merge_silver(table) == 2
    assert merge_silver(table) == 0  # même snapshot rejoué

    later = to_silver(RECORDS, datetime(2026, 10, 16, 9, tzinfo=timezone.utc))
    assert merge_silver(later) == 2

    dt = DeltaTable(str(tmp_path / "silver" / "station_status"))
    assert dt.to_pyarrow_table().num_rows == 4


def test_snapshot_exists_reports_writer(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_ROOT", str(tmp_path))
    monkeypatch.setenv("PIPELINE_ORCHESTRATOR", "airflow")
    from pipeline.lake import merge_silver, snapshot_exists

    assert snapshot_exists("20261016T0800Z", "2026-10-16") is None
    merge_silver(to_silver(RECORDS, TS))
    assert snapshot_exists("20261016T0800Z", "2026-10-16") == "airflow"
    assert snapshot_exists("20261016T0900Z", "2026-10-16") is None


def test_fallback_ingest_skips_when_primary_already_ran(tmp_path, monkeypatch):
    """GitHub Actions (secours) ne réécrit rien si Airflow a déjà collecté l'heure."""
    import pipeline.run as run
    from pipeline.lake import merge_silver

    monkeypatch.setenv("LAKE_ROOT", str(tmp_path))
    monkeypatch.setattr(run, "snapshot_time", lambda: TS)
    monkeypatch.setattr(run, "fetch_raw", lambda: pytest.fail("ne doit pas appeler l'API"))

    monkeypatch.setenv("PIPELINE_ORCHESTRATOR", "airflow")
    merge_silver(to_silver(RECORDS, TS))

    monkeypatch.setenv("PIPELINE_ORCHESTRATOR", "github-actions")
    result = run.ingest()
    assert result == {"snapshot_id": "20261016T0800Z", "skipped": True, "already_ingested_by": "airflow"}


@pytest.mark.parametrize("question, intent", [
    ("Bastille est-elle souvent vide le matin ?", "historical"),
    ("Quelle est la tendance à Nation le week-end ?", "historical"),
    ("Y a-t-il des stations vides maintenant ?", "anomaly"),
    ("Y a-t-il des anomalies ?", "anomaly"),
    ("Combien de vélos à Bastille en ce moment ?", "realtime"),
    ("Génère un rapport PDF", "report"),
])
def test_router_intents(question, intent):
    from src.ai.router import classify_question_rule_based
    assert classify_question_rule_based(question)["intent"] == intent


def test_search_stations_ignores_accents_order_and_punctuation():
    import pandas as pd
    from src.data.filters import search_stations

    df = pd.DataFrame({"name": [
        "ATHÈNES - CLICHY", "MAIRIE DE CLICHY", "BESSIÈRES - PORTE DE CLICHY",
        "PLACE DE LA RÉPUBLIQUE - VOLTAIRE", "BASTILLE - BISCORNET",
    ]})
    assert len(search_stations(df, "clichy")) == 3
    assert search_stations(df, "porte clichy")["name"].tolist() == ["BESSIÈRES - PORTE DE CLICHY"]
    assert search_stations(df, "republique")["name"].tolist() == ["PLACE DE LA RÉPUBLIQUE - VOLTAIRE"]
    assert search_stations(df, "athenes")["name"].tolist() == ["ATHÈNES - CLICHY"]
    assert search_stations(df, "  ").empty
