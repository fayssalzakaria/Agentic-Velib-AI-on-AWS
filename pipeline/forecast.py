"""
forecast.py — Entraînement du modèle « la station sera-t-elle vide dans 1 heure ? » (LightGBM).

1. features à partir de tout le silver (src/ml/features.py, partagé avec la prédiction)
2. découpage dans le temps : entraînement < validation < test (jamais de mélange passé/futur)
3. LightGBM avec early stopping sur la validation
4. comparaison sur le test à deux baselines : persistance et profil habituel de la station
5. publication (latest.json) seulement si le modèle bat les deux baselines

    python -m pipeline.run forecast
"""
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from deltalake import DeltaTable

from pipeline.lake import _delta_uri
from pipeline.settings import delta_storage_options, silver_uri

logger = logging.getLogger(__name__)

MODEL_NAME = "forecast_empty_1h"
MIN_DAYS = int(os.environ.get("FORECAST_MIN_DAYS", 14))
TEST_SHARE, VALID_SHARE = 0.2, 0.15
TOP_K = 20

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "verbose": -1,
}


def _read_silver() -> pd.DataFrame:
    from src.ml.features import SILVER_COLUMNS

    table = DeltaTable(_delta_uri(silver_uri()), storage_options=delta_storage_options())
    return table.to_pandas(columns=SILVER_COLUMNS + ["date"])


def _time_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpage par instants : les derniers snapshots servent au test, ceux d'avant à la validation."""
    instants = np.sort(data["ts"].unique())
    test_start = instants[int(len(instants) * (1 - TEST_SHARE))]
    valid_start = instants[int(len(instants) * (1 - TEST_SHARE - VALID_SHARE))]
    return (
        data[data["ts"] < valid_start],
        data[(data["ts"] >= valid_start) & (data["ts"] < test_start)],
        data[data["ts"] >= test_start],
    )


def _precision_at_k(test: pd.DataFrame, scores: np.ndarray, k: int = TOP_K) -> float:
    """
    Usage opérateur : à chaque snapshot, parmi les stations NON vides, les k jugées les plus à
    risque sont-elles vraiment vides 1 h plus tard ? Moyenne sur les snapshots du test.
    """
    frame = test.assign(score=scores)
    frame = frame[frame["is_empty_now"] == 0]
    hits = [
        group.nlargest(k, "score")["empty_next"].mean()
        for _, group in frame.groupby("ts")
        if len(group) >= k
    ]
    return float(np.mean(hits)) if hits else float("nan")


def _evaluate(name: str, test: pd.DataFrame, scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = test["empty_next"].to_numpy()
    probas = np.clip(scores, 0, 1)
    return {
        "name": name,
        "auc": round(float(roc_auc_score(y, scores)), 4),
        "average_precision": round(float(average_precision_score(y, scores)), 4),
        "brier": round(float(brier_score_loss(y, probas)), 4),
        f"precision_at_{TOP_K}": round(_precision_at_k(test, scores), 4),
    }


def train_forecast(silver: pd.DataFrame | None = None) -> dict:
    import lightgbm as lgb

    from src.ml.features import FEATURES, LABEL, training_frame
    from src.ml.storage import model_uri, write_json, write_text

    silver = _read_silver() if silver is None else silver
    days = silver["date"].nunique() if "date" in silver else silver["ingested_at"].dt.date.nunique()
    if days < MIN_DAYS:
        result = {"status": "skipped", "reason": f"{days} jour{'s' if days > 1 else ''} d'historique sur {MIN_DAYS} requis", "days": int(days)}
        logger.info(f"Modèle non entraîné : {result['reason']}")
        write_json(model_uri(MODEL_NAME, "status.json"), {**result, "checked_at": datetime.now(timezone.utc)})
        return result

    data = training_frame(silver).dropna(subset=[LABEL])
    train, valid, test = _time_split(data)
    logger.info(f"Jeu d'entraînement : {len(train):,} / validation : {len(valid):,} / test : {len(test):,} lignes")

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(
        train[FEATURES], train[LABEL],
        eval_X=valid[FEATURES],
        eval_y=valid[LABEL],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    scores = model.predict_proba(test[FEATURES])[:, 1]

    base_rate = float(train[LABEL].mean())
    results = {
        "model": _evaluate("LightGBM", test, scores),
        "persistence": _evaluate("Persistance (vide maintenant)", test, test["is_empty_now"].to_numpy()),
        "profile": _evaluate("Profil habituel de la station", test, test["past_rate_empty"].fillna(base_rate).to_numpy()),
    }
    best_baseline_auc = max(results["persistence"]["auc"], results["profile"]["auc"])
    promoted = (results["model"]["auc"] > best_baseline_auc
                and results["model"]["brier"] < results["profile"]["brier"])

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    importance = dict(sorted(
        zip(FEATURES, model.booster_.feature_importance(importance_type="gain").round(1).tolist()),
        key=lambda item: -item[1],
    ))
    metrics = {
        "model_name": MODEL_NAME,
        "version": version,
        "trained_at": datetime.now(timezone.utc),
        "promoted": promoted,
        "horizon": "1 heure",
        "days_of_history": int(days),
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "test_period": [test["ts"].min(), test["ts"].max()],
        "base_rate_empty": round(base_rate, 4),
        "best_iteration": int(model.best_iteration_ or PARAMS["n_estimators"]),
        "features": FEATURES,
        "feature_importance_gain": importance,
        "evaluation": results,
    }

    write_text(model_uri(MODEL_NAME, version, "model.txt"), model.booster_.model_to_string())
    write_json(model_uri(MODEL_NAME, version, "metrics.json"), metrics)
    if promoted:
        write_json(model_uri(MODEL_NAME, "latest.json"), metrics)
    write_json(model_uri(MODEL_NAME, "status.json"),
               {"status": "trained", "version": version, "promoted": promoted, "checked_at": datetime.now(timezone.utc)})

    logger.info(
        f"Modèle {version} : AUC {results['model']['auc']} "
        f"(persistance {results['persistence']['auc']}, profil {results['profile']['auc']}) "
        f"→ {'publié' if promoted else 'NON publié (ne bat pas les baselines)'}"
    )
    return {"status": "trained", "version": version, "promoted": promoted,
            "auc": results["model"]["auc"], "best_baseline_auc": best_baseline_auc}
