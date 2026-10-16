"""
settings.py — Configuration du lakehouse Vélib' (bronze / silver / gold).

LAKE_ROOT pointe soit vers S3 (prod), soit vers un dossier local (dev / tests) :
- prod  : LAKE_ROOT=s3://velib-pipeline-fz-prod-data/lake
- local : LAKE_ROOT=./lake
"""
import os

API_URL = os.environ.get(
    "VELIB_API_URL",
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "velib-disponibilite-en-temps-reel/exports/json",
)

PARIS_TIMEZONE = "Europe/Paris"
AWS_REGION = os.environ.get("AWS_REGION", "eu-north-1")


def lake_root() -> str:
    """Racine du lakehouse, sans slash final. Lue à l'appel pour faciliter les tests."""
    return os.environ.get("LAKE_ROOT", "./lake").rstrip("/")


def is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def bronze_uri() -> str:
    return f"{lake_root()}/bronze/station_status"


def silver_uri() -> str:
    return f"{lake_root()}/silver/station_status"


def gold_uri(table: str) -> str:
    return f"{lake_root()}/gold/{table}.parquet"


def orchestrator() -> str:
    """Qui exécute le pipeline : 'airflow' (principal), 'github-actions' (secours), 'manual'."""
    return os.environ.get("PIPELINE_ORCHESTRATOR", "manual")


def delta_storage_options() -> dict:
    """
    Options delta-rs pour S3.

    Deux orchestrateurs peuvent écrire (Airflow + GitHub Actions en secours).
    Les écritures conditionnelles S3 (If-None-Match) garantissent qu'un seul
    commit Delta gagne pour une version donnée : le second échoue avec
    CommitFailedError au lieu d'écraser silencieusement le premier.
    Pas besoin de table de verrou DynamoDB.
    """
    if not is_s3(lake_root()):
        return {}
    return {
        "AWS_REGION": AWS_REGION,
        "AWS_CONDITIONAL_PUT": "etag",
    }
