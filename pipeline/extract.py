"""
extract.py — Récupération brute de l'API Open Data Vélib' (v2.1, endpoint d'export).
"""
import logging

import requests

from pipeline.settings import API_URL

logger = logging.getLogger(__name__)


def fetch_raw(timeout: int = 60, retries: int = 3) -> list[dict]:
    """Retourne la liste brute des stations, telle que renvoyée par l'API."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(API_URL, timeout=timeout)
            response.raise_for_status()
            records = response.json()
            if not isinstance(records, list) or not records:
                raise ValueError("Réponse API vide ou inattendue.")
            logger.info(f"{len(records)} stations récupérées (tentative {attempt})")
            return records
        except (requests.RequestException, ValueError) as e:
            last_error = e
            logger.warning(f"Échec API Vélib' (tentative {attempt}/{retries}) : {e}")
    raise RuntimeError(f"API Vélib' indisponible : {last_error}")
