import re
import unicodedata

import pandas as pd


def normalize_name(text: str) -> str:
    """Majuscules, sans accents ni ponctuation : « Châtelet - Rivoli » -> « CHATELET RIVOLI »."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()


def search_stations(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Stations dont le nom contient tous les mots de la recherche, sans tenir compte des accents ni de l'ordre."""
    words = normalize_name(query).split()
    if not words or df.empty:
        return df.iloc[0:0]
    names = df["name"].map(normalize_name)
    mask = pd.Series(True, index=df.index)
    for word in words:
        mask &= names.str.contains(word, regex=False)
    return df[mask]


def apply_filters(
    df: pd.DataFrame,
    filtre_type: str,
    filtre_etat: str,
    filtre_min_velos: int,
) -> pd.DataFrame:
    df_filtered = df.copy()

    if filtre_type == "Mecaniques uniquement":
        df_filtered = df_filtered[df_filtered["mechanical"] > 0]
    elif filtre_type == "Electriques uniquement":
        df_filtered = df_filtered[df_filtered["ebike"] > 0]

    if filtre_etat == "Vides":
        df_filtered = df_filtered[df_filtered["is_empty"]]
    elif filtre_etat == "Pleines":
        df_filtered = df_filtered[df_filtered["is_full"]]
    elif filtre_etat == "Disponibles":
        df_filtered = df_filtered[
            (~df_filtered["is_empty"]) & (~df_filtered["is_full"])
        ]

    df_filtered = df_filtered[df_filtered["numbikesavailable"] >= filtre_min_velos]

    # Deduplique — garde la ligne la plus recente par station
    if "snapshot_id" in df_filtered.columns:
        df_filtered = (
            df_filtered
            .sort_values("snapshot_id", ascending=False)
            .drop_duplicates(subset=["station_id"])
        )
    elif "run_at" in df_filtered.columns:
        df_filtered = (
            df_filtered
            .sort_values("run_at", ascending=False)
            .drop_duplicates(subset=["station_id"])
        )

    return df_filtered