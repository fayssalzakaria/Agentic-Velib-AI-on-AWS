import pandas as pd
import requests
import streamlit as st



# CHARGEMENT DES DONNEES
@st.cache_data(ttl=300)
def load_from_api():
    """Charge depuis l'API Vélib' opendata.paris.fr"""
    all_records = []
    for start in range(0, 2000, 1000):
        params = {
            "dataset": "velib-disponibilite-en-temps-reel",
            "rows": 1000,
            "start": start,
        }
        r = requests.get(
            "https://opendata.paris.fr/api/records/1.0/search/",
            params=params,
            timeout=30,
        )
        records = r.json().get("records", [])
        all_records.extend(records)
        if len(records) < 1000:
            break

    rows = []
    for rec in all_records:
        f = rec.get("fields", {})
        rows.append({
            "station_id":        f.get("stationcode"),
            "name":              f.get("name", "").upper(),
            "numbikesavailable": int(f.get("numbikesavailable", 0)),
            "mechanical":        int(f.get("mechanical", 0)),
            "ebike":             int(f.get("ebike", 0)),
            "numdocksavailable": int(f.get("numdocksavailable", 0)),
            "is_empty": int(f.get("numbikesavailable", 0)) == 0,
            "is_full":  int(f.get("numdocksavailable", 0)) == 0,
            "bike_ratio":        round(
                int(f.get("numbikesavailable", 0)) /
                max(int(f.get("numbikesavailable", 0)) + int(f.get("numdocksavailable", 0)), 1),
                2
            ),
            "lat": f.get("coordonnees_geo", [None, None])[0] if f.get("coordonnees_geo") else None,
            "lon": f.get("coordonnees_geo", [None, None])[1] if f.get("coordonnees_geo") else None,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_from_lake():
    """Dernier snapshot collecté (silver du lakehouse). Renvoie (df, description) ou (None, erreur)."""
    from src.data import lake

    if not lake.is_configured():
        return None, "LAKE_ROOT non configuré."
    df, info = lake.latest_snapshot()
    if df.empty:
        return None, "Aucun snapshot dans le lakehouse."
    return df, info
