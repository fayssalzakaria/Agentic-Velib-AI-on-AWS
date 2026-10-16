"""
history.py — Historique des stations, lu dans le lakehouse (silver Delta via DuckDB).
"""
import pandas as pd
import streamlit as st

from src.data import lake


@st.cache_data(ttl=600, show_spinner=False)
def get_station_history(station_name: str, hours: int = 24) -> pd.DataFrame:
    return lake.station_history(station_name, hours)


@st.cache_data(ttl=600, show_spinner=False)
def get_available_stations(hours: int = 24) -> list:
    return lake.available_stations(hours)


@st.cache_data(ttl=300, show_spinner=False)
def get_lake_status() -> dict:
    return lake.lake_status()
