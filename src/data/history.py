"""
history.py — État du lakehouse pour la barre latérale (lu dans le silver Delta via DuckDB).
"""
import streamlit as st

from src.data import lake


@st.cache_data(ttl=300, show_spinner=False)
def get_lake_status() -> dict:
    return lake.lake_status()
