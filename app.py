import streamlit as st

from config import MODE_HISTORY
from src.data.filters import apply_filters
from src.ui.history import render_history_page
from src.ui.ui import (
    get_session_snapshot,
    render_ai_tabs,
    render_charts,
    render_downloads,
    render_lake_sidebar,
    render_map,
    render_metrics,
    render_sidebar,
    render_snapshot_header,
    render_station_detail,
)

st.set_page_config(
    page_title="Vélib' Dashboard",
    page_icon="",
    layout="wide",
)

st.title("Vélib' Dashboard Paris")

mode, filters, days = render_sidebar()
render_lake_sidebar()

if mode == MODE_HISTORY:
    st.header("📚 Historique — tous les snapshots du lakehouse")
    render_history_page(days)
    # L'assistant IA a besoin de l'état actuel pour ses outils temps réel
    df_now = get_session_snapshot()
    if df_now is not None:
        render_ai_tabs(df_now)
    st.stop()

st.header("⏱️ Temps réel — snapshot de l'API")
df = render_snapshot_header()
if df is None or df.empty:
    st.stop()

df_filtered = apply_filters(df, *filters)
st.sidebar.info(f"{len(df_filtered)} stations après filtres")

render_metrics(df_filtered)
render_station_detail(df_filtered)
render_map(df_filtered)
render_charts(df_filtered)
render_ai_tabs(df_filtered)
render_downloads(df_filtered)
