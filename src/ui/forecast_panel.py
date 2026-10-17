"""
forecast_panel.py — 🔮 Prévision : risque que chaque station soit vide dans 1 heure (LightGBM).
"""
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from src.data import lake
from src.ml import forecast


@st.cache_data(ttl=600, show_spinner=False)
def _predictions():
    return forecast.predict_next_hour()


@st.cache_data(ttl=600, show_spinner=False)
def _status():
    return forecast.model_status()


def _model_card(model: dict) -> None:
    evaluation = model["evaluation"]
    table = pd.DataFrame([
        {
            "Méthode": v["name"],
            "AUC ↑": v["auc"],
            "Précision moyenne ↑": v["average_precision"],
            "Brier ↓": v["brier"],
            "Précision top 20 ↑": v["precision_at_20"],
        }
        for v in (evaluation["model"], evaluation["persistence"], evaluation["profile"])
    ])
    st.caption(
        f"Version {model['version']} — entraîné sur {model['days_of_history']} jours "
        f"({model['rows']['train']:,} lignes), testé sur les jours suivants jamais vus à l'entraînement. "
        "Publié parce qu'il bat les deux baselines."
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    importance = pd.DataFrame(
        list(model["feature_importance_gain"].items()), columns=["Variable", "Importance (gain)"]
    )
    fig = px.bar(importance, x="Importance (gain)", y="Variable", orientation="h")
    fig.update_layout(title="Ce qui pèse le plus dans la prévision", height=360,
                      yaxis={"autorange": "reversed"}, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=600, show_spinner=False)
def _monitoring(table: str) -> pd.DataFrame:
    return lake.query(f"SELECT * FROM read_parquet('{lake.lake_root()}/gold/{table}.parquet')")


def _production_performance(model: dict) -> None:
    """Le modèle face à la réalité : prévisions enregistrées chaque heure, évaluées chaque nuit."""
    daily = _monitoring("model_monitoring_daily")
    if daily.empty:
        st.info("Pas encore de suivi en production : les prévisions sont enregistrées à chaque collecte "
                "et comparées à la réalité chaque nuit.")
        return

    offline_auc = model["evaluation"]["model"]["auc"]
    fig = px.line(daily, x="date", y="auc", markers=True, labels={"date": "Jour", "auc": "AUC en production"})
    fig.add_hline(y=offline_auc, line_dash="dash", annotation_text=f"AUC au test : {offline_auc}")
    fig.update_layout(title="Qualité jour par jour (prévisions vs réalité)", height=280, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    col_calib, col_drift = st.columns(2)
    calibration = _monitoring("model_calibration")
    if not calibration.empty:
        fig = px.scatter(calibration, x="mean_predicted", y="observed_rate", size="predictions",
                         labels={"mean_predicted": "Risque annoncé", "observed_rate": "Fréquence réelle de vide"})
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        fig.update_layout(title="Calibration (7 derniers jours)", height=320, margin=dict(t=40, b=10))
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        col_calib.plotly_chart(fig, use_container_width=True)
        col_calib.caption("Sur la diagonale : quand le modèle annonce 30 %, la station est vide 30 % du temps.")

    drift = _monitoring("model_drift")
    if not drift.empty:
        col_drift.markdown("**Dérive des données (PSI, dernières 24 h vs entraînement)**")
        icons = {"stable": "🟢", "à surveiller": "🟠", "dérive": "🔴"}
        col_drift.dataframe(
            drift.assign(État=drift["status"].map(lambda v: f"{icons.get(v, '')} {v}"))[["feature", "psi", "État"]]
            .rename(columns={"feature": "Variable", "psi": "PSI"}),
            hide_index=True, use_container_width=True,
        )
        col_drift.caption("PSI < 0,1 : stable · 0,1 à 0,25 : à surveiller · > 0,25 : dérive, réentraînement conseillé.")


def render_forecast() -> None:
    st.subheader("🔮 Prévision : stations vides dans 1 heure")
    if not lake.is_configured():
        st.info("Prévision indisponible : lakehouse non configuré.")
        st.divider()
        return

    predictions, info = _predictions()
    if predictions is None:
        status = _status()["status"]
        if status.get("status") == "skipped":
            st.info(
                f"Modèle pas encore entraîné : {status.get('reason')}. Il le sera automatiquement "
                "dès que la collecte horaire aura assez d'historique (chaque jour de la semaine plusieurs fois)."
            )
        elif status.get("status") == "trained" and not status.get("promoted"):
            st.info("Le dernier modèle entraîné ne bat pas les baselines : il n'a pas été publié.")
        else:
            st.info(info.get("reason", "Prévision indisponible."))
        st.divider()
        return

    st.caption(
        f"Prévision pour **{info['predicted_for']:%H:%M}**, à partir du snapshot de "
        f"{info['snapshot_at']:%d/%m %H:%M} (collecte horaire du lakehouse)."
    )
    at_risk = predictions[(predictions["is_empty_now"] == 0)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Stations à risque (> 50 %)", int((at_risk["risk"] > 0.5).sum()))
    c2.metric("Vides au dernier snapshot", int(predictions["is_empty_now"].sum()))
    c3.metric("Vides attendues (somme des risques)", f"{predictions['risk'].sum():.0f}")

    st.markdown("**À rééquilibrer en priorité** : stations encore approvisionnées mais à fort risque de vide")
    top = at_risk.head(15).assign(
        Risque=lambda d: (d["risk"] * 100).round(0).astype(int).astype(str) + " %",
        Vélos=lambda d: d["bikes"].astype(int),
        Tendance_1h=lambda d: d["delta_1h"].map(lambda x: "—" if pd.isna(x) else f"{x:+.0f}"),
    )[["name", "Vélos", "Tendance_1h", "Risque"]].rename(columns={"name": "Station", "Tendance_1h": "Évolution sur 1 h"})
    st.dataframe(top, hide_index=True, use_container_width=True)

    map_df = predictions.dropna(subset=["lat", "lon"]).copy()
    map_df["color"] = [[int(29 + 134 * r), int(158 - 113 * r), int(117 - 72 * r), 200] for r in map_df["risk"]]
    map_df["risk_txt"] = (map_df["risk"] * 100).round(0).astype(int).astype(str) + " %"
    st.pydeck_chart(
        pdk.Deck(
            layers=[pdk.Layer(
                "ScatterplotLayer",
                data=map_df[["name", "lat", "lon", "color", "risk_txt", "bikes"]],
                get_position="[lon, lat]", get_fill_color="color", get_radius=45,
                radius_min_pixels=3, radius_max_pixels=12, pickable=True,
            )],
            initial_view_state=pdk.ViewState(latitude=48.8566, longitude=2.3522, zoom=11.5),
            tooltip={"html": "<b>{name}</b><br/>Risque d'être vide dans 1 h : {risk_txt}<br/>Vélos au dernier snapshot : {bikes}",
                     "style": {"fontSize": "13px"}},
            map_provider="carto", map_style="light",
        ),
        use_container_width=True,
    )
    st.caption("🟢 faible risque → 🔴 risque élevé d'être vide dans 1 heure")

    with st.expander("Fiche du modèle : performances face aux baselines"):
        _model_card(info["model"])
    with st.expander("📈 Performance en production (monitoring)"):
        _production_performance(info["model"])
    st.divider()
