import os
from datetime import datetime
from src.ai.assistant import run_unified_assistant
import pandas as pd
import plotly.express as px
import pydeck as pdk
import pytz
import streamlit as st

from config import MODE_HISTORY, MODE_REALTIME, PARIS_TIMEZONE
from src.ai.chatbot import ask_groq, build_context
from src.data import lake
from src.data.filters import search_stations
from src.data.data_loader import fetch_api_snapshot
from src.data.history import get_lake_status
from src.ai.vector_store import extract_station_from_query, build_qdrant_index, ask_with_qdrant
from src.ai.nlp_utils import normalize_text, extract_keywords, detect_time_expression
from src.ai.router import classify_question
def render_sidebar():
    """Choix du mode ; les filtres ne concernent que le snapshot temps réel."""
    st.sidebar.title("Configuration")

    mode = st.sidebar.radio(
        "Mode",
        [MODE_REALTIME, MODE_HISTORY],
        index=0,
        captions=[
            "L'état du réseau à l'instant où vous chargez la page",
            "Statistiques sur tous les snapshots collectés",
        ],
    )
    st.sidebar.divider()

    if mode == MODE_HISTORY:
        from src.ui.history import render_period_selector
        return mode, None, render_period_selector()

    st.sidebar.subheader("Filtres")
    filtre_type = st.sidebar.selectbox(
        "Type de velo",
        ["Tous", "Mecaniques uniquement", "Electriques uniquement"],
    )
    filtre_etat = st.sidebar.selectbox(
        "Etat des stations",
        ["Tous", "Disponibles", "Vides", "Pleines"],
        index=0,
    )
    filtre_min_velos = st.sidebar.slider(
        "Minimum de velos disponibles",
        min_value=0,
        max_value=50,
        value=0,
    )
    return mode, (filtre_type, filtre_etat, filtre_min_velos), None


def get_session_snapshot(refresh: bool = False) -> pd.DataFrame | None:
    """Snapshot propre à la session : pris à l'arrivée sur la page, puis à chaque rafraîchissement."""
    if refresh or "snapshot_df" not in st.session_state:
        try:
            with st.spinner("Snapshot de l'API Vélib' en cours..."):
                st.session_state.snapshot_df = fetch_api_snapshot()
            st.session_state.snapshot_at = datetime.now(pytz.timezone(PARIS_TIMEZONE))
        except Exception as e:
            st.error(f"API Vélib' indisponible : {e}")
            return None
    return st.session_state.snapshot_df


def render_snapshot_header() -> pd.DataFrame | None:
    """Heure du snapshot + bouton pour en prendre un nouveau."""
    col_info, col_button = st.columns([4, 1])
    with col_button:
        refresh = st.button("🔄 Rafraîchir le snapshot", use_container_width=True)
    df = get_session_snapshot(refresh=refresh)
    if df is not None:
        col_info.caption(
            f"Snapshot pris le {st.session_state.snapshot_at:%d/%m/%Y à %H:%M:%S} — "
            f"{len(df)} stations (API Open Data Vélib')"
        )
    return df


def render_metrics(df_filtered):
    """Statistiques du snapshot temps réel uniquement."""
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Stations", len(df_filtered))
    col2.metric("Velos dispo", f"{int(df_filtered['numbikesavailable'].sum()):,}")
    col3.metric("Dont electriques", f"{int(df_filtered['ebike'].sum()):,}")
    col4.metric("Bornes dispo", f"{int(df_filtered['numdocksavailable'].sum()):,}")
    col5.metric("Vides / pleines", f"{int(df_filtered['is_empty'].sum())} / {int(df_filtered['is_full'].sum())}")
    col6.metric("Remplissage", f"{df_filtered['bike_ratio'].mean():.1%}")
    st.divider()

def render_map(df_filtered):
    st.subheader("Carte des stations")

    if "lat" in df_filtered.columns and df_filtered["lat"].notna().any():
        map_df = df_filtered[
            ["name", "lat", "lon", "numbikesavailable", "ebike", "numdocksavailable"]
        ].dropna(subset=["lat", "lon"]).copy()
        map_df["lat"] = map_df["lat"].astype(float)
        map_df["lon"] = map_df["lon"].astype(float)
        # Rouge : aucun vélo ; vert : au moins un vélo
        map_df["color"] = [
            [163, 45, 45, 210] if bikes == 0 else [29, 158, 117, 190]
            for bikes in map_df["numbikesavailable"]
        ]

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=45,
            radius_min_pixels=3,
            radius_max_pixels=12,
            pickable=True,
        )
        tooltip = {
            "html": "<b>{name}</b><br/>Vélos : {numbikesavailable} (dont {ebike} électriques)"
                    "<br/>Bornes libres : {numdocksavailable}",
            "style": {"fontSize": "13px"},
        }
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(latitude=48.8566, longitude=2.3522, zoom=11.5),
                tooltip=tooltip,
                map_provider="carto",
                map_style="light",
            ),
            use_container_width=True,
        )
        n_empty = int((map_df["numbikesavailable"] == 0).sum())
        st.caption(f"🟢 vélos disponibles · 🔴 aucun vélo ({n_empty} stations) — survolez une station pour le détail")
    else:
        st.info("Coordonnees non disponibles pour ce snapshot.")

    st.divider()

def render_charts(df_filtered):
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top 10 mieux fournies")
        top10 = df_filtered.nlargest(10, "numbikesavailable")[
            ["name", "numbikesavailable", "ebike"]
        ]
        fig = px.bar(
            top10,
            x="numbikesavailable",
            y="name",
            orientation="h",
            color="ebike",
            color_continuous_scale="teal",
            labels={
                "numbikesavailable": "Velos",
                "name": "Station",
                "ebike": "Electriques",
            },
        )
        fig.update_layout(yaxis={"autorange": "reversed"}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Top 10 plus vides")
        empty10 = df_filtered.nsmallest(10, "numbikesavailable")[
            ["name", "numbikesavailable"]
        ]
        fig2 = px.bar(
            empty10,
            x="numbikesavailable",
            y="name",
            orientation="h",
            color="numbikesavailable",
            color_continuous_scale="reds",
            labels={"numbikesavailable": "Velos", "name": "Station"},
        )
        fig2.update_layout(yaxis={"autorange": "reversed"}, height=400)
        st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Types de velos")
        bike_data = pd.DataFrame(
            {
                "Type": ["Mecaniques", "Electriques"],
                "Nombre": [
                    int(df_filtered["mechanical"].sum()),
                    int(df_filtered["ebike"].sum()),
                ],
            }
        )
        fig3 = px.pie(
            bike_data,
            values="Nombre",
            names="Type",
            color_discrete_sequence=["#185FA5", "#1D9E75"],
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.subheader("Etat des stations")
        status_data = pd.DataFrame(
            {
                "Etat": ["Vides", "Pleines", "Partielles"],
                "Nombre": [
                    int(df_filtered["is_empty"].sum()),
                    int(df_filtered["is_full"].sum()),
                    len(df_filtered)
                    - int(df_filtered["is_empty"].sum())
                    - int(df_filtered["is_full"].sum()),
                ],
            }
        )
        fig4 = px.pie(
            status_data,
            values="Nombre",
            names="Etat",
            color_discrete_sequence=["#A32D2D", "#1D9E75", "#EF9F27"],
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

def render_chatbot(df_filtered):
    st.subheader("Ask Velib Data")

    context = build_context(df_filtered)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ex: Quelle station a le plus de velos electriques ?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                response = ask_groq(question, context)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    st.divider()

def render_downloads(df_filtered):
    st.subheader("Telechargements")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        if st.button("Generer rapport PDF", use_container_width=True):
            with st.spinner("Generation du rapport..."):
                from src.reports.report_generator import generate_pdf_report
                pdf_bytes = generate_pdf_report(df_filtered)
            st.download_button(
                "Telecharger le PDF",
                data=pdf_bytes,
                file_name=f"velib_rapport_{datetime.now(pytz.timezone('Europe/Paris')).strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with col_dl2:
        csv_bytes = df_filtered.to_csv(index=False, sep=";").encode("utf-8")
        st.download_button(
            "Telecharger CSV",
            data=csv_bytes,
            file_name="velib_latest.csv",
            mime="text/csv",
            use_container_width=True,
        )

def render_station_detail(df_filtered):
    st.subheader("Recherche de station")

    search = st.text_input(
        "Nom de la station",
        placeholder="Ex: Bastille, Nation, Mairie de Clichy...",
        key="station_search",
    )

    if not search:
        return

    results = search_stations(df_filtered, search)

    if results.empty:
        st.warning(f"Aucune station trouvee pour '{search}'")
        return

    # Plusieurs stations : on montre combien et on laisse choisir (rien de présélectionné)
    station_names = sorted(results["name"].unique().tolist())
    if len(station_names) > 1:
        bikes = results.groupby("name")["numbikesavailable"].first()
        station_name = st.selectbox(
            f"{len(station_names)} stations trouvées pour « {search} »",
            station_names,
            index=None,
            placeholder="Choisir une station…",
            format_func=lambda name: f"{name} — {int(bikes[name])} vélos",
        )
        if station_name is None:
            st.divider()
            return
    else:
        station_name = station_names[0]

    row = results[results["name"] == station_name].iloc[0]

    # Données temps réel
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Velos dispo", int(row["numbikesavailable"]))
    c2.metric("Dont electriques", int(row["ebike"]))
    c3.metric("Bornes dispo", int(row["numdocksavailable"]))
    c4.metric("Remplissage", f"{row['bike_ratio']:.0%}")

    if row["is_empty"]:
        st.error("Station vide en ce moment")
    elif row["is_full"]:
        st.warning("Station pleine en ce moment")
    else:
        st.success("Station disponible")

    st.caption("Évolution de cette station dans le temps : mode 📚 Historique.")
    st.divider()


def _get_qdrant_client_cached():
    if "qdrant_client" not in st.session_state:
        with st.spinner("Connexion Qdrant Cloud..."):
            client, n = build_qdrant_index()
            st.session_state.qdrant_client = client
            st.session_state.qdrant_docs = n
    return st.session_state.get("qdrant_client"), st.session_state.get("qdrant_docs", 0)


@st.cache_resource(ttl=3600, show_spinner=False)
def _shared_rag_index():
    """Index RAG (documents + embeddings) partagé entre toutes les sessions, reconstruit au plus 1x/h."""
    from src.ai.rag import build_rag_index
    return build_rag_index()


def _ensure_rag_documents():
    if "rag_documents" not in st.session_state:
        with st.spinner("Chargement des profils de stations (gold)..."):
            docs, n_docs = _shared_rag_index()
            st.session_state.rag_documents = docs
            st.session_state.rag_docs = n_docs
    return st.session_state.rag_documents


def _render_rag_content(df_filtered=None):
    st.info("Pour les questions sur l'etat actuel d'une station (anomalies, disponibilite), utilisez l'onglet Agent IA qui a acces aux donnees temps reel.")

    if not _ensure_rag_documents():
        st.info("Historique indisponible : aucun profil de station dans le lakehouse. Chatbot sur les données temps réel :")
        render_chatbot(df_filtered)
        return

    if st.session_state.rag_documents:
        col1, col2 = st.columns([3, 1])
        with col1:
            rag_docs = st.session_state.get("rag_docs", 0)
            st.caption(
                f"Index RAG : {rag_docs} profils de stations — "
                f"HyDE + BM25 + Cosine + RRF + MMR + Reranking"
            )
        with col2:
            if st.button("Rafraichir index"):
                _shared_rag_index.clear()
                for key in ["rag_documents", "rag_docs"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "rag_traces" not in st.session_state:
        st.session_state.rag_traces = []

    for i, msg in enumerate(st.session_state.rag_messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

        if msg["role"] == "assistant" and i // 2 < len(st.session_state.rag_traces):
            trace = st.session_state.rag_traces[i // 2]
            if trace:
                with st.expander("Details du pipeline RAG", expanded=False):
                    st.caption(f"Techniques : {' → '.join(trace.get('techniques_used', []))}")

                    col1, col2, col3 = st.columns(3)
                    tokens = trace.get("tokens", {})
                    col1.metric("Tokens total", tokens.get("total", 0))
                    col2.metric("Prompt tokens", tokens.get("prompt", 0))
                    col3.metric("Completion tokens", tokens.get("completion", 0))

                    if trace.get("hyde_query"):
                        with st.expander("Query HyDE expandée"):
                            st.text(trace["hyde_query"])

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if trace.get("bm25_top5"):
                            st.caption("Top 5 BM25 :")
                            for s in trace["bm25_top5"]:
                                st.caption(f"  • {s}")
                    with col_b:
                        if trace.get("cosine_top5"):
                            st.caption("Top 5 Cosine :")
                            for s in trace["cosine_top5"]:
                                st.caption(f"  • {s}")

                    if trace.get("final_docs"):
                        st.caption("Sources finales utilisées :")
                        for s in trace["final_docs"]:
                            st.caption(f"  • {s}")

    question = st.chat_input(
        "Ex: Bastille est-elle souvent vide le matin ?",
        key="rag_input",
    )

    if question:
        st.session_state.rag_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Pipeline RAG en cours — HyDE + BM25 + Cosine + RRF + MMR + Reranking..."):
                from src.ai.rag import ask_rag_with_qdrant_context
                from src.ai.vector_store import semantic_search
                qdrant_client, _ = _get_qdrant_client_cached()
                qdrant_docs = semantic_search(question, qdrant_client, n_results=10) if qdrant_client else []
                response, trace = ask_rag_with_qdrant_context(
                    question,
                    st.session_state.get("rag_documents"),
                    qdrant_docs,
                )
                st.session_state.rag_traces.append(trace)
            st.write(response)
            if trace and trace.get("relevance_score") is not None:
                score = trace["relevance_score"]
                color = "green" if score >= 80 else "orange" if score >= 50 else "red"
                st.markdown(
                    f"**Pertinence : <span style='color:{color}'>{score}/100</span>** — {trace.get('relevance_explanation','')}",
                    unsafe_allow_html=True,
                )
            st.session_state.rag_messages.append({"role": "assistant", "content": response})
        st.rerun()

def _render_agent_content(df_filtered):
    qdrant_client, _ = _get_qdrant_client_cached()
    rag_documents = st.session_state.get("rag_documents")

    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "agent_traces" not in st.session_state:
        st.session_state.agent_traces = []

    for i, msg in enumerate(st.session_state.agent_messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

        if msg["role"] == "assistant" and i // 2 < len(st.session_state.agent_traces):
            trace = st.session_state.agent_traces[i // 2]
            with st.expander("Details de l'execution", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Tokens total", trace.total_tokens)
                col2.metric("Prompt tokens", trace.prompt_tokens)
                col3.metric("Completion tokens", trace.completion_tokens)

                st.caption(f"Tools utilises : {', '.join(trace.tools_called) if trace.tools_called else 'aucun'}")

                if trace.tool_results:
                    for tool_name, result in trace.tool_results.items():
                        with st.expander(f"Resultat : {tool_name}"):
                            if tool_name == "search_history":
                                techniques = result.get("techniques_utilisees", [])
                                if techniques:
                                    st.caption(f"Techniques : {' → '.join(techniques)}")
                                sources = result.get("sources", [])
                                if sources:
                                    st.caption("Sources :")
                                    for s in sources:
                                        st.caption(f"  • {s}")
                            else:
                                st.json(result)

                if trace.prompt_sent:
                    with st.expander("Prompt envoye au LLM"):
                        st.text(trace.prompt_sent[:2000])

                score = trace.relevance_score
                color = "green" if score >= 80 else "orange" if score >= 50 else "red"
                st.markdown(
                    f"**Pertinence : <span style='color:{color}'>{score}/100</span>**",
                    unsafe_allow_html=True,
                )
                if trace.relevance_explanation:
                    st.caption(trace.relevance_explanation)

    question = st.chat_input(
        "Ex: Y a-t-il des anomalies ? Analyse Bastille sur l'historique.",
        key="agent_input",
    )

    if question:
        st.session_state.agent_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Agent en cours — HyDE + BM25 + Cosine + RRF + MMR si historique requis..."):
                from src.ai.agent import run_agent
                response, trace = run_agent(
                    question, df_filtered, qdrant_client, rag_documents
                )
                st.session_state.agent_traces.append(trace)
            st.write(response)
            st.session_state.agent_messages.append(
                {"role": "assistant", "content": response}
            )
        st.rerun()

def _render_semantic_content():
    qdrant_client, qdrant_docs = _get_qdrant_client_cached()

    if qdrant_client is None:
        st.info("Qdrant non configure ou pas de snapshots disponibles.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Index Qdrant : {qdrant_docs} documents")
    with col2:
        if st.button("Rafraichir Qdrant"):
            for key in ["qdrant_client", "qdrant_docs"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    query = st.text_input(
        "Recherche",
        placeholder="Ex: stations vides le matin, patterns weekend...",
        key="qdrant_search",
    )

    if not query:
        return

    with st.spinner("Recherche semantique..."):
        response, sem_trace = ask_with_qdrant(query, qdrant_client)

    st.write(response)

    if sem_trace.get("relevance_score") is not None:
        score = sem_trace["relevance_score"]
        color = "green" if score >= 80 else "orange" if score >= 50 else "red"
        st.caption(f"Tokens : {sem_trace.get('tokens', 0)}")
        st.markdown(
            f"**Pertinence : <span style='color:{color}'>{score}/100</span>** — {sem_trace.get('relevance_explanation','')}",
            unsafe_allow_html=True,
        )


def render_lake_sidebar():
    st.sidebar.divider()
    st.sidebar.subheader("Lakehouse")

    if not lake.is_configured():
        st.sidebar.info("Non configuré — mode temps réel seul.")
        return

    status = get_lake_status()
    if not status.get("available"):
        st.sidebar.warning("Inaccessible ou vide — mode temps réel seul.")
        return

    st.sidebar.success(
        f"{status['snapshots']} snapshots — dernier à "
        f"{status['last_snapshot']:%H:%M} ({status['last_ingested_by']})"
    )

def _render_unified_assistant_content(df_filtered):
    st.info(
        "Assistant IA unifié : il choisit automatiquement entre temps réel, anomalies, RAG historique et recherche sémantique."
    )

    qdrant_client, _ = _get_qdrant_client_cached()

    try:
        _ensure_rag_documents()
    except Exception:
        st.session_state.rag_documents = None
        st.session_state.rag_docs = 0

    if "unified_messages" not in st.session_state:
        st.session_state.unified_messages = []

    if "unified_traces" not in st.session_state:
        st.session_state.unified_traces = []

    for i, msg in enumerate(st.session_state.unified_messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

        if msg["role"] == "assistant":
            trace_index = i // 2

            if trace_index < len(st.session_state.unified_traces):
                trace = st.session_state.unified_traces[trace_index]

                with st.expander("Détails assistant IA", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Intention", trace.intent)
                    col2.metric("Confiance", f"{trace.confidence:.0%}")
                    col3.metric("Tools", len(trace.tools_used))

                    st.caption(f"Raison du routage : {trace.routing_reason}")
                    from src.ai.nlp_utils import extract_keywords, detect_time_expression, normalize_text

                    st.subheader("Analyse NLP")

                    st.caption("Question normalisée :")
                    st.code(normalize_text(trace.question))

                    keywords = extract_keywords(trace.question)
                    time_expressions = detect_time_expression(trace.question)

                    st.caption("Mots-clés :")
                    st.write(", ".join(keywords) if keywords else "Aucun mot-clé détecté")

                    st.caption("Expressions temporelles :")
                    st.write(", ".join(time_expressions) if time_expressions else "Aucune expression temporelle détectée")

                    if trace.tools_used:
                        st.caption(f"Outils utilisés : {', '.join(trace.tools_used)}")

                    if trace.tokens:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Tokens total", trace.tokens.get("total", 0))
                        c2.metric("Prompt", trace.tokens.get("prompt", 0))
                        c3.metric("Completion", trace.tokens.get("completion", 0))

                    if trace.relevance_score is not None:
                        color = (
                            "green"
                            if trace.relevance_score >= 80
                            else "orange"
                            if trace.relevance_score >= 50
                            else "red"
                        )

                        st.markdown(
                            f"**Pertinence : <span style='color:{color}'>{trace.relevance_score}/100</span>**",
                            unsafe_allow_html=True,
                        )

                        if trace.relevance_explanation:
                            st.caption(trace.relevance_explanation)

                    if trace.tool_results:
                        for tool_name, result in trace.tool_results.items():
                            with st.expander(f"Résultat outil : {tool_name}"):
                                st.json(result)

    question = st.chat_input(
        "Ex: Y a-t-il des anomalies maintenant ? Bastille est-elle souvent vide le matin ?",
        key="unified_assistant_input",
    )

    if question:
        st.session_state.unified_messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Assistant IA en cours..."):
                from src.ai.vector_store import semantic_search

                qdrant_docs = (
                    semantic_search(question, qdrant_client, n_results=10)
                    if qdrant_client
                    else []
                )

                response, trace = run_unified_assistant(
                    question,
                    df_filtered,
                    qdrant_client=qdrant_client,
                    rag_documents=st.session_state.get("rag_documents"),
                    qdrant_docs=qdrant_docs,
                    use_llm_router=False,
                )

                st.session_state.unified_traces.append(trace)

            st.write(response)

            st.session_state.unified_messages.append(
                {"role": "assistant", "content": response}
            )

        st.rerun()

def _render_nlp_content():
    st.info(
        "Analyse NLP locale : normalisation, extraction de mots-clés, détection temporelle et routage d'intention."
    )

    question = st.text_input(
        "Question à analyser",
        placeholder="Ex: Bastille est-elle souvent vide le matin ?",
        key="nlp_analysis_input",
    )

    if not question:
        return

    normalized = normalize_text(question)
    keywords = extract_keywords(question)
    time_expressions = detect_time_expression(question)
    routing = classify_question(question, use_llm=False)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Texte normalisé")
        st.code(normalized)

        st.subheader("Mots-clés extraits")
        if keywords:
            st.write(", ".join(keywords))
        else:
            st.warning("Aucun mot-clé détecté.")

    with col2:
        st.subheader("Expressions temporelles")
        if time_expressions:
            st.write(", ".join(time_expressions))
        else:
            st.info("Aucune expression temporelle détectée.")

        st.subheader("Intention détectée")
        st.metric("Intent", routing.get("intent", "unknown"))
        st.metric("Confiance", f"{routing.get('confidence', 0):.0%}")
        st.caption(routing.get("reason", ""))

    with st.expander("Résultat NLP complet"):
        st.json(
            {
                "question": question,
                "normalized": normalized,
                "keywords": keywords,
                "time_expressions": time_expressions,
                "routing": routing,
            }
        )

def render_ai_tabs(df_filtered):
    st.subheader("Intelligence Artificielle")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Assistant IA",
        "Analyse NLP",
        "Chatbot RAG",
        "Agent IA",
        "Recherche sémantique",
    ])

    with tab1:
        st.caption("Assistant unifié avec routage automatique vers les bons outils")
        _render_unified_assistant_content(df_filtered)

    with tab2:
        st.caption("Analyse locale du langage naturel utilisée par le routeur")
        _render_nlp_content()

    with tab3:
        st.caption("Posez des questions sur l'historique des stations")
        _render_rag_content(df_filtered)

    with tab4:
        st.caption("L'agent choisit automatiquement les bons outils")
        st.caption("Tools : get_station_info · get_network_stats · search_history · detect_anomalies")
        _render_agent_content(df_filtered)

    with tab5:
        st.caption("Recherche sémantique sur les patterns historiques")
        _render_semantic_content()

    st.divider()