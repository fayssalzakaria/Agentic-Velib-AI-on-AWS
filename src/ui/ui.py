import os
from datetime import datetime
from src.ai.assistant import run_unified_assistant
import pandas as pd
import plotly.express as px
import pytz
import streamlit as st

from config import PARIS_TIMEZONE, SOURCE_API, SOURCE_LAKE
from src.ai.chatbot import ask_groq, build_context
from src.data import lake
from src.data.history import get_lake_status, get_station_history
from src.ai.vector_store import extract_station_from_query, build_qdrant_index, ask_with_qdrant
from src.ai.nlp_utils import normalize_text, extract_keywords, detect_time_expression
from src.ai.router import classify_question
def render_sidebar():
    st.sidebar.title("Configuration")

    source = st.sidebar.radio(
        "Source des donnees",
        [SOURCE_API, SOURCE_LAKE],
        index=0,
    )

    st.sidebar.divider()
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

    st.sidebar.divider()
    sidebar_count = st.sidebar.empty()

    return source, filtre_type, filtre_etat, filtre_min_velos, sidebar_count


def render_source_info(source: str, df, info: str | None = None):
    st.sidebar.success(f"{len(df)} stations chargees")

    if source == SOURCE_API:
        st.caption("Source : opendata.paris.fr — temps reel")
    else:
        st.caption(f"Source : lakehouse S3 — {info}")


def render_metrics(df_filtered):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Stations", len(df_filtered))
    col2.metric("Velos dispo", f"{int(df_filtered['numbikesavailable'].sum()):,}")
    col3.metric("Bornes dispo", f"{int(df_filtered['numdocksavailable'].sum()):,}")
    col4.metric("Stations vides", int(df_filtered["is_empty"].sum()))
    col5.metric("Remplissage", f"{df_filtered['bike_ratio'].mean():.1%}")
    st.divider()

def render_map(df_filtered):
    st.subheader("Carte des stations")

    if "lat" in df_filtered.columns and df_filtered["lat"].notna().any():
        map_df = df_filtered[["name", "lat", "lon", "numbikesavailable"]].dropna()
        st.map(
            map_df,
            latitude="lat",
            longitude="lon",
            size="numbikesavailable",
            color="#1D9E75",
        )
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

def render_last_update():
    paris_tz = pytz.timezone(PARIS_TIMEZONE)
    now = datetime.now(paris_tz).strftime("%Y-%m-%d %H:%M")
    st.caption(f"Derniere mise a jour : {now}")

def render_station_detail(df_filtered):
    st.subheader("Recherche et historique de station")

    col1, col2 = st.columns([3, 1])

    with col1:
        search = st.text_input(
            "Nom de la station",
            placeholder="Ex: Bastille, Nation, Mairie de Clichy...",
            key="station_search",
        )

    with col2:
        hours = st.selectbox(
            "Historique",
            [6, 12, 24, 48, 72, 168, 720],
            index=2,
            format_func=lambda x: f"{x}h" if x < 168 else ("7 jours" if x == 168 else "30 jours"),
        )   

    if not search:
        return

    results = df_filtered[df_filtered["name"].str.contains(search.upper(), na=False)]

    if results.empty:
        st.warning(f"Aucune station trouvee pour '{search}'")
        return

    # Selectbox si plusieurs stations matchent
    station_names = sorted(results["name"].unique().tolist())
    if len(station_names) > 1:
        station_name = st.selectbox("Choisir une station", station_names)
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

    # Historique automatique
    with st.spinner(f"Chargement historique {station_name}..."):
        df_history = get_station_history(station_name, hours)

    if df_history.empty:
        if not lake.is_configured():
            st.info("Historique indisponible : lakehouse non configuré (mode temps réel seul).")
        else:
            st.info("Pas encore d'historique pour cette station sur cette période.")
        st.divider()
        return

    collectors = ", ".join(sorted(df_history["ingested_by"].dropna().unique()))
    st.caption(
        f"Evolution sur les {hours} dernieres heures — {len(df_history)} snapshots "
        f"(collectés par : {collectors})"
    )

    fig = px.line(
        df_history,
        x="run_at",
        y=["numbikesavailable", "ebike", "mechanical"],
        labels={"run_at": "Heure", "value": "Nombre", "variable": "Type"},
        color_discrete_map={
            "numbikesavailable": "#1D9E75",
            "ebike": "#185FA5",
            "mechanical": "#EF9F27",
        },
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

def _reset_ai_state():
    for key in ["rag_documents", "rag_docs", "qdrant_client", "qdrant_docs"]:
        st.session_state.pop(key, None)


@st.cache_data(ttl=600, show_spinner=False)
def _network_snapshots(days: int = 7) -> pd.DataFrame:
    return lake.network_snapshots(days)


def render_lake_panel():
    """État du lakehouse : la collecte est automatique (Airflow, GitHub Actions en secours)."""
    st.subheader("Historique (lakehouse)")

    status = get_lake_status()
    if not status.get("available"):
        if lake.is_configured():
            st.warning("Lakehouse inaccessible ou vide — mode temps réel seul.")
        else:
            st.info("Lakehouse non configuré (LAKE_ROOT) — mode temps réel seul.")
        st.divider()
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Snapshots horaires", status["snapshots"])
    col2.metric("Depuis le", f"{status['first_snapshot']:%d/%m/%Y}")
    col3.metric("Dernier snapshot", f"{status['last_snapshot']:%d/%m %H:%M}")
    col4.metric("Collecté par", status["last_ingested_by"])

    if st.button("Rafraichir l'historique"):
        st.cache_data.clear()
        _reset_ai_state()
        st.rerun()

    trend = _network_snapshots(7)
    if not trend.empty:
        fig = px.line(
            trend,
            x="ingested_at",
            y=["total_bikes", "empty_stations"],
            labels={"ingested_at": "Heure", "value": "Nombre", "variable": "Indicateur"},
            color_discrete_map={"total_bikes": "#1D9E75", "empty_stations": "#A32D2D"},
        )
        fig.update_layout(height=260, margin=dict(t=10, b=10))
        st.caption("Réseau sur 7 jours (gold dbt, mis à jour chaque nuit)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()


def _get_qdrant_client_cached():
    if "qdrant_client" not in st.session_state:
        with st.spinner("Connexion Qdrant Cloud..."):
            client, n = build_qdrant_index()
            st.session_state.qdrant_client = client
            st.session_state.qdrant_docs = n
    return st.session_state.get("qdrant_client"), st.session_state.get("qdrant_docs", 0)


def _ensure_rag_documents():
    if "rag_documents" not in st.session_state:
        with st.spinner("Chargement des profils de stations (gold)..."):
            from src.ai.rag import build_rag_index
            docs, n_docs = build_rag_index()
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