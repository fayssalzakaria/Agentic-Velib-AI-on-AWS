"""
agent.py — Agent LangChain avec tools pour analyse Vélib'
Commit 3 Phase ML : feat(ml): LangChain agent with custom tools
"""
import os
import io
import json
import pandas as pd
import time

from llm_client import call_llm, track_usage, verify_relevance
import pytz
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentTrace:
    question: str = ""
    tools_called: list = field(default_factory=list)
    tool_results: dict = field(default_factory=dict)
    prompt_sent: str = ""
    raw_response: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    relevance_score: int = 0
    relevance_explanation: str = ""
    final_answer: str = ""
    tool_rounds: int = 0
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PARIS_TZ = pytz.timezone("Europe/Paris")
# Le LLM reçoit le nombre total + un échantillon : envoyer des centaines de noms coûte des tokens pour rien
MAX_LISTED_STATIONS = 15
MAX_TOOL_ROUNDS = 3


def _call_groq(messages: list, tools: list = None, max_tokens: int = 1000) -> dict:
    return call_llm(
        messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=0.2,
        timeout=30,
    )


def tool_get_station_info(station_name: str, df: pd.DataFrame) -> str:
    """Retourne les infos temps réel d'une station."""
    if df is None or df.empty:
        return json.dumps({"error": "Données non disponibles"})

    results = df[df["name"].str.contains(station_name.upper(), na=False)]
    if results.empty:
        return json.dumps({"error": f"Station '{station_name}' non trouvée"})

    row = results.iloc[0]
    return json.dumps({
        "station": row["name"],
        "velos_disponibles": int(row["numbikesavailable"]),
        "velos_electriques": int(row["ebike"]),
        "velos_mecaniques": int(row["mechanical"]),
        "bornes_disponibles": int(row["numdocksavailable"]),
        "taux_remplissage": f"{float(row['bike_ratio']):.0%}",
        "etat": "vide" if row["is_empty"] else "pleine" if row["is_full"] else "disponible",
    })


def tool_get_network_stats(df: pd.DataFrame) -> str:
    """Retourne les stats globales du réseau."""
    if df is None or df.empty:
        return json.dumps({"error": "Données non disponibles"})

    return json.dumps({
        "stations_actives": len(df),
        "velos_disponibles": int(df["numbikesavailable"].sum()),
        "velos_electriques": int(df["ebike"].sum()),
        "velos_mecaniques": int(df["mechanical"].sum()),
        "bornes_disponibles": int(df["numdocksavailable"].sum()),
        "taux_remplissage_moyen": f"{df['bike_ratio'].mean():.0%}",
        "stations_vides": int(df["is_empty"].sum()),
        "stations_pleines": int(df["is_full"].sum()),
        "top3_mieux_fournies": df.nlargest(3, "numbikesavailable")["name"].tolist(),
        "top3_plus_vides": df.nsmallest(3, "numbikesavailable")["name"].tolist(),
    })


def tool_search_history(query: str, qdrant_client, rag_documents=None) -> str:
    if rag_documents:
        try:
            from src.ai.rag import hybrid_search
            docs, trace = hybrid_search(
                query, rag_documents, top_k=5,
                use_hyde=True, use_mmr=True, use_rerank=True,
            )
            results = [d["text"] for d in docs]
            sources = [
                f"{d['metadata']['station']} — {d['metadata']['run_at']}"
                for d in docs
            ]
            return json.dumps({
                "resultats": results,
                "sources": sources,
                "techniques_utilisees": trace.get("techniques_used", []),
                "bm25_top3": trace.get("bm25_top5", [])[:3],
                "cosine_top3": trace.get("cosine_top5", [])[:3],
            }, ensure_ascii=False)
        except Exception as e:
            pass

    # Fallback Qdrant simple
    if qdrant_client is None:
        return json.dumps({"error": "Historique non disponible"})
    try:
        from src.ai.vector_store import semantic_search
        docs = semantic_search(query, qdrant_client, n_results=5)
        return json.dumps({"resultats": docs})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_detect_anomalies(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return json.dumps({"error": "Données non disponibles"})

    now = datetime.now(PARIS_TZ)
    heure = now.hour
    anomalies = []

    # Stations vides aux heures de pointe
    if 7 <= heure <= 9 or 17 <= heure <= 19:
        vides = df[df["is_empty"]]["name"].tolist()
        if vides:
            anomalies.append({
                "type": "stations_vides_heure_pointe",
                "heure": heure,
                "stations": vides[:MAX_LISTED_STATIONS],
                "count": len(vides),
            })

    # Stations avec très peu de bornes libres
    presque_pleines = df[df["numdocksavailable"] <= 2]
    if not presque_pleines.empty:
        anomalies.append({
            "type": "stations_presque_pleines",
            "stations": presque_pleines["name"].tolist()[:MAX_LISTED_STATIONS],
            "count": len(presque_pleines),
        })

    # Stations vides
    vides = df[df["is_empty"]]
    if not vides.empty:
        anomalies.append({
            "type": "stations_vides",
            "stations": vides["name"].tolist()[:MAX_LISTED_STATIONS],
            "count": len(vides),
        })

    # Déséquilibre vélos électriques
    total_ebike = int(df["ebike"].sum())
    total_mechanical = int(df["mechanical"].sum())
    total = total_ebike + total_mechanical
    ratio_ebike = total_ebike / max(total, 1)
    if ratio_ebike < 0.2:
        anomalies.append({
            "type": "manque_velos_electriques",
            "ratio_actuel": f"{ratio_ebike:.0%}",
            "seuil": "20%",
        })

    return json.dumps({
        "source": "temps_reel",
        "heure_analyse": f"{heure}h",
        "total_stations": len(df),
        "anomalies": anomalies if anomalies else ["Aucune anomalie detectee"],
    }, ensure_ascii=False)


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_station_info",
            "description": "Obtenir les informations temps reel d'une station Velib specifique",
            "parameters": {
                "type": "object",
                "properties": {
                    "station_name": {
                        "type": "string",
                        "description": "Nom ou partie du nom de la station",
                    }
                },
                "required": ["station_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_network_stats",
            "description": "Obtenir les statistiques globales du reseau Velib en temps reel",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": "Rechercher dans l'historique des donnees Velib pour repondre a des questions temporelles",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question ou recherche sur l'historique",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Detecter les anomalies et comportements inhabituels sur le reseau",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

AGENT_SYSTEM_PROMPT = (
    "Tu es un agent expert du réseau Vélib' à Paris. Tu disposes d'outils et tu choisis toi-même "
    "lesquels appeler. Tu peux procéder en plusieurs étapes : par exemple consulter l'état actuel "
    "d'une station, puis son historique pour dire si la situation est habituelle. "
    "État actuel et anomalies : get_station_info, get_network_stats, detect_anomalies. "
    "Habitudes et tendances : search_history. "
    "N'invente aucune donnée. Réponds en français, de façon concise ; "
    "quand tu listes des stations, cite les plus importantes (10 au maximum) et le nombre total."
)


def _execute_tool(name: str, args: dict, df, qdrant_client, rag_documents) -> str:
    if name == "get_station_info":
        return tool_get_station_info(args.get("station_name", ""), df)
    if name == "get_network_stats":
        return tool_get_network_stats(df)
    if name == "search_history":
        return tool_search_history(args.get("query", ""), qdrant_client, rag_documents)
    if name == "detect_anomalies":
        return tool_detect_anomalies(df)
    return json.dumps({"error": f"Tool {name} inconnu"})


def _format_messages(messages: list[dict]) -> str:
    """Conversation envoyée au LLM, lisible dans la trace."""
    return "\n".join(f"[{m['role'].upper()}] {m.get('content') or ''}" for m in messages if m.get("content"))


def _agent_loop(question: str, df, qdrant_client, rag_documents, trace: AgentTrace) -> tuple[str, str]:
    """
    Boucle agentique : le LLM appelle des outils, lit les résultats et peut en rappeler
    d'autres, jusqu'à MAX_TOOL_ROUNDS tours. Renvoie (réponse, données des outils).
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    all_tool_data = []

    for _ in range(MAX_TOOL_ROUNDS):
        message = _call_groq(messages, tools=TOOLS_SCHEMA)["choices"][0]["message"]
        if not message.get("tool_calls"):
            trace.prompt_sent = _format_messages(messages)
            return message.get("content") or "Aucune réponse.", "\n".join(all_tool_data)

        trace.tool_rounds += 1
        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": message["tool_calls"],
        })
        for tool_call in message["tool_calls"]:
            name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            result = _execute_tool(name, args, df, qdrant_client, rag_documents)
            trace.tools_called.append(name)
            key = name if name not in trace.tool_results else f"{name} #{trace.tools_called.count(name)}"
            trace.tool_results[key] = json.loads(result)
            all_tool_data.append(f"[{name}] {result}")
            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})

    # Nombre de tours atteint : réponse finale sans nouvel outil
    trace.prompt_sent = _format_messages(messages)
    message = _call_groq(messages)["choices"][0]["message"]
    return message.get("content") or "Aucune réponse.", "\n".join(all_tool_data)


def run_agent(question: str, df: pd.DataFrame, qdrant_client=None, rag_documents=None) -> tuple[str, AgentTrace]:
    trace = AgentTrace(question=question)

    if not GROQ_API_KEY:
        return "Cle Groq non configuree.", trace

    started = time.perf_counter()
    with track_usage() as usage:
        try:
            answer, tool_data = _agent_loop(question, df, qdrant_client, rag_documents, trace)
            trace.final_answer = trace.raw_response = answer
            trace.relevance_score, trace.relevance_explanation = _verify_relevance(question, answer, tool_data)
        except Exception as e:
            answer = trace.final_answer = f"Erreur agent : {e}"

    trace.latency_s = round(time.perf_counter() - started, 2)
    trace.usage = usage.as_dict()
    trace.prompt_tokens = usage.prompt_tokens
    trace.completion_tokens = usage.completion_tokens
    trace.total_tokens = usage.total_tokens
    return answer, trace


def _verify_relevance(question: str, answer: str, tool_data: str) -> tuple[int, str]:
    return verify_relevance(
        question=question,
        answer=answer,
        context=tool_data,
        evaluator_name="agent IA Vélib",
    )