"""
api/main.py — API REST locale pour exposer l'assistant IA Vélib

Endpoints :
- GET /health
- POST /ask
- POST /route
"""

import os
import sys
from functools import lru_cache
from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel

# Permet d'importer le projet depuis la racine
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.ai.langchain_assistant import summarize_with_langchain
from src.data.data_loader import fetch_api_snapshot
from src.ai.assistant import run_unified_assistant
from src.ai.rag import build_rag_index
from src.ai.router import classify_question
from src.ai.nlp_utils import extract_keywords, detect_time_expression, normalize_text

app = FastAPI(
    title="Vélib AI Assistant API",
    description="API REST pour l'assistant IA RAG/Agents du projet Vélib.",
    version="2.2.0",
)


class AskRequest(BaseModel):
    question: str
    use_llm_router: Optional[bool] = False

class NLPRequest(BaseModel):
    text: str

class AskResponse(BaseModel):
    answer: str
    intent: str
    rule: str = ""
    matched_keywords: list[str] = []
    confidence: Optional[float] = None
    routing_reason: str
    tools_used: list[str]
    relevance_score: Optional[int] = None
    relevance_explanation: str = ""

class LangChainRequest(BaseModel):
    question: str
    context: str

@lru_cache(maxsize=1)
def _rag_documents():
    """Profils de stations du lakehouse, chargés une fois par processus (None si indisponible)."""
    documents, _ = build_rag_index()
    return documents


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "velib-ai-assistant-api",
    }


@app.post("/route")
def route_question(payload: AskRequest) -> Dict[str, Any]:
    routing = classify_question(
        payload.question,
        use_llm=payload.use_llm_router,
    )

    return routing


@app.post("/ask", response_model=AskResponse)
def ask_assistant(payload: AskRequest) -> AskResponse:
    df = fetch_api_snapshot()

    answer, trace = run_unified_assistant(
        payload.question,
        df,
        qdrant_client=None,
        rag_documents=_rag_documents(),
        qdrant_docs=[],
        use_llm_router=payload.use_llm_router,
    )

    return AskResponse(
        answer=answer,
        intent=trace.intent,
        rule=trace.rule,
        matched_keywords=trace.matched_keywords,
        confidence=trace.confidence,
        routing_reason=trace.routing_reason,
        tools_used=trace.tools_used,
        relevance_score=trace.relevance_score,
        relevance_explanation=trace.relevance_explanation,
    )

@app.post("/ask/langchain")
def ask_langchain(payload: LangChainRequest) -> Dict[str, Any]:
    return summarize_with_langchain(
        question=payload.question,
        context=payload.context,
    )
@app.post("/nlp/analyze")
def analyze_nlp(payload: NLPRequest) -> Dict[str, Any]:
    return {
        "text": payload.text,
        "normalized": normalize_text(payload.text),
        "keywords": extract_keywords(payload.text),
        "time_expressions": detect_time_expression(payload.text),
    }