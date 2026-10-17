"""
langchain_assistant.py — Intégration LangChain avec Groq.

Ce module démontre l'utilisation d'un framework moderne d'IA générative
en complément du pipeline RAG custom du projet.
"""

import os
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


def get_langchain_llm():
    api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY non configurée.")

    return ChatGroq(
        model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0.2,
        api_key=api_key,
    )


def summarize_with_langchain(question: str, context: str) -> Dict[str, Any]:
    llm = get_langchain_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu es un expert en mobilité urbaine et en analyse de données Vélib. "
                "Réponds en français, clairement, sans inventer de données.",
            ),
            (
                "human",
                """
Question :
{question}

Contexte :
{context}

Réponse :
""",
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return {
        "answer": response.content,
        "framework": "LangChain",
        "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
    }
