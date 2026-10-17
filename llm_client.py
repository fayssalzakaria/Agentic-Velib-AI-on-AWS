"""
llm_client.py — Client LLM centralisé pour Groq

Objectif :
- Éviter de répéter requests.post(...) dans agent.py, rag.py, chatbot.py, report_generator.py
- Centraliser le modèle, l'URL, les erreurs, le JSON parsing et le calcul des tokens
"""

import json
import os
import re
from typing import Any

import requests


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = os.environ.get(
    "GROQ_URL",
    "https://api.groq.com/openai/v1/chat/completions",
)

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Les modèles qui raisonnent consomment des tokens de réflexion dans la même limite
# que la réponse : on réduit la réflexion et on garantit un minimum de tokens.
REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "low")
MIN_TOKENS_REASONING_MODEL = 1024
_THINK_TAGS = re.compile(r"<think>.*?</think>", re.DOTALL)


def _is_reasoning_model(model: str) -> bool:
    return "gpt-oss" in model.lower()


def _clean_content(content: str | None) -> str:
    """Retire une éventuelle réflexion laissée dans le texte (<think>…</think>)."""
    return _THINK_TAGS.sub("", content or "").strip()


def _raise_for_groq_error(response: requests.Response) -> None:
    """Remonte le message d'erreur de Groq (modèle inconnu, clé invalide, quota…)."""
    if response.ok:
        return
    try:
        detail = response.json().get("error", {}).get("message", "")
    except ValueError:
        detail = response.text[:300]
    raise RuntimeError(f"Groq {response.status_code} : {detail or response.reason}")


def is_llm_configured() -> bool:
    """
    Vérifie si la clé Groq est disponible.
    """
    return bool(GROQ_API_KEY)


def call_llm(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = "auto",
    max_tokens: int = 1000,
    temperature: float = 0.2,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Appelle le modèle Groq et retourne la réponse JSON complète.

    Utilisé pour :
    - chatbot simple
    - RAG
    - agent IA avec tools
    - génération de rapport
    - évaluation de pertinence
    """

    if not GROQ_API_KEY:
        raise RuntimeError("Clé GROQ_API_KEY non configurée.")

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"

    if _is_reasoning_model(model):
        payload["reasoning_effort"] = REASONING_EFFORT
        payload["max_tokens"] = max(max_tokens, MIN_TOKENS_REASONING_MODEL)

    response = requests.post(
        GROQ_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
        json=payload,
        timeout=timeout,
    )

    _raise_for_groq_error(response)
    data = response.json()

    # Réponse normalisée : texte sans réflexion, et pas de champ « reasoning »
    # (l'agent renvoie ce message à Groq lors du second appel)
    for choice in data.get("choices", []):
        message = choice.get("message", {})
        message.pop("reasoning", None)
        message["content"] = _clean_content(message.get("content"))
    return data


def call_llm_text(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    timeout: int = 30,
) -> tuple[str, dict[str, int]]:
    """
    Appel simple texte -> texte.

    Retourne :
    - réponse texte
    - informations tokens
    """

    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    data = call_llm(
        messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )

    answer = data["choices"][0]["message"].get("content", "")
    usage = data.get("usage", {})

    tokens = {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }

    return answer, tokens


def call_llm_json(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 500,
    temperature: float = 0.1,
    timeout: int = 30,
) -> tuple[dict[str, Any], dict[str, int]]:
    """
    Appelle le LLM en demandant une réponse JSON.

    Cette fonction nettoie les éventuels blocs Markdown :
    ```json
    {...}
    ```
    """

    answer, tokens = call_llm_text(
        prompt,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )

    cleaned = answer.strip()

    if "```" in cleaned:
        cleaned = (
            cleaned
            .replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
        )

    try:
        return json.loads(cleaned), tokens
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Réponse LLM non JSON valide : {cleaned}"
        ) from e


def extract_answer_and_tokens(data: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """
    Extrait le contenu texte et les tokens depuis une réponse Groq complète.
    Utile pour agent.py quand on utilise directement call_llm avec tools.
    """

    answer = data["choices"][0]["message"].get("content", "")
    usage = data.get("usage", {})

    tokens = {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }

    return answer, tokens


def verify_relevance(
    *,
    question: str,
    answer: str,
    context: str,
    evaluator_name: str = "système RAG",
) -> tuple[int, str]:
    """
    Évalue automatiquement la pertinence d'une réponse par rapport aux données.

    Retourne :
    - score entre 0 et 100
    - explication courte
    """

    prompt = f"""Tu es un évaluateur de qualité pour un {evaluator_name}.

Évalue la pertinence de la réponse par rapport à la question et aux données fournies.

Question :
{question}

Données utilisées :
{context[:2000]}

Réponse générée :
{answer[:1500]}

Réponds UNIQUEMENT avec un JSON valide :
{{
  "score": <entier entre 0 et 100>,
  "explication": "<1 ou 2 phrases expliquant le score>"
}}

Critères :
- 90-100 : réponse précise, basée sur les données, sans hallucination
- 70-89 : réponse correcte mais incomplète
- 50-69 : réponse partiellement correcte
- 0-49 : réponse incorrecte ou hallucinée
"""

    try:
        data, _ = call_llm_json(
            prompt,
            max_tokens=200,
            temperature=0.1,
            timeout=20,
        )

        score = int(data.get("score", 0))
        explanation = data.get("explication", "")

        score = max(0, min(100, score))

        return score, explanation

    except Exception as e:
        return 0, f"Erreur évaluation : {e}"