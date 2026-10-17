"""
llm_client.py — Client LLM centralisé pour Groq

Objectif :
- Éviter de répéter requests.post(...) dans agent.py, rag.py, chatbot.py, report_generator.py
- Centraliser le modèle, l'URL, les erreurs, le JSON parsing et le calcul des tokens
"""

import json
import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
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


# Tarifs Groq en $ par million de tokens (entrée, sortie), relevés sur /models.
# Sur l'offre gratuite rien n'est facturé : c'est le coût équivalent en offre payante.
PRICES_PER_MILLION = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.80, 4.00),
}


@dataclass
class UsageReport:
    """Consommation cumulée des appels LLM faits pendant un bloc `track_usage()`."""
    calls: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0
    cost_usd: float = 0.0
    models: set = field(default_factory=set)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "llm_calls": self.calls,
            "llm_errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_seconds": round(self.llm_seconds, 2),
            "cost_usd": round(self.cost_usd, 6),
        }


# Pile des mesures actives : un appel compte dans toutes (ex. l'assistant qui délègue à l'agent)
_active_reports: ContextVar[tuple] = ContextVar("llm_usage", default=())


@contextmanager
def track_usage():
    """Mesure tous les appels LLM faits dans le bloc (y compris HyDE, routeur, juge…)."""
    report = UsageReport()
    token = _active_reports.set(_active_reports.get() + (report,))
    try:
        yield report
    finally:
        _active_reports.reset(token)


def _record_usage(model: str, usage: dict | None, seconds: float, error: bool = False) -> None:
    usage = usage or {}
    prompt, completion = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    price_in, price_out = PRICES_PER_MILLION.get(model, (0.0, 0.0))
    for report in _active_reports.get():
        report.calls += 1
        report.errors += int(error)
        report.prompt_tokens += prompt
        report.completion_tokens += completion
        report.llm_seconds += seconds
        report.cost_usd += (prompt * price_in + completion * price_out) / 1_000_000
        report.models.add(model)


MAX_RATE_LIMIT_RETRIES = 2


def _retry_delay(response: requests.Response) -> float:
    try:
        return min(float(response.headers.get("retry-after", 5)), 20.0)
    except (TypeError, ValueError):
        return 5.0


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

    started = time.perf_counter()
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        response = requests.post(
            GROQ_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json=payload,
            timeout=timeout,
        )
        # Limite de débit de l'offre gratuite : on attend le délai indiqué par Groq et on réessaie
        if response.status_code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
            break
        time.sleep(_retry_delay(response))

    if not response.ok:
        _record_usage(model, None, time.perf_counter() - started, error=True)
    _raise_for_groq_error(response)
    data = response.json()
    _record_usage(model, data.get("usage"), time.perf_counter() - started)

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
