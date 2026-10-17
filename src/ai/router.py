"""
router.py — Routage automatique des questions utilisateur
"""

from llm_client import call_llm_json
from src.ai.prompts import ROUTER_PROMPT
from src.ai.nlp_utils import extract_keywords, detect_time_expression

KEYWORDS_REALTIME = [
    "maintenant",
    "actuellement",
    "en ce moment",
    "disponible",
    "disponibles",
    "combien de vélo",
    "combien de velos",
    "station",
]

KEYWORDS_ANOMALY = [
    "anomalie",
    "anomalies",
    "vide",
    "vides",
    "pleine",
    "pleines",
    "critique",
    "critiques",
    "déséquilibre",
    "desequilibre",
]

KEYWORDS_HISTORICAL = [
    "historique",
    "souvent",
    "tendance",
    "tendances",
    "évolution",
    "evolution",
    "hier",
    "semaine",
    "week-end",
    "weekend",
    "matin",
    "soir",
    "midi",
    "comparaison",
    "compare",
]

# Marqueurs d'habitude : priment sur les mots d'anomalie (« souvent vide » = historique)
KEYWORDS_HABIT = [
    "souvent",
    "d'habitude",
    "habituellement",
    "généralement",
    "generalement",
    "en général",
    "en general",
    "régulièrement",
    "regulierement",
    "toujours",
    "jamais",
    "historique",
    "tendance",
    "tendances",
]

# Marqueurs d'instant présent : la question porte sur maintenant, pas sur l'historique
KEYWORDS_NOW = [
    "maintenant",
    "actuellement",
    "en ce moment",
    "aujourd'hui",
]

# Question sur le futur proche : le modèle de prévision répond
KEYWORDS_FORECAST = [
    "dans une heure",
    "dans 1h",
    "dans 1 h",
    "dans l'heure",
    "prévision",
    "prevision",
    "prévoir",
    "prédire",
    "predire",
    "sera vide",
    "sera-t-elle vide",
    "va être vide",
    "va se vider",
    "bientôt vide",
    "risque",
]

KEYWORDS_REPORT = [
    "rapport",
    "bilan",
    "synthèse",
    "synthese",
    "résumé",
    "resume",
    "pdf",
]


def _matched(q: str, keywords: list[str]) -> list[str]:
    return [kw for kw in keywords if kw in q]


def _route(intent: str, rule: str, matched: list[str], reason: str) -> dict:
    """
    Résultat du routage par règles. Pas de « confiance » chiffrée : une règle s'applique ou non.
    On renvoie donc la règle et les mots qui l'ont déclenchée, ce qui est vérifiable.
    """
    return {"intent": intent, "rule": rule, "matched": matched, "confidence": None, "reason": reason}


def classify_question_rule_based(question: str) -> dict:
    """
    Classification sans LLM, par règles testées dans l'ordre (la première qui s'applique gagne).
    Rapide, gratuite et déterministe.
    """
    q = question.lower()
    keywords = extract_keywords(question)
    time_expressions = detect_time_expression(question)
    now = _matched(q, KEYWORDS_NOW)

    if found := _matched(q, KEYWORDS_REPORT):
        return _route("report", "rapport", found, "La question demande un rapport ou une synthèse.")

    if found := _matched(q, KEYWORDS_FORECAST):
        return _route("forecast", "prévision", found, "La question porte sur l'état de la station dans l'heure à venir.")

    if not now and (found := _matched(q, KEYWORDS_HABIT)):
        return _route("historical", "habitude", found, "La question porte sur une habitude ou une tendance.")

    if found := _matched(q, KEYWORDS_ANOMALY):
        return _route("anomaly", "anomalie", found, "La question mentionne des anomalies ou des stations critiques.")

    if not now and (found := _matched(q, KEYWORDS_HISTORICAL)):
        return _route(
            "historical", "période de temps", found,
            f"La question vise une période ({', '.join(time_expressions) or ', '.join(found)}).",
        )

    if found := _matched(q, KEYWORDS_REALTIME) or now:
        return _route("realtime", "temps réel", found, "La question porte sur l'état actuel du réseau.")

    return _route(
        "general", "aucune", [],
        f"Aucune règle ne reconnaît la question (mots-clés : {', '.join(keywords[:5]) or 'aucun'}).",
    )


def classify_question(question: str, use_llm: bool = False) -> dict:
    """
    Route la question vers une intention.

    Par défaut, utilise les règles pour éviter un appel LLM inutile.
    Si use_llm=True, demande une classification au modèle.
    """
    rule_result = classify_question_rule_based(question)

    if not use_llm:
        return rule_result

    try:
        prompt = f"""
Question utilisateur :
{question}
"""

        data, _ = call_llm_json(
            prompt,
            system_prompt=ROUTER_PROMPT,
            max_tokens=150,
            temperature=0.1,
            timeout=15,
        )

        intent = data.get("intent", rule_result["intent"])
        confidence = float(data.get("confidence") or 0.0)
        reason = data.get("reason", rule_result["reason"])

        valid_intents = {
            "realtime",
            "anomaly",
            "historical",
            "semantic",
            "report",
            "forecast",
            "general",
        }

        if intent not in valid_intents:
            return rule_result

        return {
            "intent": intent,
            "rule": "routeur LLM",
            "matched": [],
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": reason,
        }

    except Exception:
        return rule_result