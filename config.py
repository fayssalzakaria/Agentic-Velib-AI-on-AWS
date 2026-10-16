import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODE_REALTIME = "⏱️ Temps réel (snapshot)"
MODE_HISTORY = "📚 Historique (lakehouse S3)"
PARIS_TIMEZONE = "Europe/Paris"