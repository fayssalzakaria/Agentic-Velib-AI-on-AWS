import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SOURCE_API = "API Velib' (temps reel)"
SOURCE_LAKE = "Lakehouse S3 (dernier snapshot)"
PARIS_TIMEZONE = "Europe/Paris"