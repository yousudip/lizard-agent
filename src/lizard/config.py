"""Settings, loaded from .env at the project root."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
CHROME_CDP_URL = os.getenv("CHROME_CDP_URL") or None

# LLM baseline for the race (scripts/race.py)
OPENROUTER_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")


def api_key() -> str:
    """The Jev API key, or a clear explanation of where to put one."""
    key = (os.getenv("TYPESAFE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            f"TYPESAFE_API_KEY is not set.\n"
            f"Add it to {ROOT / '.env'} — the line is already there, "
            f"just paste the key after the '=' sign."
        )
    return key
