# backend/tools/language_filter_tool.py
from __future__ import annotations
import os
import re
from typing import Iterable
from openai import OpenAI

# --- Helpers ---

def _get_client() -> OpenAI:
    key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing. Set it in .env or environment.")
    return OpenAI(api_key=key)

# set minim, extensibil ușor; păstrăm lower-case
_BAD_WORDS_RO: tuple[str, ...] = (
    "prost", "proasta", "proastă", "tâmpit", "idiot", "idiota", "bou",
    "fraier", "fraiero", "handicapat", "handicapată", "imbecil", "imbecila",
    "cretin", "cretina", "panarama",
)

_BAD_WORDS_EN: tuple[str, ...] = (
    "stupid", "idiot", "moron", "dumb", "retard", "retarded", "jerk",
    "asshole", "bastard", "loser", "dickhead",
)

# le combinăm într-o singură expresie cu word boundaries
_BAD_PATTERNS: list[re.Pattern] = [
    re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in (*_BAD_WORDS_RO, *_BAD_WORDS_EN)
]

def _contains_blacklist(t: str) -> bool:
    t = (t or "").lower()
    for pat in _BAD_PATTERNS:
        if pat.search(t):
            return True
    return False

# --- Public API ---

def is_offensive(text: str) -> bool:
    """
    Returnează True dacă mesajul e ofensator.
    1) Heuristic lexical (rapid, fără API) -> dacă match, întoarce True
    2) Moderation API (fallback/extra) -> dacă 'flagged', întoarce True
    În caz de eroare, fail-open (False) dar loghează.
    """
    t = (text or "").strip()
    if not t:
        return False

    # 1) blacklist rapid
    if _contains_blacklist(t):
        return True

    # 2) Moderation API (dacă cheia e validă)
    try:
        client = _get_client()
        resp = client.moderations.create(
            model="omni-moderation-latest",
            input=t,
        )
        result = resp.results[0]
        flagged = bool(getattr(result, "flagged", False))
        return flagged
    except Exception as e:
        print(f"[Offensive Filter Error] {e}")
        return False
