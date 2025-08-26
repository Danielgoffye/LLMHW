# backend/tools/language_filter_tool.py
import os
from typing import Optional
from openai import OpenAI

__all__ = ["is_offensive"]

# Config: dacă vrei fallback la LLM când Moderation e indisponibil
USE_LLM_FALLBACK = True
LLM_MODEL = "gpt-4o-mini"              # ieftin și suficient pentru yes/no
MODERATION_MODEL = "omni-moderation-latest"

_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    """Lazy-init OpenAI client; nu îl creăm la import."""
    global _client
    if _client is None:
        raw = os.getenv("OPENAI_API_KEY", "")
        api_key = raw.strip().strip('"').strip("'")  # protecție newline/ghilimele
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing. Set it in .env or environment.")
        _client = OpenAI(api_key=api_key)
    return _client

def _fallback_llm_check(text: str) -> bool:
    """
    Fallback prin LLM (chat) în caz că Moderation e indisponibil.
    Returnează True dacă LLM consideră mesajul ofensator (yes/no).
    """
    try:
        client = _get_client()
        prompt = (
            "Classify the message as offensive or not. "
            "Offensive means rude, harassing, hate, sexual content, threats, self-harm encouragement, etc. "
            "Reply with exactly one word: 'yes' or 'no'.\n\n"
            f"Message: {text}"
        )
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        return ans.startswith("yes")
    except Exception as e:
        # dacă și fallback-ul eșuează, fail-open
        print(f"[Offensive Fallback Error] {e}")
        return False

def is_offensive(text: str) -> bool:
    """
    Primary: OpenAI Moderation (fără prompt; model specializat).
    Fallback opțional: LLM yes/no (prompted).
    Fail-open: dacă ambele pică, returnăm False ca să nu dăm 500.
    """
    t = (text or "").strip()
    if not t:
        return False

    # 1) Moderation
    try:
        client = _get_client()
        resp = client.moderations.create(model=MODERATION_MODEL, input=t)
        return bool(resp.results[0].flagged)
    except Exception as e:
        print(f"[Offensive Moderation Error] {e}")

    # 2) LLM fallback (opțional)
    if USE_LLM_FALLBACK:
        return _fallback_llm_check(t)

    # 3) Fail-open
    return False
