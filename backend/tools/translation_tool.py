# backend/tools/translation_tool.py
from __future__ import annotations

import os
from typing import Optional

# ...existing code...

from openai import OpenAI
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def _get_client() -> OpenAI:
    raw = os.getenv("OPENAI_API_KEY", "")
    api_key = raw.strip().strip('"').strip("'")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Set it in .env or environment.")
    return OpenAI(api_key=api_key)

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

def translate(text: str, target_lang: str = "en", source_lang: Optional[str] = None) -> str:
    text = (text or "")
    if not text.strip():
        return text

    if source_lang is None:
        source_lang = detect_language(text)

    # dacă necunoscut, nu riscăm traducere aberantă
    if source_lang == "unknown":
        return text

    # dacă e deja în limba țintă, return direct
    if source_lang.lower().startswith(target_lang.lower()):
        return text

    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}. "
        f"Keep the meaning and tone as close as possible:\n\n{text}"
    )

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text
