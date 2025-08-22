import os
from typing import Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from langdetect import detect, DetectorFactory

# pentru consistență în detectare
DetectorFactory.seed = 0

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "unknown"


def translate(text: str, target_lang: str = "en", source_lang: Optional[str] = None) -> str:
    if not text.strip():
        return ""

    if source_lang is None:
        source_lang = detect_language(text)

    if source_lang == target_lang:
        return text  # nu traduce dacă e deja în limba țintă

    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}. "
        f"Keep the meaning and tone as close as possible:\n\n{text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        translated = response.choices[0].message.content.strip()
        return translated
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text  # fallback: returnează textul original
