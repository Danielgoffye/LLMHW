# backend/LLMHW.py
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

from openai import OpenAI

# Tools / store
from backend.vector_store.retriever import BookRetriever
from backend.tools.translation_tool import detect_language, translate
from backend.tools.language_filter_tool import is_offensive
from backend.tools.tts_tool import speak
from backend.tools.book_summary_tool import (
    get_summary_by_title,
    list_titles,
    normalize_title,
    resolve_title_from_any_text,
)
from backend.tools.stt_tool import capture_and_transcribe_vad


# ---------------- OpenAI client (lazy) ----------------

def _get_client() -> OpenAI:
    """
    Lazy-init pentru clientul OpenAI. NU creăm client la import, doar când chiar avem nevoie.
    Curățăm potențialele newline/ghilimele din cheie.
    """
    raw = os.getenv("OPENAI_API_KEY", "")
    api_key = raw.strip().strip('"').strip("'")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Set it in .env or environment.")
    return OpenAI(api_key=api_key)


# ---------------- Vector retriever ----------------

# Chroma nu depinde de cheia OpenAI, deci poate fi creat la import fără risc
retriever = BookRetriever()


# ---------------- Heuristici limbă ----------------

def looks_like_romanian(text: str) -> bool:
    t = (text or "").lower()
    ro_hints = [
        "îmi", "imi", "poți", "poti", "te rog", "mulțumesc", "multumesc",
        "carte", "despre", "vreau", "dragoni", "magie",
        "ce este", "spune-mi", "spunemi", "știi", "stii",
        "bună", "buna", "salut"
    ]
    if any(ch in t for ch in "ăâîșț"):
        return True
    return any(h in t for h in ro_hints)


def enforce_detected_lang(user_input: str, detected: Optional[str]) -> str:
    """
    Override pentru cazuri ambigue: dacă pare română, forțăm 'ro'.
    Dacă detectarea e necunoscută, cădem pe 'en'.
    """
    if detected in {"it", "es", "pt", "fr"} and looks_like_romanian(user_input):
        return "ro"
    if detected in {None, "", "unknown"}:
        return "en"
    return detected


# ---------------- Extindere tematică pentru RAG ----------------

THEME_SYNONYMS = {
    "friendship": ["friends", "bond", "companionship", "ally"],
    "magic": ["wizardry", "sorcery", "magical", "fantasy"],
    "love": ["romance", "affection"],
    "war": ["battle", "conflict"],
    "freedom": ["liberty", "escape"],
    "society": ["social", "community", "class"],
    "adventure": ["quest", "journey"],
}

RO_TO_EN_SEED = {
    "prieteni": ["friendship", "friends"],
    "prietenie": ["friendship"],
    "magie": ["magic", "fantasy"],
    "iubire": ["love", "romance"],
    "dragoste": ["love", "romance"],
    "razboi": ["war", "battle"],
    "război": ["war", "battle"],
    "libertate": ["freedom"],
    "societate": ["society", "social"],
    "aventura": ["adventure"],
    "aventură": ["adventure"],
}

def expand_thematic_query(en_text: str) -> list[str]:
    """
    Dacă interogarea pare tematică (nu e titlu clar), întoarce 2-3 variante extinse.
    1) plecăm de la en_text (tokenizat)
    2) adăugăm sinonime EN
    3) includem mapările RO -> EN dacă au rămas termeni RO după traducere
    """
    base = (en_text or "").lower().strip()
    tokens = re.findall(r"[a-z0-9]+", base)
    seeds = set(tokens)

    # mapăm termeni RO -> EN
    for w in tokens:
        if w in RO_TO_EN_SEED:
            for en in RO_TO_EN_SEED[w]:
                seeds.add(en)

    # adăugăm sinonime EN
    for w in list(seeds):
        if w in THEME_SYNONYMS:
            seeds.update(THEME_SYNONYMS[w])

    # construim câteva variante scurte
    variants: list[str] = []

    # varianta „all seeds”
    if seeds:
        variants.append(" ".join(sorted(seeds)))

    # subset cu termeni importanți
    important = [w for w in seeds if w in THEME_SYNONYMS or w in {
        "magic", "friendship", "war", "love", "freedom", "society", "adventure"
    }]
    if important:
        variants.append(" ".join(sorted(important)))

    # originalul (dacă e non-gol)
    if base:
        variants.append(base)

    # unic & max 3
    out: list[str] = []
    for v in variants:
        v = v.strip()
        if v and v not in out:
            out.append(v)
    return out[:3]


# ---------------- Lookup titlu în text ----------------

def extract_lookup_candidate(en_text: str) -> Optional[str]:
    """
    Extrage un posibil titlu din formulări tipice (după normalizarea în EN).
    Ex: what is <title> / what's <title> / who is <title> / who wrote <title>
        tell me about <title> / do you know anything about <title>
    Scurt numeric: „1984”
    """
    t = (en_text or "").strip().lower()
    pats = [
        r"^(what\s+is|what's)\s+(.+?)\??$",
        r"^who\s+is\s+(.+?)\??$",
        r"^(who\s+wrote)\s+(.+?)\??$",
        r"^(tell\s+me\s+about)\s+(.+?)\??$",
        r"^(do\s+you\s+know\s+anything\s+about)\s+(.+?)\??$",
        r"^(what\s+can\s+you\s+tell\s+me\s+about)\s+(.+?)\??$",
        r"^what\s+is\s+(.+?)\??$",
    ]
    for p in pats:
        m = re.match(p, t)
        if m:
            return m.groups()[-1].strip()

    if len(t) <= 10 and any(ch.isdigit() for ch in t):
        return t
    return None


def find_title_in_text(en_text: str, titles: list[str]) -> Optional[str]:
    """
    Caută un titlu care apare textual în întrebare (case-insensitive).
    1) încercare cu word boundaries pe titlul exact
    2) fallback: comparăm string-ul normalizat (fără spații/punctuație)
    """
    t_low = (en_text or "").lower()

    # match exact cu word boundaries
    for title in titles:
        pat = r"\b" + re.escape(title.lower()) + r"\b"
        if re.search(pat, t_low):
            return title

    # fallback: normalizare
    norm_titles = {normalize_title(tt): tt for tt in titles}
    norm_text = normalize_title(t_low)
    for nk, orig in norm_titles.items():
        if nk and nk in norm_text:
            return orig

    return None


def is_question_about_books(text: str) -> bool:
    """
    Euristică de bază ca să nu răspundem la întrebări care nu au legătură cu cărți/povești.
    """
    keywords = [
        # EN
        "book", "novel", "read", "story", "recommend", "suggest",
        "magic", "war", "friendship", "freedom", "society", "fantasy",
        "adventure", "love", "romance", "dragon", "dragons",
        # RO
        "carte", "roman", "poveste", "recomanda", "recomandă", "sugereaza", "sugerează",
        "magie", "razboi", "război", "prietenie", "prieteni", "libertate", "societate",
        "aventura", "aventură", "iubire", "dragoni",
        # îmbunătățiri de intenție
        "o carte", "o poveste", "citit", "vreau", "caut"
    ]
    t = (text or "").lower()
    return any(kw in t for kw in keywords)


# ---------------- Main chat flow ----------------

def chat_with_llm(user_input: str) -> Tuple[str, str, Optional[str]]:
    """
    Flow:
      1) Detectează limba, filtrează limbaj nepotrivit (cu override pentru RO)
      2) Normalizează la EN (pentru parsing & RAG)
      3) LOOKUP STRICT de titlu (dacă titlul apare în întrebare)
      4) Dacă nu găsim titlu exact: RAG tematic (prag strâns + sinonime)
      5) Fallback clar (fără "ghicit")

    Returnează: (text_de_afisat, limba_detectata, summary_pentru_TTS_ou_None)
    """
    # 1) Limba + ofensiv
    raw_lang = detect_language(user_input)
    detected_lang = enforce_detected_lang(user_input, raw_lang)

    if is_offensive(user_input):
        msg = "Your message contains inappropriate language. Please rephrase politely."
        return (translate(msg, target_lang=detected_lang) if detected_lang != "en" else msg), detected_lang, None

    # 2) Normalizare la EN (pentru lookup/RAG)
    english_input = user_input if detected_lang == "en" else translate(
        user_input, target_lang="en", source_lang=detected_lang
    )

    # 3) LOOKUP STRICT (folosește utilitarul din book_summary_tool)
    exact_title = resolve_title_from_any_text(user_input, english_input)
    if exact_title:
        full_summary = get_summary_by_title(exact_title)
        if full_summary:
            resp_en = f"{exact_title}\n\n{full_summary}"
            if detected_lang != "en":
                localized_resp = translate(resp_en, target_lang=detected_lang)
                localized_summary = translate(full_summary, target_lang=detected_lang)
                return localized_resp, detected_lang, localized_summary
            return resp_en, detected_lang, full_summary
        # dacă nu avem summary, continuăm cu RAG (nu ghicim)

    # 4) RAG tematic (extindem ușor interogarea)
    expanded = expand_thematic_query(english_input)
    candidates: list[tuple[float, str, str]] = []  # (distance, title, summary)

    for q in expanded:
        ms = retriever.query(q, top_k=3)
        for m in ms:
            candidates.append((float(m.distance), m.title, m.summary))

    candidates.sort(key=lambda x: x[0])

    if candidates:
        best_dist, title, summary = candidates[0]
        # prag puțin relaxat pentru teme (ajustează dacă vrei mai strict)
        if best_dist <= 1.6:
            # LLM – răspuns conversațional în limba utilizatorului
            lang_directive = {
                "ro": "Respond in Romanian.",
                "en": "Respond in English."
            }.get(detected_lang, f"Respond in {detected_lang}.")

            system_prompt = (
                "You are an intelligent assistant that recommends books based on user interests. "
                "Use the provided context to give a helpful and natural recommendation. "
                + lang_directive
            )
            user_prompt = f'''User asked: "{english_input}"

Context: "{summary}"

Respond with a friendly book suggestion. Mention the book title if relevant.'''

            client = _get_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            model_answer = (response.choices[0].message.content or "").strip()

            # Rezumat complet din sursa locală (pt. afișare + TTS)
            full_summary = get_summary_by_title(title)
            localized_summary = (
                translate(full_summary, target_lang=detected_lang)
                if (detected_lang != "en" and full_summary)
                else full_summary
            )

            summary_block = ""
            if full_summary:
                if detected_lang != "en":
                    summary_block = f"\n\nIată un rezumat detaliat al *{title}*:\n{localized_summary}"
                else:
                    summary_block = f"\n\nHere's a detailed summary of *{title}*:\n{full_summary}"

            final_out = f"{model_answer}{summary_block}"
            return final_out, detected_lang, localized_summary

    # Dacă întrebarea NU pare despre cărți/povești, ghidăm utilizatorul
    if not is_question_about_books(english_input):
        msg = "Please ask something related to books or stories."
        return (translate(msg, target_lang=detected_lang) if detected_lang != "en" else msg), detected_lang, None

    # 5) Fallback clar, fără „ghicit”
    msg = "Sorry, I don't have information about that. Please ask about a specific book title or describe the kind of story you want."
    return (translate(msg, target_lang=detected_lang) if detected_lang != "en" else msg), detected_lang, None


# ---------------- CLI util (opțional) ----------------

if __name__ == "__main__":
    print("=== Smart Book Recommender (LLM + RAG) ===")
    print("Type 'voice' to dictate your question (auto-stops on silence).")
    print()

    while True:
        user_input = input("Your question (or 'exit' | 'voice'): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        if user_input.lower() == "voice":
            try:
                text = capture_and_transcribe_vad(language_hint=None)
            except Exception as e:
                print(f"[Voice] Error: {e}")
                print("\n" + "="*60 + "\n")
                continue

            if not text or text == "YOU_SAID_NOTHING":
                print("[Voice] You said nothing. Returning to menu.")
                print("\n" + "="*60 + "\n")
                continue

            print(f"[Voice] You said: {text}")
            user_input = text

        print("\nThinking...\n")
        result, lang, summary = chat_with_llm(user_input)
        print(result)

        if isinstance(summary, str) and summary.strip():
            choice = input("Do you want to hear the recommendation? (y/n): ").strip().lower()
            if choice == "y":
                speak(summary, lang=lang)

        print("\n" + "="*60 + "\n")
