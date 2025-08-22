import os
import re
from dotenv import load_dotenv
from openai import OpenAI

from vector_store.retriever import BookRetriever
from tools.translation_tool import detect_language, translate
from tools.language_filter_tool import is_offensive
from tools.tts_tool import speak
from tools.book_summary_tool import get_summary_by_title, list_titles, normalize_title

# Setup OpenAI
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Initialize retriever
retriever = BookRetriever()

# ---------------- Language heuristics ----------------

def looks_like_romanian(text: str) -> bool:
    t = (text or "").lower()
    # indicii frecvente de RO; extinde-le la nevoie
    ro_hints = [
        "îmi", "imi", "poți", "poti", "te rog", "mulțumesc", "multumesc",
        "carte", "despre", "vreau", "dragoni", "magie",
        "ce este", "spune-mi", "spunemi", "știi", "stii",
        "bună", "buna", "salut"
    ]
    # diacritice românești sau cuvinte cheie
    if any(ch in t for ch in "ăâîșț"):
        return True
    return any(h in t for h in ro_hints)

def enforce_detected_lang(user_input: str, detected: str) -> str:
    """
    Suprascrie detectarea pentru cazurile ambigue.
    Preferăm RO când vedem indicii clare,
    altfel păstrăm limba detectată; dacă e necunoscută, cădem pe 'en'.
    """
    if detected in {"it", "es", "pt", "fr"} and looks_like_romanian(user_input):
        return "ro"
    if detected in {None, "", "unknown"}:
        return "en"
    return detected

# ---------------- Lookup helpers (titles) ----------------

def extract_lookup_candidate(en_text: str) -> str | None:
    """
    Extrage un posibil titlu din formulări tipice (după normalizarea în EN).
    Exemple:
      - what is <title> / what's <title> / who is <title> / who wrote <title>
      - tell me about <title> / do you know anything about <title>
      - scurt numeric: "1984"
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

    # fallback: token scurt numeric (ex: "1984")
    if len(t) <= 10 and any(ch.isdigit() for ch in t):
        return t
    return None

def find_title_in_text(en_text: str, titles: list[str]) -> str | None:
    """
    Caută un titlu care apare textual în întrebare (case-insensitive).
    1) încercare cu word boundaries pe titlul exact
    2) fallback: compară string-ul normalizat (fără spații/punctuație)
    """
    t_low = (en_text or "").lower()

    # 1) match exact cu word boundaries
    for title in titles:
        pat = r"\b" + re.escape(title.lower()) + r"\b"
        if re.search(pat, t_low):
            return title

    # 2) fallback: normalizare
    norm_titles = {normalize_title(tt): tt for tt in titles}
    norm_text = normalize_title(t_low)
    for nk, orig in norm_titles.items():
        if nk and nk in norm_text:
            return orig

    return None

# ---------------- Main chat flow ----------------

def chat_with_llm(user_input: str):
    """
    Flow:
      1) Detectează limba, filtrează limbaj nepotrivit (cu override pentru RO)
      2) Normalizează la EN (pentru parsing)
      3) LOOKUP STRICT de titlu (dacă titlul apare în întrebare)
      4) Dacă nu găsim titlu exact: RAG (prag strâns) pentru recomandare tematică
      5) Fallback clar (fără "ghicit")

    Returnează: (text_afisat, limba_detectata, summary_pentru_TTS_ou_None)
    """

    # 1) Detectăm limba și filtrăm limbaj nepotrivit
    raw_lang = detect_language(user_input)
    detected_lang = enforce_detected_lang(user_input, raw_lang)

    if is_offensive(user_input):
        msg = "Your message contains inappropriate language. Please rephrase politely."
        return (translate(msg, target_lang=detected_lang) if detected_lang != "en" else msg), detected_lang, None

    # 2) Normalizează la EN pentru parsing / RAG
    english_input = user_input if detected_lang == "en" else translate(
        user_input, target_lang="en", source_lang=detected_lang
    )

    # 3) LOOKUP STRICT de titlu: dacă apare în text sau în formă tipică de întrebare
    titles = list_titles()

    # 3.a) avem un candidat extras din formulări tipice?
    candidate = extract_lookup_candidate(english_input)
    exact_title = find_title_in_text(english_input, titles) if candidate is None else None

    # 3.b) dacă avem candidat, încercăm mai întâi acel string, apoi normalizarea
    if candidate:
        exact_title = find_title_in_text(candidate, titles)
        if not exact_title:
            norm_titles = {normalize_title(tt): tt for tt in titles}
            cand_norm = normalize_title(candidate)
            exact_title = norm_titles.get(cand_norm)

    if exact_title:
        full_summary = get_summary_by_title(exact_title)
        if full_summary:
            # răspuns clar: Titlu + Rezumat
            full_text_en = f"{exact_title}\n\n{full_summary}"
            if detected_lang != "en":
                localized_text = translate(full_text_en, target_lang=detected_lang)
                localized_summary = translate(full_summary, target_lang=detected_lang)
                return localized_text, detected_lang, localized_summary
            return full_text_en, detected_lang, full_summary
        # dacă nu avem summary pentru titlul găsit, nu ghicim -> continuăm cu RAG

    # 4) RAG cu prag strâns: recomandare tematică doar când e foarte relevant
    matches = retriever.query(english_input, top_k=1)
    if matches:
        top = matches[0]
        dist = float(top.distance)
        if dist <= 1.2:
            title = top.title
            summary = top.summary

            # LLM – răspuns conversațional; impunem limba de răspuns
            # dacă utilizatorul vrea ro, spunem explicit "Respond in Romanian"
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

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            model_answer = response.choices[0].message.content.strip()

            # Dacă am forțat deja limba prin system prompt, nu mai traducem model_answer.
            # Doar adăugăm rezumatul local (tradus la nevoie).
            full_summary = get_summary_by_title(title)
            if detected_lang != "en" and full_summary:
                localized_summary = translate(full_summary, target_lang=detected_lang)
            else:
                localized_summary = full_summary

            summary_block = ""
            if full_summary:
                if detected_lang != "en":
                    summary_block = f"\n\nIată un rezumat detaliat al *{title}*:\n{localized_summary}"
                else:
                    summary_block = f"\n\nHere's a detailed summary of *{title}*:\n{full_summary}"

            final_out = f"{model_answer}{summary_block}"
            return final_out, detected_lang, localized_summary

    # 5) Fallback clar, fără „ghicit”
    msg = "Sorry, I don't have information about that. Please ask about a specific book title or describe the kind of story you want."
    return (translate(msg, target_lang=detected_lang) if detected_lang != "en" else msg), detected_lang, None


if __name__ == "__main__":
    print("=== Smart Book Recommender (LLM + RAG) ===\n")
    while True:
        user_input = input("Your question (or 'exit'): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        print("\nThinking...\n")

        # resetăm mereu summary la fiecare iterație
        result, lang, summary = chat_with_llm(user_input)
        print(result)

        # întreabă de TTS doar dacă summary este un string non-gol
        if isinstance(summary, str) and summary.strip():
            choice = input("Do you want to hear the recommendation? (y/n): ").strip().lower()
            if choice == "y":
                speak(summary, lang=lang)

        print("\n" + "="*60 + "\n")
