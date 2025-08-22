import os
from dotenv import load_dotenv
from openai import OpenAI
from vector_store.retriever import BookRetriever
from tools.book_summary_tool import get_summary_by_title
from tools.translation_tool import detect_language, translate
from tools.language_filter_tool import is_offensive
from tools.tts_tool import speak


# Setup OpenAI
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Initialize retriever
retriever = BookRetriever()
def is_question_about_books(text: str) -> bool:
    """Heuristic: considerăm că întrebarea e validă dacă are termeni relevanți."""
    keywords = [
        "book", "novel", "read", "story", "recommend", "suggest",
        "magic", "war", "friendship", "freedom", "society", "fantasy"
    ]
    text = text.lower()
    return any(kw in text for kw in keywords)


def chat_with_llm(user_input: str):
    if is_offensive(user_input):
        msg = "Your message contains inappropriate language. Please rephrase politely."
        lang = detect_language(user_input)
        return translate(msg, target_lang=lang), lang, None
    
    # 0. Detectăm limba inițială
    detected_lang = detect_language(user_input)

    # 1. Verificare semantică simplă (doar pentru limbile cunoscute)
    if detected_lang == "en":
        is_valid = is_question_about_books(user_input)
    else:
        translated_input = translate(user_input, target_lang="en", source_lang=detected_lang)
        is_valid = is_question_about_books(translated_input)

    if not is_valid:
        msg = "Please ask something related to books or stories."
        return translate(msg, target_lang=detected_lang), detected_lang, None

    # 2. Traducem în engleză (dacă e cazul)
    english_input = user_input if detected_lang == "en" else translate(user_input, target_lang="en", source_lang=detected_lang)

    # 3. RAG – căutăm potrivirea semantică
    matches = retriever.query(english_input, top_k=1)
    if not matches:
        msg = "Sorry, I couldn't find any relevant books in my library."
        return translate(msg, target_lang=detected_lang), detected_lang, None

    top_match = matches[0]
    if top_match.distance > 1.5:
        msg = "Sorry, I couldn't find any book that matches your request."
        return translate(msg, target_lang=detected_lang), detected_lang, None

    title = top_match.title
    summary = top_match.summary

    # 4. LLM – generăm răspuns conversațional în engleză
    system_prompt = (
        "You are an intelligent assistant that recommends books based on user interests. "
        "Use the provided context to give a helpful and natural recommendation."
    )

    user_prompt = f"""User asked: "{english_input}"

Context: "{summary}"

Respond with a friendly book suggestion. Mention the book title if relevant.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )

    main_answer_en = response.choices[0].message.content.strip()

    # 5. Tool: rezumat complet (din JSON)
    full_summary = get_summary_by_title(title)

    # localizează rezumatul pentru TTS
    if detected_lang != "en" and full_summary:
        localized_summary = translate(full_summary, target_lang=detected_lang)
    else:
        localized_summary = full_summary

    summary_text = f"\n\nHere's a detailed summary of *{title}*:\n{full_summary}" if full_summary else ""
    full_response_en = f"{main_answer_en}{summary_text}"
    
    # 6. Traducem înapoi în limba utilizatorului (dacă nu e engleză)
    if detected_lang != "en":
        return translate(full_response_en, target_lang=detected_lang), detected_lang, localized_summary
    return full_response_en, detected_lang, localized_summary


if __name__ == "__main__":
    print("=== Smart Book Recommender ===\n")
    while True:
        user_input = input("Your question (or 'exit'): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        print("\nThinking...\n")
        result, lang, summary = chat_with_llm(user_input)
        print(result)
        choice = input("Do you want to hear the recommendation? (y/n): ").strip().lower()
        if choice == "y" and summary:
            speak(summary, lang=lang)
        print("\n" + "="*60 + "\n")
