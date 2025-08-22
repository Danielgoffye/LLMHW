import os
from dotenv import load_dotenv
from openai import OpenAI
from vector_store.retriever import BookRetriever
from tools.book_summary_tool import get_summary_by_title

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
    if not is_question_about_books(user_input):
        return "I'm here to help with book recommendations. Please ask something related to stories, books, or themes."

    # 1. Caută cea mai bună potrivire în vector store
    matches = retriever.query(user_input, top_k=1)
    if not matches:
        return "Sorry, I couldn't find any relevant books."

    top_match = matches[0]
    title = top_match.title
    summary = top_match.summary

    # 2. Pregătește promptul pentru LLM
    system_prompt = (
        "You are an intelligent assistant that recommends books to users "
        "based on their interests. You will use the provided context to recommend a book."
    )

    user_prompt = f"""User asked: "{user_input}"

Use this context to answer: "{summary}"

Respond in a friendly and helpful tone. Mention the book title if relevant."""

    # 3. Trimite către LLM (gpt-4o)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )

    main_answer = response.choices[0].message.content.strip()

    # 4. Apelează local tool-ul pentru rezumat complet
    full_summary = get_summary_by_title(title)
    if full_summary:
        final_answer = f"{main_answer}\n\nHere's a detailed summary of *{title}*:\n{full_summary}"
    else:
        final_answer = f"{main_answer}\n\n(No detailed summary found for '{title}')"

    return final_answer

if __name__ == "__main__":
    print("=== Smart Book Recommender (LLM + RAG) ===\n")
    while True:
        user_input = input("Your question (or 'exit'): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        print("\nThinking...\n")
        result = chat_with_llm(user_input)
        print(result)
        print("\n" + "="*60 + "\n")
