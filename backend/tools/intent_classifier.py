import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Clase de intentie acceptate:
# - lookup: utilizatorul cere informatii despre o carte anume
# - recommendation: vrea o sugestie pe baza de tematici/interese
# - nonsense: nu are legatura cu cartile

def classify_intent(user_input: str) -> str:
    prompt = (
        "Classify the intent of the following message into one of the categories: "
        "'lookup', 'recommendation', or 'nonsense'.\n"
        "Respond ONLY with the exact category name.\n\n"
        f"Message: {user_input}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        intent = response.choices[0].message.content.strip().lower()
        if intent in {"lookup", "recommendation", "nonsense"}:
            return intent
        return "nonsense"  # fallback safe
    except Exception as e:
        print(f"[Intent Classification Error] {e}")
        return "nonsense"