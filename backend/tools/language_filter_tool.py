import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def is_offensive(text: str) -> bool:
    if not text.strip():
        return False

    prompt = (
        "Determine if the following message contains any offensive, rude, threatening, "
        "sexually explicit, or inappropriate language. "
        "Return only 'yes' or 'no'.\n\n"
        f"Message: {text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().lower()
        return result.startswith("yes")
    except Exception as e:
        print(f"[Offensive Filter Error] {e}")
        return False  # fallback: nu blocăm dacă apare eroare
