import json
import os
from typing import Optional

BOOKS_PATH = "backend/data/book_summaries.json"

def get_summary_by_title(title: str) -> Optional[str]:
    if not os.path.exists(BOOKS_PATH):
        raise FileNotFoundError(f"Book summaries file not found at {BOOKS_PATH}")
    
    with open(BOOKS_PATH, "r", encoding="utf-8") as f:
        books = json.load(f)

    for book in books:
        if book["title"].strip().lower() == title.strip().lower():
            return book["summary"]

    return None

def list_titles() -> list[str]:
    """Lista titlurilor cunoscute din DB-ul local."""
    try:
        # Dacă ai un dict/structură internă (ex. BOOKS_DB) folosește-l aici:
        from tools.book_summary_tool import BOOKS_DB  # dacă există
        return list(BOOKS_DB.keys())
    except Exception:
        # Fallback rapid – completează/ajustează după colecția ta reală
        return [
            "1984",
            "Animal Farm",
            "Brave New World",
            "Harry Potter and the Philosopher's Stone",
            "The Hobbit",
            "The Chronicles of Narnia",
            "Fahrenheit 451",
            "The Catcher in the Rye",
            "The Lord of the Rings",
            "To Kill a Mockingbird",
            "The Great Gatsby",
        ]

def normalize_title(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())
