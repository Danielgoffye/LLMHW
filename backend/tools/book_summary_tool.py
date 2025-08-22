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
