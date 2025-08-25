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
        from backend.tools.book_summary_tool import BOOKS_DB  # dacă există
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

# ... păstrează importurile tale existente
import re
from functools import lru_cache
from difflib import get_close_matches

# asigură-te că ai deja normalize_title, list_titles, get_summary_by_title în acest fișier

@lru_cache()
def title_alias_map() -> dict[str, str]:
    """
    alias (lower, normalizat) -> titlu canonic exact (cheia din JSON).
    Completează cu alias-uri utile. Am adăugat mockingbird & co.
    """
    aliases = {
        # Harry Potter
        "harry potter": "Harry Potter and the Philosopher's Stone",
        "harry potter si piatra filosofala": "Harry Potter and the Philosopher's Stone",
        "harry potter și piatra filosofală": "Harry Potter and the Philosopher's Stone",

        # The Hobbit
        "the hobbit": "The Hobbit",
        "hobbitul": "The Hobbit",

        # 1984
        "1984": "1984",

        # The Great Gatsby
        "the great gatsby": "The Great Gatsby",
        "marele gatsby": "The Great Gatsby",

        # Narnia
        "narnia": "The Chronicles of Narnia",
        "the chronicles of narnia": "The Chronicles of Narnia",

        # To Kill a Mockingbird (variantes + typos)
        "to kill a mockingbird": "To Kill a Mockingbird",
        "mockingbird": "To Kill a Mockingbird",
        "mocking bird": "To Kill a Mockingbird",
        "mockin bird": "To Kill a Mockingbird",     # typo frecvent
        "mockinbird": "To Kill a Mockingbird",
        "sa ucizi o pasare cantatoare": "To Kill a Mockingbird",   # RO
        "să ucizi o pasăre cântătoare": "To Kill a Mockingbird",
    }
    # normalizează cheile (alias-urile)
    return {normalize_title(k): v for k, v in aliases.items()}

def _resolve_in_single_text(raw_text: str) -> str | None:
    """
    1) word-boundary pe titluri canonice
    2) alias-uri normalizate
    3) normalized title substrings
    4) fuzzy pe alias-uri + titluri (pentru typos: 'mockin bird')
    """
    if not raw_text:
        return None

    t_low = raw_text.lower()

    # 1) direct titles (word-boundary)
    for title in list_titles():
        if re.search(r"\b" + re.escape(title.lower()) + r"\b", t_low):
            return title

    # 2) alias normalized
    norm_text = normalize_title(t_low)
    amap = title_alias_map()
    for ak, canonical in amap.items():
        if ak and ak in norm_text:
            return canonical

    # 3) normalized title substrings
    norm_titles = {normalize_title(tt): tt for tt in list_titles()}
    for nk, orig in norm_titles.items():
        if nk and nk in norm_text:
            return orig

    # 4) FUZZY fallback (difflib) pe alias-uri și titluri normalizate
    #    încercăm întâi alias-urile (sunt mai scurte/robuste)
    alias_keys = list(amap.keys())
    # get_close_matches pe întreg textul normalizat
    close = get_close_matches(norm_text, alias_keys, n=1, cutoff=0.82)
    if close:
        return amap[close[0]]

    # altă tactică fuzzy: spargem textul în tokeni și căutăm pe n-gramuri scurte (2-4 cuvinte)
    tokens = norm_text.split()
    candidates = set()
    for size in (2, 3, 4):
        for i in range(0, max(0, len(tokens) - size + 1)):
            cand = " ".join(tokens[i:i+size])
            candidates.add(cand)

    if candidates:
        # fuzzy pe alias-uri
        close = get_close_matches(" ".join(tokens), alias_keys, n=1, cutoff=0.75)
        if close:
            return amap[close[0]]
        # fuzzy pe n-gramuri
        for cand in candidates:
            c1 = get_close_matches(cand, alias_keys, n=1, cutoff=0.8)
            if c1:
                return amap[c1[0]]

    # fuzzy pe titluri canonice (normalizate)
    canon_keys = list(norm_titles.keys())
    close2 = get_close_matches(norm_text, canon_keys, n=1, cutoff=0.82)
    if close2:
        return norm_titles[close2[0]]

    return None

def resolve_title_from_any_text(*texts: str) -> str | None:
    """Încearcă pe rând în toate textele (original + tradus EN, etc.)."""
    for tx in texts:
        hit = _resolve_in_single_text(tx)
        if hit:
            return hit
    return None
