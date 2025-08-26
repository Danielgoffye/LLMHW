# backend/vector_store/retriever.py
from __future__ import annotations

import os
from typing import List, Optional

# .env este încărcat o singură dată în main.py

from openai import OpenAI
import chromadb
from chromadb.config import Settings

# Dacă tu ai deja un dataclass BookMatch, păstrează-l.
from dataclasses import dataclass

@dataclass
class BookMatch:
    title: str
    summary: str
    distance: float

def _get_client() -> OpenAI:
    """Lazy-init OpenAI client (NU la import)."""
    raw = os.getenv("OPENAI_API_KEY", "")
    api_key = raw.strip().strip('"').strip("'")  # <- curățăm newline/spații/ghilimele
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY missing. Set it in your project .env or as environment variable."
        )
    return OpenAI(api_key=api_key)


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Encapsulează cererea de embeddings (text-embedding-3-small)."""
    client = _get_client()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    # OpenAI returnează embeddings în ordinea input-ului
    return [d.embedding for d in resp.data]

class BookRetriever:
    def __init__(
        self,
        persist_dir: str = "backend/vector_store/chroma_db",
        collection_name: str = "books",
    ) -> None:
        # NU atinge OPENAI aici; doar Chroma
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(allow_reset=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def query(self, text: str, top_k: int = 1) -> List[BookMatch]:
        text = (text or "").strip()
        if not text:
            return []

        # Embedding doar acum (cheia trebuie să existe DOAR aici)
        query_emb = _embed_texts([text])[0]

        res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=max(1, int(top_k)),
            include=["metadatas", "distances", "documents"],  # documents dacă ții summary acolo
        )

        matches: List[BookMatch] = []

        # Chroma returnează liste imbricate
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]

        for i in range(len(ids)):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            title = meta.get("title") or (docs[i][:80] if i < len(docs) else "Unknown")
            summary = meta.get("summary") or (docs[i] if i < len(docs) else "")
            distance = float(dists[i]) if i < len(dists) else 0.0
            matches.append(BookMatch(title=title, summary=summary, distance=distance))

        return matches
