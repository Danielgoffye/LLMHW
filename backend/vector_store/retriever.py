import os
from typing import List, Dict, Any
from dataclasses import dataclass

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class RetrievalItem:
    title: str
    summary: str
    distance: float


class BookRetriever:
    def __init__(
        self,
        persist_path: str = "backend/vector_store/chroma_db",
        collection_name: str = "books",
        embed_model: str = "text-embedding-3-small",
    ):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing in backend/.env")

        self._client = OpenAI(api_key=api_key)
        self._embed_model = embed_model

        self._chroma = chromadb.PersistentClient(path=persist_path)
        try:
            self._col = self._chroma.get_collection(collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Collection '{collection_name}' not found at {persist_path}. "
                f"Build it first with vector_store_builder.py."
            ) from e

    def _embed(self, text: str) -> List[float]:
        return self._client.embeddings.create(
            model=self._embed_model, input=text
        ).data[0].embedding

    def query(self, question: str, top_k: int = 5) -> List[RetrievalItem]:
        q_emb = self._embed(question)
        res = self._col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        items: List[RetrievalItem] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            items.append(
                RetrievalItem(
                    title=meta.get("title", "Unknown"),
                    summary=doc,
                    distance=float(dist),
                )
            )
        return items
