import sys
import os

sys.path.append(os.path.abspath("backend"))
from backend.vector_store.retriever import BookRetriever

if __name__ == "__main__":
    retriever = BookRetriever()
    question = "I want a book about friendship and magic"
    results = retriever.query(question, top_k=3)

    print(f"Query: {question}\n")
    for i, r in enumerate(results, start=1):
        print(f"{i}. {r.title} | distance: {r.distance:.4f}")
        print(f"   summary: {r.summary}\n")
