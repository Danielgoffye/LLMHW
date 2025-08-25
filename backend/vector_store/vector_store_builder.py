import os
import json
import shutil
# ...existing code...
from openai import OpenAI
import chromadb

# --- Config ---
DATA_PATH = "backend/data/book_summaries.json"
PERSIST_PATH = "backend/vector_store/chroma_db"
COLLECTION_NAME = "books"
EMBED_MODEL = "text-embedding-3-small"

# --- Setup secrets ---
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    api_key = api_key.strip()
if not api_key:
    raise ValueError("OPENAI_API_KEY missing in backend/.env")
client = OpenAI(api_key=api_key)

# --- Load data ---
with open(DATA_PATH, "r", encoding="utf-8") as f:
    books = json.load(f)
if not isinstance(books, list) or not books:
    raise ValueError("book_summaries.json must be a non-empty JSON list")

# --- Ensure a clean persistent directory (optional but clear for first build) ---
# Comment the next 3 lines if you want to keep previous data and just upsert new docs.
if os.path.exists(PERSIST_PATH):
    shutil.rmtree(PERSIST_PATH)

# --- Persistent Chroma client ---
chroma_client = chromadb.PersistentClient(path=PERSIST_PATH)

# Drop existing collection if present (for a clean rebuild)
existing = [c.name for c in chroma_client.list_collections()]
if COLLECTION_NAME in existing:
    chroma_client.delete_collection(COLLECTION_NAME)

collection = chroma_client.create_collection(name=COLLECTION_NAME)

# --- Build embeddings and add to Chroma ---
ids = []
docs = []
metas = []
vectors = []

for idx, book in enumerate(books):
    title = book["title"].strip()
    summary = book["summary"].strip()
    if not title or not summary:
        continue

    # Create embedding
    emb = client.embeddings.create(model=EMBED_MODEL, input=summary).data[0].embedding

    ids.append(f"book-{idx}")
    docs.append(summary)
    metas.append({"title": title})
    vectors.append(emb)

# Important: pass `embeddings` explicitly so Chroma stores vectors
collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)

print(f"OK: Built collection '{COLLECTION_NAME}' with {len(ids)} items.")
print(f"Persisted at: {os.path.abspath(PERSIST_PATH)}")
