import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PERSIST_PATH = "backend/vector_store/chroma_db"
COLLECTION_NAME = "books"

chroma_client = chromadb.PersistentClient(path=PERSIST_PATH)
col = chroma_client.get_collection(COLLECTION_NAME)

query = "friendship and magic"
# Option 1: use LLM embedding for the query
q_emb = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding

res = col.query(query_embeddings=[q_emb], n_results=3, include=["metadatas", "documents", "distances"])
for i, title in enumerate(res["metadatas"][0]):
    print(i+1, "-", title["title"], "| score:", res["distances"][0][i])
