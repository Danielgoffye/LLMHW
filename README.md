# Smart Librarian – AI Book Recommender

## Overview
Smart Librarian is a full-stack AI-powered book recommendation assistant. It uses OpenAI LLMs for chat, book recommendations, translation, and moderation, and provides a modern React-based frontend. The backend is built with FastAPI and supports voice input, text-to-speech, and image generation (DALL-E).

## Features
- Conversational book recommendations (chat with LLM + RAG)
- Book summary lookup and thematic search
- Language detection and translation (RO/EN)
- Offensive language filtering (local + OpenAI Moderation)
- Voice input (speech-to-text)
- Text-to-speech (TTS) for summaries
- Image generation for book summaries (DALL-E)
- Modern, responsive frontend (React, Babel, HTML/CSS)
- Persistent vector store (ChromaDB)

## Project Structure
```
LLMHW/
├── backend/
│   ├── api/                # FastAPI endpoints (chat, voice, tts, image)
│   ├── tools/              # Language, TTS, STT, translation, moderation
│   ├── vector_store/       # ChromaDB vector store, retriever logic
│   ├── data/               # Book summaries (JSON)
│   ├── static/             # Audio and image files for frontend
│   ├── LLMHW.py            # Main logic (chat, RAG, etc)
│   └── ...
├── frontend/
│   └── index.html          # React + Babel SPA frontend
├── requirements.txt        # Python dependencies
├── .env                    # OpenAI API key and secrets (see below)
└── README.md               # This file
```

## Setup & Installation

### 1. Clone the repository
```sh
git clone <repo-url>
cd LLMHW
```

### 2. Python environment
```sh
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install dependencies
```sh
pip install -r requirements.txt
```

### 4. Add your OpenAI API key
Create a file called `.env` in the `backend/` folder (or project root) with:
```
OPENAI_API_KEY=sk-...
```

### 5. Start the backend (FastAPI)
```sh
uvicorn backend.api.main:app --reload --port 8000
```

### 6. Start the frontend (static server)
```sh
python -m http.server 5173
# or use Live Server/other static server in frontend/
```

### 7. Access the app
Open [http://localhost:5173/frontend/](http://localhost:5173/frontend/) in your browser.

## Usage
- Chat with the assistant about books, genres, or ask for recommendations.
- Use voice input (🎤) for speech-to-text.
- Listen to summaries (🔊) or generate images (🎨) for book summaries.
- Offensive language is filtered and will prompt a polite warning.

## Commands & API
- `/api/chat` – Main chat endpoint (POST)
- `/api/tts` – Text-to-speech (POST)
- `/api/voice/transcribe` – Speech-to-text (POST, audio)
- `/api/image/generate` – Image generation (POST)

## Assignment Context
This project was developed as part of the "Essentials of LLM" assignment. It demonstrates:
- LLM-based RAG (Retrieval Augmented Generation)
- Prompt engineering for recommendations
- Language detection, translation, and moderation
- Integration with OpenAI APIs (chat, moderation, TTS, DALL-E)
- Modern frontend integration with backend APIs

## Requirements
- Python 3.10+
- OpenAI API key (with access to chat, moderation, and DALL-E)
- Internet connection

## Notes
- For best results, use a valid OpenAI key with access to all required models.
- The vector store (ChromaDB) is persistent in `backend/vector_store/chroma_db`.
- You can extend the book summaries in `backend/data/book_summaries.json`.

## Authors
- Daniel Rotaru

---
