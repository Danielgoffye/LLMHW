# backend/api/main.py
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)  # încarcă .env din root înainte de orice import

# DEBUG: normalizează și afișează cheie (evită newline-uri din copy/paste)
import os
key = os.getenv("OPENAI_API_KEY")
if key:
    key = key.strip().strip('"').strip("'")
    os.environ["OPENAI_API_KEY"] = key
print(f"[DEBUG] OPENAI_API_KEY: {repr(key)} (length: {len(key) if key else 0})")

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Routers
from .routes_chat import router as chat_router
from .routes_voice import voice_router
from .routes_tts import tts_router
from .routes_image import router as image_router  # <-- NEW

def create_app() -> FastAPI:
    app = FastAPI(title="LLMHW API", version="0.1.0")

    # CORS pentru frontend local (vite / live server / file:// via simple server)
    default_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost",
        "http://127.0.0.1",
    ]
    extra_origins = os.getenv("CORS_EXTRA_ORIGINS", "")
    origins = default_origins + [o.strip() for o in extra_origins.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static pentru audio & images
    static_dir = Path("backend/static")
    (static_dir / "audio").mkdir(parents=True, exist_ok=True)
    (static_dir / "images").mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Rute API
    app.include_router(chat_router)
    app.include_router(voice_router)
    app.include_router(tts_router)
    app.include_router(image_router)  # <-- NEW

    @app.get("/api/health")
    def health():
        return {"ok": True}

    return app

app = create_app()
