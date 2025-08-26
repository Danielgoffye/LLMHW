# backend/api/routes_voice.py
import os
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from backend.tools.stt_tool import transcribe_file

voice_router = APIRouter(prefix="/api/voice", tags=["voice"])

ALLOWED_MIME = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/webm", "audio/ogg", "audio/m4a", "audio/x-m4a"
}
MAX_SIZE_BYTES = 25 * 1024 * 1024  # 25MB

@voice_router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if audio.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported audio type: {audio.content_type}")

    # Limită de mărime (best-effort; pentru fișiere mari, folosește streaming sau un reverse proxy limit)
    data = await audio.read()
    if len(data) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (>25MB)")

    # Salvăm temporar și transcriem
    fd, path = tempfile.mkstemp(prefix="llmhw_upl_", suffix=os.path.splitext(audio.filename or '')[-1] or ".wav")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(data)

        text = transcribe_file(path, language_hint=None)  # poți pune "ro" dacă vrei hint
        if not text or text == "YOU_SAID_NOTHING":
            return JSONResponse({"text": ""})
        return JSONResponse({"text": text})
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
