# backend/api/routes_tts.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

from backend.tools.tts_tool import synthesize_to_file

tts_router = APIRouter(prefix="/api", tags=["tts"])

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    lang: str = Field(default="en")

class TTSResponse(BaseModel):
    url: str

@tts_router.post("/tts", response_model=TTSResponse)
def tts(req: TTSRequest):
    url = synthesize_to_file(req.text, lang=req.lang, static_audio_dir="backend/static/audio")
    if not url:
        raise HTTPException(status_code=500, detail="TTS synthesis failed")
    return JSONResponse({"url": url})
