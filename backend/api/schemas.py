# backend/api/schemas.py
from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    answer: str
    lang: str
    summary: Optional[str] = None
    tts_available: bool = True
    title: Optional[str] = None
