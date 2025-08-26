from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1)
    locale: str | None = None   # opțional: forțezi limba (ex. "ro", "en")

class ChatResponse(BaseModel):
    answer: str
    summary: str | None = None
    lang: str
    tts_available: bool = False
