from fastapi import APIRouter, HTTPException
from .schemas import ChatRequest, ChatResponse

# Folosim direct logica existentă (CLI) ca să nu dublăm codul:
from ..LLMHW import chat_with_llm
from backend.tools.translation_tool import detect_language  # doar dacă vrei să forțezi locale

router = APIRouter(prefix="/api", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    user_text = (req.text or "").strip()
    if not user_text:
        raise HTTPException(status_code=422, detail="Empty text")

    # Dacă ai vrea să forțezi limba de răspuns (ex. req.locale),
    # poți pre-procesa aici. Pentru moment, lăsăm fluxul actual să decidă.
    answer, lang, summary = chat_with_llm(user_text)

    return ChatResponse(
        answer=answer,
        summary=summary,
        lang=lang,
        tts_available=bool(summary and summary.strip())
    )
