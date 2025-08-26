# backend/api/routes_chat.py
from fastapi import APIRouter, HTTPException
from .schemas import ChatRequest, ChatResponse

# Refolosim logica de chat (nu duplicăm):
from ..LLMHW import chat_with_llm

router = APIRouter(prefix="/api", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    user_text = (req.text or "").strip()
    if not user_text:
        raise HTTPException(status_code=422, detail="Empty text")

    # IMPORTANT: chat_with_llm trebuie să întoarcă acum 4 valori:
    # (answer: str, lang: str, summary: Optional[str], title: Optional[str])
    answer, lang, summary, title = chat_with_llm(user_text)

    return ChatResponse(
        answer=answer,
        summary=summary,
        lang=lang,
        tts_available=bool(summary and summary.strip()),
        title=title
    )
