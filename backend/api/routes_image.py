# backend/api/routes_image.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

router = APIRouter(prefix="/api/image", tags=["image"])

def _get_client() -> OpenAI:
    raw = os.getenv("OPENAI_API_KEY", "")
    api_key = raw.strip().strip('"').strip("'")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Set it in .env or environment.")
    return OpenAI(api_key=api_key)

class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="What to draw/generate")
    size: Optional[str] = Field(default="1024x1024", description="256x256 | 512x512 | 1024x1024")

@dataclass
class ImageResponse:
    data_url: str

@router.post("/generate")
def generate_image(payload: ImageRequest):
    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty.")

    try:
        client = _get_client()
        # OpenAI Images API – modelul actual recomandat pentru generare
        resp = client.images.generate(
            model="gpt-4o-mini",
            prompt=prompt,
            size=payload.size or "1024x1024",
            # OPTIONAL: quality="high",
            # OPTIONAL: background="transparent",
            n=1,
        )
        b64 = resp.data[0].b64_json
        data_url = f"data:image/png;base64,{b64}"
        return {"data_url": data_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")
