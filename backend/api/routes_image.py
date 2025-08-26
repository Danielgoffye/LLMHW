# backend/api/routes_image.py
from __future__ import annotations
import os, base64, uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

router = APIRouter(prefix="/api/image", tags=["image"])

class ImageGenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    style: Optional[str] = Field(default="vivid")
    size: Optional[str] = Field(default="1024x1024")
    n: Optional[int] = Field(default=1, ge=1, le=4)

class ImageResponse(BaseModel):
    images: list[dict]
    success: bool

def _get_client() -> OpenAI:
    key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing.")
    return OpenAI(api_key=key)

@router.post("/generate", response_model=ImageResponse)
def generate_image(req: ImageGenRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt.")
    size = req.size if req.size in ["1024x1024","1024x1792","1792x1024"] else "1024x1024"
    n = max(1, min(4, req.n or 1))
    client = _get_client()
    try:
        resp = client.images.generate(model="dall-e-2", prompt=prompt, size=size, n=n)
        images = []
        out_dir = Path("backend/static/images"); out_dir.mkdir(parents=True, exist_ok=True)
        for d in resp.data:
            if getattr(d, "b64_json", None):
                img_bytes = base64.b64decode(d.b64_json)
                fn = f"{uuid.uuid4().hex}.png"
                (out_dir / fn).write_bytes(img_bytes)
                images.append({"url": f"/static/images/{fn}", "filename": fn})
            elif getattr(d, "url", None):
                images.append({"url": d.url, "filename": f"gen_{uuid.uuid4().hex}.png"})
        return ImageResponse(images=images, success=True)
    except Exception as e:
        print("Image gen error:", e)
        # Returnează mereu un JSON valid, nu doar HTTPException
        return ImageResponse(images=[], success=False)
