import os
import uuid
from pathlib import Path
from typing import Optional

from gtts import gTTS
from playsound import playsound

# ---- CLI: redare locală (la fel ca versiunea ta) ----

def speak(text: str, lang: str = "en"):
    if not text or not text.strip():
        return
    try:
        filename = "tts_output.mp3"
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f"[TTS ERROR] {e}")

# ---- API: sintetizează în fișier și întoarce URL relativ (/static/audio/...) ----

def synthesize_to_file(text: str, lang: str = "en", static_audio_dir: str = "backend/static/audio") -> Optional[str]:
    """
    Creează un fișier mp3 în static/audio și întoarce URL-ul relativ
    (ex: /static/audio/<id>.mp3) pentru a fi redat în browser.
    """
    try:
        Path(static_audio_dir).mkdir(parents=True, exist_ok=True)
        file_id = uuid.uuid4().hex
        out_path = Path(static_audio_dir) / f"{file_id}.mp3"

        tts = gTTS(text=text, lang=lang)
        tts.save(str(out_path))

        # fastapi main.py montează /static -> backend/static
        return f"/static/audio/{out_path.name}"
    except Exception as e:
        print(f"[TTS synth error] {e}")
        return None
