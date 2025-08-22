import os
from gtts import gTTS
from playsound import playsound

def speak(text: str, lang: str = "en"):
    if not text.strip():
        return
    try:
        filename = "tts_output.mp3"
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f"[TTS ERROR] {e}")
