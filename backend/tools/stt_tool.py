import os
import math
import queue
import tempfile
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
# ...existing code...
from openai import OpenAI
# .env este încărcat o singură dată în main.py

DEFAULT_SR = 16000
BLOCK_SIZE = 1024
DTYPE = "float32"

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        api_key = api_key.strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Set it in .env or environment.")
    return OpenAI(api_key=api_key)

def _rms_dbfs(frame: np.ndarray) -> float:
    if frame.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(np.square(frame), dtype=np.float64))
    if rms <= 1e-10:
        return -120.0
    return float(20.0 * math.log10(rms + 1e-12))

def record_until_silence(
    max_duration_s: float = 30.0,
    silence_dbfs: float = -40.0,
    min_silence_s: float = 0.8,
    sr: int = DEFAULT_SR,
) -> np.ndarray:
    """
    Înregistrează până când detectează o perioadă de liniște.
    """
    q_audio = queue.Queue()
    collected = []

    blocks_per_second = sr / BLOCK_SIZE
    silence_blocks_needed = max(1, int(min_silence_s * blocks_per_second))
    silence_counter = 0

    total_blocks_limit = int(max_duration_s * blocks_per_second)
    total_blocks = 0

    print("Start recording… Speak now (auto-stops on silence).")

    def _callback(indata, frames, time, status):
        nonlocal silence_counter, total_blocks
        mono = indata[:, 0].astype(np.float32, copy=False)
        q_audio.put(mono.copy())
        total_blocks += 1

    with sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype=DTYPE,
        blocksize=BLOCK_SIZE,
        callback=_callback,
    ):
        while True:
            try:
                block = q_audio.get(timeout=1.0)
            except queue.Empty:
                break

            collected.append(block)

            level_db = _rms_dbfs(block)
            if level_db < silence_dbfs:
                silence_counter += 1
            else:
                silence_counter = 0

            if silence_counter >= silence_blocks_needed:
                break
            if total_blocks >= total_blocks_limit:
                break

    if not collected:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(collected, axis=0)

def _save_wav(buf: np.ndarray, sr: int = DEFAULT_SR) -> str:
    fd, path = tempfile.mkstemp(prefix="llmhw_", suffix=".wav")
    os.close(fd)
    if buf.dtype != np.float32:
        buf = buf.astype(np.float32)
    audio_int16 = np.int16(np.clip(buf, -1.0, 1.0) * 32767)
    wav_write(path, sr, audio_int16)
    return path

def transcribe_file(path: str, language_hint: Optional[str] = None) -> str:
    """
    Trimite fișierul la OpenAI Whisper API.
    Lazy-init pentru client.
    """
    client = _get_client()
    with open(path, "rb") as f:
        kwargs = {"model": "whisper-1", "file": f}
        if language_hint:
            kwargs["language"] = language_hint
        resp = client.audio.transcriptions.create(**kwargs)
    return (resp.text or "").strip()

def capture_and_transcribe_vad(
    language_hint: Optional[str] = None,
    max_duration_s: float = 30.0,
    silence_dbfs: float = -40.0,
    min_silence_s: float = 1.3,
) -> str:
    """
    Înregistrează până la liniște și transcrie. Dacă nu detectăm vorbire reală,
    întoarcem YOU_SAID_NOTHING.
    """
    audio = record_until_silence(
        max_duration_s=max_duration_s,
        silence_dbfs=silence_dbfs,
        min_silence_s=min_silence_s,
        sr=DEFAULT_SR,
    )

    # garduri împotriva „phantom speech”
    if audio.size < DEFAULT_SR * 0.5:
        print("[Voice] No significant audio captured.")
        return "YOU_SAID_NOTHING"

    avg_db = _rms_dbfs(audio)
    if avg_db < -45.0:
        print(f"[Voice] Very low level ({avg_db:.1f} dBFS).")
        return "YOU_SAID_NOTHING"

    wav_path = _save_wav(audio, sr=DEFAULT_SR)
    try:
        print("Stop recording. Transcribing…")
        text = transcribe_file(wav_path, language_hint=language_hint)
        if (not text) or (len(text.split()) == 1 and audio.size < DEFAULT_SR * 1.0):
            print("[Voice] Whisper result not reliable.")
            return "YOU_SAID_NOTHING"
        return text
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass
