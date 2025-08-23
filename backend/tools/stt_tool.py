# backend/tools/stt_tool.py
import os
import math
import queue
import tempfile
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_SR = 16000   # 16 kHz mono (suficient pentru Whisper)
BLOCK_SIZE = 1024    # mărimea blocului audio (cadru)
DTYPE = "float32"    # format intern

def _rms_dbfs(frame: np.ndarray) -> float:
    """Calculează nivelul în dBFS pentru un cadru mono float32 în [-1, 1]."""
    if frame.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(np.square(frame), dtype=np.float64))
    if rms <= 1e-10:
        return -120.0
    db = 20.0 * math.log10(rms + 1e-12)
    # db ~ 0 pentru semnal la amplitudine 1.0; valori tipice: vorbire -20 .. -10 dBFS, liniște < -45 dBFS
    return float(db)

def record_until_silence(
    max_duration_s: float = 30.0,
    silence_dbfs: float = -40.0,
    min_silence_s: float = 0.8,
    sr: int = DEFAULT_SR,
) -> np.ndarray:
    """
    Înregistrează din microfon până când detectează 'min_silence_s' de liniște sub 'silence_dbfs' dBFS,
    sau până la 'max_duration_s'. Returnează un vector mono float32 în [-1, 1].
    """
    q_audio = queue.Queue()
    collected = []

    # câte blocuri consecutive marcăm ca liniște
    blocks_per_second = sr / BLOCK_SIZE
    silence_blocks_needed = max(1, int(min_silence_s * blocks_per_second))
    silence_counter = 0

    total_blocks_limit = int(max_duration_s * blocks_per_second)
    total_blocks = 0

    print("Start recording… Speak now (auto-stops on silence).")

    def _callback(indata, frames, time, status):
        nonlocal silence_counter, total_blocks
        if status:
            # poți printa status dacă vrei
            pass
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
                # dacă nu mai vin blocuri, ne oprim
                break

            collected.append(block)

            # VAD simplu: dacă nivelul RMS pe bloc e sub prag, creștem contorul; altfel îl resetăm
            level_db = _rms_dbfs(block)
            if level_db < silence_dbfs:
                silence_counter += 1
            else:
                silence_counter = 0

            # oprim pe liniște prelungită
            if silence_counter >= silence_blocks_needed:
                break

            # oprim dacă depășim durata maximă
            if total_blocks >= total_blocks_limit:
                break

    if not collected:
        return np.zeros((0,), dtype=np.float32)
    audio = np.concatenate(collected, axis=0)

    # mici tăieri de capete tăcute (fade-in/out pe baza pragului)
    # opțional: putem decupa tăcerile la început/sfârșit în funcție de prag
    return audio

def save_wav(tmp_audio: np.ndarray, sr: int = DEFAULT_SR) -> str:
    """Salvează buffer-ul audio într-un WAV temporar și returnează calea."""
    fd, tmp_path = tempfile.mkstemp(prefix="llmhw_", suffix=".wav")
    os.close(fd)
    # convertim la int16 pentru fișier WAV
    if tmp_audio.dtype != np.float32:
        tmp_audio = tmp_audio.astype(np.float32)
    audio_int16 = np.int16(np.clip(tmp_audio, -1.0, 1.0) * 32767)
    wav_write(tmp_path, sr, audio_int16)
    return tmp_path

def transcribe_file(path: str, language_hint: Optional[str] = None) -> str:
    """Trimite fișierul la OpenAI Whisper API."""
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
    min_silence_s: float = 1.5,
) -> str:
    """
    Înregistrează până la liniște și transcrie.
    Dacă nu se detectează vorbire reală, returnează 'YOU_SAID_NOTHING'.
    """
    audio = record_until_silence(
        max_duration_s=max_duration_s,
        silence_dbfs=silence_dbfs,
        min_silence_s=min_silence_s,
        sr=DEFAULT_SR,
    )

    # 1) Dacă avem prea puțin audio (<0.5s) => nimic
    if audio.size < DEFAULT_SR * 0.5:
        print("[Voice] No significant audio captured.")
        return "YOU_SAID_NOTHING"

    # 2) Dacă nivelul mediu e foarte mic (sub -45 dBFS) => nimic
    avg_db = _rms_dbfs(audio)
    if avg_db < -45.0:
        print(f"[Voice] Very low level ({avg_db:.1f} dBFS).")
        return "YOU_SAID_NOTHING"

    # Dacă pare audio valid, îl trimitem la Whisper
    tmp_path = save_wav(audio, sr=DEFAULT_SR)
    try:
        print("Stop recording. Transcribing…")
        text = transcribe_file(tmp_path, language_hint=language_hint)

        # 3) Dacă Whisper dă un singur cuvânt și audio <1s => probabil fals
        if (not text) or (len(text.split()) == 1 and audio.size < DEFAULT_SR * 1.0):
            print("[Voice] Whisper result not reliable.")
            return "YOU_SAID_NOTHING"

        return text
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
