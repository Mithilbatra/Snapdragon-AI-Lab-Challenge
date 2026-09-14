"""SnapAssist local server — FastAPI backend for the offline web UI.

Everything runs on localhost against the on-device engine; no request ever leaves the machine.
Sync endpoints are used deliberately: FastAPI runs them in a threadpool, so a slow model call
doesn't block the event loop, and we avoid manual executor juggling.

Backends are chosen by env vars so the same server runs on a dev Mac (CPU) and on the
Snapdragon HP PC (NPU/Genie) with no code change:

    SNAP_ASR_BACKEND = cpu | npu       (default cpu)
    SNAP_LLM_BACKEND = cpu | genie     (default cpu)
    SNAP_LLM_MODEL   = <hf id>         (default Qwen/Qwen2.5-0.5B-Instruct, dev only)
    SNAP_ASR_MODEL   = tiny|base|...   (default tiny, dev only)

Run:  python -m app.server   (or: uvicorn app.server:app)
"""

from __future__ import annotations

import io
import os
import threading
import time
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.history import History
from edge_infer import npu_report
from engine.asr import Transcript, Whisper
from engine.llm import LocalLLM

UI_DIR = Path(__file__).parent / "ui"

ASR_BACKEND = os.getenv("SNAP_ASR_BACKEND", "cpu")
LLM_BACKEND = os.getenv("SNAP_LLM_BACKEND", "cpu")
ASR_MODEL = os.getenv("SNAP_ASR_MODEL", "tiny")
LLM_MODEL = os.getenv("SNAP_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

app = FastAPI(title="SnapAssist", docs_url=None, redoc_url=None)

# --- lazily-loaded, shared model instances (load once, reuse) ---------------
_lock = threading.Lock()
_asr: Whisper | None = None
_llm: LocalLLM | None = None
_history = History()


def get_asr() -> Whisper:
    global _asr
    with _lock:
        if _asr is None:
            _asr = Whisper(backend=ASR_BACKEND, model_size=ASR_MODEL)
        return _asr


def get_llm() -> LocalLLM:
    global _llm
    with _lock:
        if _llm is None:
            _llm = LocalLLM(backend=LLM_BACKEND, model_id=LLM_MODEL)
        return _llm


# --- request models ---------------------------------------------------------
class SummarizeReq(BaseModel):
    transcript: str
    language: str = "English"


class DraftReq(BaseModel):
    summary: str
    language: str = "English"


class AskReq(BaseModel):
    question: str
    context: str


class SaveReq(BaseModel):
    title: str = "Untitled meeting"
    language: str = "English"
    transcript: str = ""
    summary: str = ""
    action_items: list[str] = []


# --- helpers ----------------------------------------------------------------
def _wav_bytes_to_float(data: bytes, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Decode uploaded 16-bit PCM WAV bytes to float mono at target_rate (no ffmpeg)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getsampwidth() != 2:
            raise HTTPException(400, "Only 16-bit PCM WAV is supported.")
        rate = w.getframerate()
        ch = w.getnchannels()
        frames = w.readframes(w.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if rate != target_rate:
        from engine.audio import _resample_linear
        audio = _resample_linear(audio, rate, target_rate)
    return audio, target_rate


# --- routes -----------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/app.js")
def appjs() -> FileResponse:
    return FileResponse(UI_DIR / "app.js", media_type="application/javascript")


@app.get("/api/status")
def status() -> JSONResponse:
    rep = npu_report()
    on_npu = ASR_BACKEND == "npu" and rep["npu_available"]
    return JSONResponse({
        "asr_backend": ASR_BACKEND,
        "llm_backend": LLM_BACKEND,
        "asr_model": ASR_MODEL if ASR_BACKEND == "cpu" else "whisper (AI Hub / QNN)",
        "llm_model": LLM_MODEL if LLM_BACKEND == "cpu" else "Phi-3.5 / Llama-3.2 (Genie)",
        "compute": "Hexagon NPU" if on_npu else "CPU (dev)",
        "npu": rep,
        "offline": True,
    })


@app.post("/api/transcribe")
def transcribe(file: UploadFile) -> JSONResponse:
    data = file.file.read()
    audio, rate = _wav_bytes_to_float(data)
    asr = get_asr()
    t0 = time.perf_counter()
    # Reuse the backend dispatch by writing a tiny in-memory path:
    tr: Transcript = (asr._transcribe_npu if asr.backend == "npu" else asr._transcribe_cpu)(
        audio, rate, "en")
    return JSONResponse({
        "transcript": tr.text,
        "language": tr.language,
        "asr_secs": round(time.perf_counter() - t0, 2),
        "on_npu": asr.on_npu,
        "audio_secs": round(len(audio) / rate, 1),
    })


@app.post("/api/summarize")
def summarize(req: SummarizeReq) -> JSONResponse:
    if not req.transcript.strip():
        raise HTTPException(400, "Empty transcript.")
    llm = get_llm()
    t0 = time.perf_counter()
    notes = llm.summarize(req.transcript, language=req.language)
    return JSONResponse({
        "summary": notes.summary,
        "decisions": notes.decisions,
        "action_items": notes.action_items,
        "llm_secs": round(time.perf_counter() - t0, 2),
    })


@app.post("/api/draft")
def draft(req: DraftReq) -> JSONResponse:
    llm = get_llm()
    t0 = time.perf_counter()
    email = llm.draft_reply(req.summary, language=req.language)
    return JSONResponse({"email": email, "llm_secs": round(time.perf_counter() - t0, 2)})


@app.post("/api/ask")
def ask(req: AskReq) -> JSONResponse:
    llm = get_llm()
    t0 = time.perf_counter()
    answer = llm.answer(req.question, req.context)
    return JSONResponse({"answer": answer, "llm_secs": round(time.perf_counter() - t0, 2)})


@app.get("/api/history")
def history() -> JSONResponse:
    rows = _history.recent()
    return JSONResponse([dict(r) for r in rows])


@app.post("/api/save")
def save(req: SaveReq) -> JSONResponse:
    mid = _history.add(req.title, req.language, req.transcript, req.summary, req.action_items)
    return JSONResponse({"id": mid})


def main() -> None:
    import uvicorn
    port = int(os.getenv("SNAP_PORT", "8000"))
    print(f"SnapAssist → http://127.0.0.1:{port}   "
          f"(ASR={ASR_BACKEND}, LLM={LLM_BACKEND}, offline)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
