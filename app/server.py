"""Local FastAPI server for the SnapAssist web UI. Binds to localhost; nothing leaves the device.

Backends are selected by environment variable so the same server runs on a dev machine (CPU) and
on the Snapdragon PC (NPU / Genie) unchanged:

    SNAP_ASR_BACKEND  cpu | npu     (default cpu)
    SNAP_LLM_BACKEND  cpu | genie   (default cpu)
    SNAP_ASR_MODEL    tiny | base | small           (CPU dev only)
    SNAP_LLM_MODEL    <hf id>                        (CPU dev only)
    SNAP_PORT         default 8000
"""

from __future__ import annotations

import io
import os
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.history import History
from edge_infer import npu_report
from engine.asr import Whisper
from engine.audio import resample
from engine.llm import LocalLLM

UI = Path(__file__).parent / "ui"
ASR_BACKEND = os.getenv("SNAP_ASR_BACKEND", "cpu")
LLM_BACKEND = os.getenv("SNAP_LLM_BACKEND", "cpu")
ASR_MODEL = os.getenv("SNAP_ASR_MODEL", "tiny")
LLM_MODEL = os.getenv("SNAP_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

app = FastAPI(title="SnapAssist", docs_url=None, redoc_url=None)

_lock = threading.Lock()
_asr: Whisper | None = None
_llm: LocalLLM | None = None
_history = History()
_latency = {"asr": deque(maxlen=10), "llm": deque(maxlen=10)}


def asr() -> Whisper:
    global _asr
    with _lock:
        if _asr is None:
            _asr = Whisper(backend=ASR_BACKEND, model_size=ASR_MODEL)
        return _asr


def llm() -> LocalLLM:
    global _llm
    with _lock:
        if _llm is None:
            _llm = LocalLLM(backend=LLM_BACKEND, model_id=LLM_MODEL)
        return _llm


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


def _decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise HTTPException(400, "Only 16-bit PCM WAV is supported.")
        rate, channels = wav.getframerate(), wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        audio, rate = resample(audio, rate, 16000), 16000
    return audio, rate


def _record_latency(kind: str, seconds: float) -> None:
    _latency[kind].append(seconds)


def _mean(values) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI / "index.html")


@app.get("/app.js")
def script() -> FileResponse:
    return FileResponse(UI / "app.js", media_type="application/javascript")


@app.get("/api/sample")
def sample() -> JSONResponse:
    path = Path("recordings/sample_meeting.txt")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return JSONResponse({"transcript": text})


@app.get("/api/status")
def status() -> JSONResponse:
    report = npu_report()
    on_npu = ASR_BACKEND == "npu" and report["npu_available"]
    return JSONResponse({
        "asr_backend": ASR_BACKEND,
        "llm_backend": LLM_BACKEND,
        "asr_model": ASR_MODEL if ASR_BACKEND == "cpu" else "Whisper (AI Hub / QNN)",
        "llm_model": LLM_MODEL if LLM_BACKEND == "cpu" else "Phi-3.5 / Llama-3.2 (Genie)",
        "compute": "Hexagon NPU" if on_npu else "CPU (dev)",
        "avg_asr_secs": _mean(_latency["asr"]),
        "avg_llm_secs": _mean(_latency["llm"]),
        "offline": True,
        "npu": report,
    })


@app.post("/api/transcribe")
def transcribe(file: UploadFile) -> JSONResponse:
    audio, rate = _decode_wav(file.file.read())
    engine = asr()
    start = time.perf_counter()
    try:
        result = engine.transcribe(audio, rate, "en")
    except (RuntimeError, FileNotFoundError, ImportError) as exc:
        raise HTTPException(503, f"ASR backend not ready: {exc}")
    elapsed = round(time.perf_counter() - start, 2)
    _record_latency("asr", elapsed)
    return JSONResponse({
        "transcript": result.text,
        "asr_secs": elapsed,
        "on_npu": engine.on_npu,
        "audio_secs": round(len(audio) / rate, 1),
    })


@app.post("/api/summarize")
def summarize(req: SummarizeReq) -> JSONResponse:
    if not req.transcript.strip():
        raise HTTPException(400, "Empty transcript.")
    start = time.perf_counter()
    try:
        notes = llm().summarize(req.transcript, req.language)
    except (RuntimeError, FileNotFoundError, ImportError) as exc:
        raise HTTPException(503, f"LLM backend not ready: {exc}")
    elapsed = round(time.perf_counter() - start, 2)
    _record_latency("llm", elapsed)
    return JSONResponse({
        "summary": notes.summary,
        "decisions": notes.decisions,
        "action_items": notes.action_items,
        "llm_secs": elapsed,
    })


@app.post("/api/draft")
def draft(req: DraftReq) -> JSONResponse:
    start = time.perf_counter()
    email = llm().draft_reply(req.summary, req.language)
    return JSONResponse({"email": email, "llm_secs": round(time.perf_counter() - start, 2)})


@app.post("/api/ask")
def ask(req: AskReq) -> JSONResponse:
    start = time.perf_counter()
    answer = llm().answer(req.question, req.context)
    return JSONResponse({"answer": answer, "llm_secs": round(time.perf_counter() - start, 2)})


@app.get("/api/history")
def history() -> JSONResponse:
    return JSONResponse([dict(row) for row in _history.recent()])


@app.post("/api/save")
def save(req: SaveReq) -> JSONResponse:
    mid = _history.add(req.title, req.language, req.transcript, req.summary, req.action_items)
    return JSONResponse({"id": mid})


def main() -> None:
    import uvicorn
    port = int(os.getenv("SNAP_PORT", "8000"))
    print(f"SnapAssist on http://127.0.0.1:{port}  (asr={ASR_BACKEND}, llm={LLM_BACKEND}, offline)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
