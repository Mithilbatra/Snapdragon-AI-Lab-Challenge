# SnapAssist

A private, offline AI meeting assistant for Snapdragon-powered HP PCs. Record a meeting and get a
transcript, a summary with decisions and action items, a drafted follow-up email, and answers to
questions about the discussion — in English and Indian languages, entirely on the device.

Every model runs on the Hexagon NPU through ONNX Runtime's QNN Execution Provider (the LLM via
Genie). No audio, notes or drafts leave the machine, and the whole thing works in airplane mode.

Built for the Snapdragon AI Lab Build & Present Challenge.

## Layout

SnapAssist is split into a reusable NPU runtime and the app built on top of it:

- **`edge_infer/`** — runs an exported model on the NPU via the QNN EP, falls back to CPU for
  development, and benchmarks the two. Reusable across projects.
- **`engine/`** — the meeting pipeline: Whisper (ASR), an LLM for summaries/drafts/Q&A, MeloTTS.
- **`app/`** — a local FastAPI server and an offline web UI, plus SQLite history.

```
snapassist/
├── edge_infer/
│   ├── runtime.py     # QNN session factory, NPU detection, CPU fallback
│   ├── benchmark.py   # NPU vs CPU latency
│   └── models.py      # AI Hub model registry
├── engine/
│   ├── asr.py         # Whisper (cpu / npu backends) + streaming
│   ├── llm.py         # summarize / draft / answer (cpu / genie backends)
│   ├── tts.py         # MeloTTS
│   ├── audio.py       # dependency-free WAV loading
│   └── pipeline.py    # end-to-end flow + CLI
├── app/
│   ├── server.py      # FastAPI endpoints
│   ├── launcher.py    # packaged-app entry point
│   ├── history.py     # SQLite
│   └── ui/            # index.html + app.js (no external deps)
├── scripts/           # verify_npu.py, fetch_models.py, build_exe.py
├── tests/
└── docs/
```

## Models (Qualcomm AI Hub)

| Stage | Model | Runtime |
|-------|-------|---------|
| Transcription | Whisper | ONNX + QNN EP |
| Summary / decisions / action items / drafts / Q&A | Phi-3.5-mini or Llama-3.2-3B | Genie |
| Spoken read-back (optional) | MeloTTS | ONNX + QNN EP |

## Run it (development, CPU — any machine)

```bash
pip install -r requirements-dev.txt

python -m app.server                                    # http://127.0.0.1:8000
python -m engine.pipeline recordings/jfk.wav --reply    # or from audio on the CLI
python -m engine.pipeline --text recordings/sample_meeting.txt --reply
pytest -q
```

The dev backend uses a small 0.5 B model, so summaries are modest; the on-device model is far
better. Bump the dev model with `LocalLLM(model_id="Qwen/Qwen2.5-1.5B-Instruct")` if you want
nicer local output.

## Run it on the Snapdragon HP PC

```powershell
pip install -r requirements.txt
python scripts/verify_npu.py            # must report QNNExecutionProvider
python scripts/fetch_models.py          # export AI Hub models
$env:SNAP_ASR_BACKEND="npu"; $env:SNAP_LLM_BACKEND="genie"
python -m app.server
```

Full device runbook: [docs/DEVICE_SETUP.md](docs/DEVICE_SETUP.md). Status of what's built vs. what
needs the device: [docs/STATUS.md](docs/STATUS.md). Roadmap: [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md).
