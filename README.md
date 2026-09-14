# SnapAssist

**A private, offline AI meeting & notes assistant for Snapdragon-powered HP PCs.**
Record → transcribe → summarize → action items → draft follow-ups — 100% on-device,
English + Indic languages, zero cloud. Built for the Snapdragon AI Lab Build & Present Challenge.

Everything runs on the **Hexagon NPU** via **ONNX Runtime + QNN Execution Provider**.
No audio, no notes, no email ever leaves the machine — the demo runs in airplane mode.

---

## Two layers

SnapAssist is deliberately split so the bottom layer is a reusable, resume-worthy artifact.

```
┌──────────────────────────── SnapAssist (flagship app) ───────────────────────────┐
│  system tray + global hotkey · transcript UI · summaries · draft replies · history │
└───────────────────────────────────────────────────────────────────────────────────┘
                                        │  uses
                                        ▼
┌──────────────────────────── edge_infer (reusable core) ──────────────────────────┐
│  load any Qualcomm AI Hub / ONNX model on the Hexagon NPU (QNN EP)                 │
│  automatic CPU fallback · device/NPU verification · benchmark harness              │
└───────────────────────────────────────────────────────────────────────────────────┘
        ONNX Runtime 1.19+ · QNNExecutionProvider · Genie (LLM) · Windows ARM64
```

- **`edge_infer/`** — the open-sourceable core: a thin runtime that puts Whisper / Llama-3.2 /
  MeloTTS on the NPU, falls back to CPU on dev machines, and benchmarks NPU vs CPU.
- **`engine/`** — the flagship pipeline (ASR → LLM → TTS) built on `edge_infer`.
- **`app/`** — the tray app + UI.

## Model pipeline (all from Qualcomm AI Hub)

| Stage | Model | Runtime |
|-------|-------|---------|
| Transcription (streaming) | Whisper-Base / Distil-Whisper | ONNX + QNN EP |
| Summary / action items / drafts / Q&A | Phi-3.5-mini or Llama-3.2-3B | Genie (QNN) |
| Spoken responses (optional) | MeloTTS (EN/Indic) | ONNX + QNN EP |

## Quick start (Windows ARM64)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
# Phase 0 — the single most important check: is the NPU actually usable?
python scripts/verify_npu.py
```

`verify_npu.py` must print `QNNExecutionProvider` as an available provider and successfully
run a tiny model on it. **Do not build further until this passes** — the whole submission
depends on it.

## Repo layout

```
snapassist/
├── edge_infer/       # reusable NPU runtime (the resume piece)
│   ├── runtime.py    # QNN session factory + NPU/device check + fallback
│   ├── benchmark.py  # NPU vs CPU latency / throughput harness
│   └── models.py     # AI Hub model registry
├── engine/           # flagship ASR→LLM→TTS pipeline
│   ├── asr.py        # Whisper (streaming)
│   ├── llm.py        # summarize / draft / answer (Genie)
│   ├── tts.py        # MeloTTS speak-back
│   └── pipeline.py   # end-to-end meeting flow
├── app/              # tray app + UI + local history
├── scripts/          # verify_npu.py, fetch_models.py
├── models/           # downloaded ONNX artifacts (gitignored)
└── docs/             # BUILD_PLAN.md, PROPOSAL.md, architecture
```

## Status

**Phases 1–2 complete (CPU dev backend); Phase 3–4 partial; NPU/Genie ready for the device.**
See [docs/STATUS.md](docs/STATUS.md) for the exact done-vs-device breakdown.

The full app runs offline on any dev machine:

```bash
pip install -r requirements-dev.txt

# 1) the web app (record / upload / paste → summary → draft → save), offline
python -m app.server            # → http://127.0.0.1:8000

# 2) or the headless pipeline
python -m engine.pipeline recordings/jfk.wav --reply          # from audio
python -m engine.pipeline --text recordings/sample_meeting.txt --reply   # from text

# 3) tests
pytest -q
```

On the Snapdragon HP PC, flip to the NPU with `SNAP_ASR_BACKEND=npu SNAP_LLM_BACKEND=genie`
(or the pipeline flags) — see [docs/DEVICE_SETUP.md](docs/DEVICE_SETUP.md).

Backends switch with flags — `--asr-backend npu` and `--llm-backend genie` on the Snapdragon
HP PC (Phase 1b). Dev-model quality is limited by the tiny 0.5B model; bump it via
`LocalLLM(model_id="Qwen/Qwen2.5-1.5B-Instruct")`, or use Genie on-device for full quality.

See [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) for the roadmap and
[docs/PROPOSAL.md](docs/PROPOSAL.md) for the submission outline.
