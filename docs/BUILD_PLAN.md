# SnapAssist — Build Plan

**Product:** A private, offline AI meeting & notes assistant for Snapdragon-powered HP PCs.
**Architecture:** reusable `edge_infer` NPU core + `SnapAssist` flagship app.
**Target:** HP OmniBook (Snapdragon X / X2), Windows ARM64, Hexagon NPU (80–85 TOPS).
**Non-negotiable:** all inference on the NPU via ONNX Runtime QNN EP / Genie; runs in airplane mode.

---

## Current status (see docs/STATUS.md)

| Phase | State | Notes |
|-------|-------|-------|
| 0 · Environment & proof-of-NPU | ✅ dev / ⬜ device | `verify_npu.py` degrades cleanly off-device; run on the HP PC to confirm QNN |
| 1 · Core engine (headless) | ✅ done | CPU backend runs offline end-to-end (`engine/pipeline.py`) |
| 1b · NPU/Genie wiring | ⬜ device-only | code paths + runbook ready (`docs/DEVICE_SETUP.md`); needs the Snapdragon PC |
| 2 · App shell & UX | ✅ done | FastAPI + offline web UI, verified working in-browser (record/upload/paste → summary → draft → save) |
| 3 · Differentiators | 🟡 partial | language toggle, compute/offline badge, live latency, history done; NPU meter + streaming captions + MeloTTS pending device |
| 4 · Package/present/docs | 🟡 partial | diagram, demo script, packaging config, tests, docs done; actual `.exe` builds on the ARM64 device |

---

## 0. The winning wedge (design everything around this)

SnapAssist must never look like "a thinner Copilot." Its three defensible differentiators —
put them in the app UI, the demo, and every doc:

1. **100% offline & private** — email/meeting audio never leaves the device (the live-demo trick: airplane mode ON).
2. **Indic-first** — transcribe & summarize in Hindi/Tamil/Telugu/etc. (Copilot is weak; Indian judges reward it).
3. **Provably on-NPU** — a live NPU/latency badge, backed by real `edge_infer` benchmarks.

Maps to the 4 judging criteria: Technical Implementation (multi-model NPU pipeline + reusable core),
Use Case & Innovation (privacy + Indic + real daily tool), Deployment & Accessibility (offline `.exe`,
spoken output, Indic access), Presentation & Documentation (live airplane-mode demo + benchmarks + docs).

---

## 1. Phased roadmap

Each phase ends in something you can demo. Do them in order; Phase 0 is a hard gate.

### Phase 0 — Environment & proof-of-NPU  *(highest risk — do first)*
- HP Snapdragon PC: ARM64 Python, `pip install onnxruntime-qnn`, Qualcomm AI Hub account, QAIRT SDK.
- **Gate:** `python scripts/verify_npu.py` prints `QNNExecutionProvider` **and** runs a real model on it.
- Deliverable: a screenshot of that pass. Everything downstream assumes it.

### Phase 1 — Core engine (headless)
- `scripts/fetch_models.py`: export Whisper (ONNX+QNN) and the Genie LLM bundle into `models/`.
- Implement `engine/asr.py` `transcribe_file` and `engine/llm.py` `summarize` / `draft_reply`.
- **Milestone:** `python -m engine.pipeline recordings/sample.wav` prints a summary + action items, offline.
- Capture `edge_infer.benchmark` numbers (NPU vs CPU) — these become your headline metrics.

### Phase 2 — App shell & UX
- `app/tray.py`: system tray + global hotkey (start/stop recording), transcript window, buttons.
- `app/history.py` already scaffolds local SQLite storage.
- **Stack decision (decide here, not at the deadline):** PyQt6 first; if ARM64 wheels fail, fall back
  to local FastAPI + a lightweight web UI in a webview. The engine layer is unchanged either way.
- **Milestone:** click "record" → transcript appears → "Summarize" produces notes, all in the UI.

### Phase 3 — Differentiators (the point-scorers)
- Airplane-mode badge + **live NPU-utilization / latency meter**.
- **Indic toggle** — pick languages Whisper handles best (Hindi first); demo those.
- Streaming transcription (`engine/asr.StreamingTranscriber`) for live captions.
- Optional: MeloTTS spoken responses (accessibility); email workflow from a local `.eml`/Outlook export
  (no live account needed for the demo → safer and still "real").
- **Milestone:** the full "wow" flow works with Wi-Fi physically off.

### Phase 4 — Package, present, document
- Package to a Windows ARM64 installer/`.exe` (PyInstaller or a Tauri bundle over the web UI).
- Demo video: screen recording + voiceover, **airplane mode visible on screen**.
- 1-page proposal (`docs/PROPOSAL.md`) + architecture diagram + reproducible README.
- Publish `edge_infer` as its own GitHub repo with the benchmark table + a short blog post (resume value).

---

## 2. Deployment path (the part judges probe)

```
Qualcomm AI Hub model ──compile──▶ ONNX (+ QNN context binary)
                                      │
                                      ▼
        ONNX Runtime + QNNExecutionProvider  (Whisper, MeloTTS)   ─┐
                                                                    ├─▶ Hexagon NPU
        Genie runtime (context binaries + tokenizer)  (LLM)       ─┘
```
- Quantize on x64 (or a separate x64 Python on the ARM64 box) — ARM64 quantization is limited.
- Cache QNN context binaries (handled in `edge_infer/runtime.py`) so >1GB models don't recompile per launch.

## 3. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| NPU not actually used (silent CPU fallback) | `verify_npu.py` gate + `EdgeSession.on_npu` asserted in the pipeline |
| PyQt6 unavailable on Windows ARM64 | Web-UI fallback decided in Phase 2, engine untouched |
| Indic ASR accuracy | Lead with Hindi + English; treat other languages as "supported," demo the strong ones |
| LLM latency on-device | 3B / mini-class, quantized; stream tokens so it *feels* instant |
| "It's just Copilot" | Lead every artifact with offline + private + Indic + NPU meter |
| Large model downloads | `fetch_models.py` + models/ gitignored; document exact export steps |

## 4. Submission deliverables checklist

- [ ] Working Windows ARM64 prototype (runs offline)
- [ ] `verify_npu.py` PASS screenshot + `edge_infer` NPU-vs-CPU benchmark table
- [ ] Demo video (airplane mode on screen)
- [ ] 1-page proposal + architecture diagram (`docs/`)
- [ ] Reproducible README (AI Hub → QNN setup)
- [ ] `edge_infer` published as standalone repo (resume)
- [ ] One-line "why on-device, not cloud" answer rehearsed for judges

## 5. Resume payoff

- *Built & open-sourced `edge_infer` — deploys Whisper/Llama-3.2/MeloTTS on Qualcomm Hexagon NPU via ONNX Runtime QNN; Nx lower latency and fully offline vs cloud.*
- *Shipped SnapAssist, a private on-device meeting assistant for Windows ARM64; quantized & benchmarked ASR+LLM pipelines; packaged a native installer.*
- Signals a scarce **edge-AI / ML-systems** specialization.
