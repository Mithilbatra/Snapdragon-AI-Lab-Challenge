# SnapAssist — status: what's built, what needs the device

This project was scaffolded and developed on a Mac (no Hexagon NPU). Everything that can be built
and verified without Snapdragon hardware is **done and tested**; the NPU/Genie paths are written
and documented but can only be validated **on the HP Snapdragon PC**.

## ✅ Done and verified (runs offline on any dev machine, CPU)
- **`edge_infer` core** — QNN-EP session factory with NPU detection + CPU fallback + context
  caching (`runtime.py`), benchmark harness (`benchmark.py`), model registry (`models.py`).
  Degrades gracefully when `onnxruntime` / QNN is absent.
- **Engine (Phase 1)** — Whisper ASR (transformers CPU backend) + LLM summarize/draft/answer
  (Qwen2.5 CPU backend) + end-to-end `engine/pipeline.py`. Transcribes and summarizes offline.
- **App (Phase 2)** — FastAPI server + self-contained offline web UI. Verified in a real browser:
  paste/upload/record → transcribe → summarize (decisions + action items) → draft email → ask →
  save to local SQLite history. Client-side mic→16 kHz WAV encoding (no ffmpeg, nothing uploaded raw).
- **Differentiators (Phase 3, partial)** — language selector (English + 6 Indic), compute badge,
  offline indicator, per-request latency chips, local history.
- **Phase 4 assets** — architecture diagram (`docs/architecture.svg`), demo script, packaging
  config (`scripts/build_exe.py`, `app/launcher.py`, `docs/PACKAGING.md`), smoke tests (all pass).

### Measured CPU baseline (this Mac, warm, offline)
| Stage | Model | Input | Latency |
|-------|-------|-------|---------|
| ASR | whisper-tiny | 11s audio | ~11s |
| Summarize | Qwen2.5-0.5B | ~550-char transcript | ~12–21s |
These are the **CPU fallback** numbers. The NPU numbers (expected multiples faster + far better
LLM quality with Phi-3.5/Llama-3.2) get measured in Phase 1b and pasted into `docs/PROPOSAL.md`.

## ⬜ Requires the Snapdragon HP PC (code + runbook ready)
- **Phase 0 gate** — `python scripts/verify_npu.py` confirming `QNNExecutionProvider`.
- **Phase 1b** — export AI Hub Whisper (ONNX+QNN) and the Genie LLM bundle; wire
  `asr.py::_transcribe_npu` and `llm.py::_generate_genie` (both raise a clear error off-device today).
- **NPU utilization meter**, streaming live captions, MeloTTS spoken output.
- **Build the `.exe`** (PyInstaller doesn't cross-compile; build on ARM64).

Full step-by-step: **`docs/DEVICE_SETUP.md`**.

## How to run what's built
```bash
pip install -r requirements-dev.txt
python -m app.server            # http://127.0.0.1:8000  (offline)
python -m engine.pipeline --text recordings/sample_meeting.txt --reply
pytest -q
```
