# Status: built vs. needs the device

SnapAssist was developed on a Mac, which has no Hexagon NPU. Everything that can be run without
Snapdragon hardware is implemented and tested; the NPU/Genie code is written against Qualcomm's
documented interfaces but can only be **validated on the HP Snapdragon PC**.

## Runs and verified here (offline, CPU)
- **`edge_infer`** — QNN-EP session factory with NPU detection, CPU fallback and context caching;
  benchmark harness; graceful when `onnxruntime`/QNN is absent.
- **Engine** — Whisper ASR (CPU) and LLM summarize/draft/answer (CPU) with robust JSON parsing;
  end-to-end pipeline and CLI.
- **App** — FastAPI server + offline web UI: record/upload/paste → transcribe → summarise
  (decisions + action items) → draft email → ask → save to SQLite. Live compute badge, offline
  indicator, rolling per-request latency. Backend errors surface as 503s, not crashes.
- **Tests** — 13 passing: NPU report, JSON extraction (prose, code fences, nested braces, bad
  shapes), WAV loading, status endpoint, pipeline with a fake LLM, backend validation.
- **Packaging & docs** — architecture diagram, demo script, `SnapAssist.spec` + `build_exe.py` +
  `launcher.py`, device runbook.

### CPU baseline (dev laptop, warm, offline)
| Stage | Model | Input | Latency |
|-------|-------|-------|---------|
| ASR | whisper-tiny | 11 s audio | ~11–15 s |
| Summarize | Qwen2.5-0.5B | ~550-char transcript | ~11–21 s |

These are CPU-fallback numbers; the NPU numbers get measured in device bring-up.

## Written, needs the Snapdragon PC to validate
- **NPU ASR** (`asr.py::_transcribe_npu`) — runs Whisper's encoder/decoder through `EdgeSession`;
  needs the exported `whisper_*_encoder.onnx` / `_decoder.onnx`. Confirm the exported input names
  match (standard `input_features` / `input_ids` + `encoder_hidden_states`).
- **Genie LLM** (`llm.py::_generate_genie`) — calls `genie-t2t-run` against the Genie bundle in
  `models/genie/`.
- **MeloTTS** (`tts.py`), **streaming captions** (`asr.py::StreamingTranscriber`, needs a mic +
  `sounddevice`), and the actual `.exe` (PyInstaller builds on ARM64, not here).
- **Phase 0 gate** — `python scripts/verify_npu.py` reporting `QNNExecutionProvider`.

Step-by-step: [DEVICE_SETUP.md](DEVICE_SETUP.md).

## Honest caveat on the dev model
The 0.5 B dev LLM produces shallow summaries and does not reliably reply in Hindi even though the
`language` parameter flows end-to-end. Phi-3.5-mini / Llama-3.2-3B via Genie fix both on-device.
