# Device runbook — Snapdragon HP PC (Phase 0 + Phase 1b)

Everything in this repo runs today on a dev machine with CPU backends. This file is the
step-by-step for moving inference onto the **Hexagon NPU** on the HP OmniBook (Snapdragon X / X2).
Do these **on the device**, in order.

## Phase 0 — environment & proof-of-NPU (hard gate)

1. Install **ARM64** Python 3.11+ (not the x64 build) from python.org.
2. Create a venv and install the device runtime:
   ```powershell
   python -m venv .venv; .venv\Scripts\activate
   pip install -r requirements.txt          # pulls onnxruntime-qnn on ARM64
   ```
3. Install the **QAIRT SDK** (Qualcomm AI Runtime) and confirm `QnnHtp.dll` is on PATH.
4. Create a free **Qualcomm AI Hub** account and `qai-hub configure` with your API token.
5. **Gate:** prove the NPU is usable.
   ```powershell
   python scripts\verify_npu.py            # must list QNNExecutionProvider
   ```
   Then run a real model on it (after Phase 1b step 1):
   ```powershell
   python scripts\verify_npu.py models\whisper_base_encoder.onnx   # expect "running on the Hexagon NPU"
   ```
   Do not proceed until this passes.

## Phase 1b — put the models on the NPU

### 1. Whisper (ASR) — ONNX + QNN EP
- Install the model deps and export via AI Hub:
  ```bash
  pip install "qai-hub-models[whisper-base]"
  # precompiled_qnn_onnx is loadable by ONNX Runtime's QNN EP (plain float onnx isn't offered)
  python -m qai_hub_models.models.whisper_base.export --device "Snapdragon X Elite CRD" --target-runtime precompiled_qnn_onnx
  ```
- Wire `engine/asr.py::Whisper._transcribe_npu`: log-mel front-end → `EdgeSession(encoder)` →
  greedy/beam decode with `EdgeSession(decoder)`. Reuse `qai_hub_models` Whisper pre/post-processing;
  only the two heavy graphs run on the NPU.
- Run: `python -m engine.pipeline recordings\meeting.wav --asr-backend npu`

### 2. LLM (Phi-3.5-mini / Llama-3.2-3B) — Genie
- Produce the Genie on-device bundle (QNN context binaries + tokenizer + genie config) per the
  AI Hub "LLM on-device with Genie" guide; place it under `models/genie/`.
- Wire `engine/llm.py::LocalLLM._generate_genie` to call the Genie generate endpoint (Python
  binding or `genie-t2t-run`), streaming tokens back.
- Run: `python -m engine.pipeline --text recordings\sample_meeting.txt --llm-backend genie`

### 3. (Optional) MeloTTS + TrOCR
- Same EdgeSession pattern (`engine/tts.py`, `engine/ocr.py`).

## Phase 1b — capture benchmarks (for the proposal)
```powershell
python -c "from edge_infer.benchmark import compare_npu_vs_cpu, print_comparison; ..."
```
Record NPU vs CPU latency + tokens/sec and paste into `docs/PROPOSAL.md`.

## Run the app on-device
```powershell
$env:SNAP_ASR_BACKEND="npu"; $env:SNAP_LLM_BACKEND="genie"
python -m app.server        # open http://127.0.0.1:8000, turn on airplane mode, demo
```
The compute badge in the UI flips to **“Hexagon NPU · Genie”** automatically.
