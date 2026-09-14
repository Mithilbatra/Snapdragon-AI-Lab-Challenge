# Judging cheat sheet

What to show and say for each scoring criterion. Have the app running on the HP Snapdragon PC in
airplane mode before you start.

## Technical Implementation
- **Show:** `python scripts/verify_npu.py` printing `QNNExecutionProvider`, then the app's compute
  badge reading **Hexagon NPU**.
- **Say:** "Whisper, Phi-3.5 and MeloTTS — all Qualcomm AI Hub models — run on the NPU through the
  QNN Execution Provider and Genie. The reusable `edge_infer` layer handles placement, caching and
  the CPU fallback."
- **Back-up:** `python scripts/benchmark.py models/whisper_tiny_encoder.onnx` → the NPU-vs-CPU table.

## Application Use Case & Innovation
- **Show:** record a short meeting → summary, decisions, action items → drafted follow-up.
- **Say:** "It's a real daily tool, but the innovation is what only on-device allows — privacy,
  offline, no subscription, and Indian languages. Copilot can't do this offline."

## Deployment & Accessibility
- **Show:** the packaged `SnapAssist.exe` launching straight into the browser; the airplane-mode
  icon on the taskbar; switch **Language → Hindi** and summarise again.
- **Say:** "One executable, no internet, no account. Usable where cloud tools can't go — regulated
  data, or low-connectivity regions. Spoken read-back adds accessibility."

## Presentation & Documentation
- **Show:** the architecture diagram and the README/DEVICE_SETUP docs.
- **Say:** "Reproducible from a clean machine: install, verify the NPU, fetch models, run."

## The one question to be ready for
**"Why not just use the cloud / Copilot?"** → Privacy (audio never leaves the device), offline (no
connectivity needed), cost (no per-minute fees), latency (real-time on the NPU), plus Indic support.

## Final checklist before judging
- [ ] `verify_npu.py` passes on the device
- [ ] Models exported (`fetch_models.py`) and `SNAP_ASR_BACKEND=npu SNAP_LLM_BACKEND=genie`
- [ ] Benchmark numbers pasted into `docs/PROPOSAL.md`
- [ ] Demo recorded in airplane mode
- [ ] `.exe` built and tested offline
