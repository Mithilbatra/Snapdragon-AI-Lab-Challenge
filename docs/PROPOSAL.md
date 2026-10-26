# SnapAssist

*A private, offline AI meeting assistant for Snapdragon-powered HP PCs.*
Snapdragon AI Lab — Build & Present Challenge.

## The problem
Meetings generate notes, decisions and follow-ups that people rarely capture well. The tools that
help — Otter, Fireflies, Copilot — stream your audio and notes to the cloud. That rules them out
for anyone handling confidential conversations (legal, medical, HR, finance) or working with poor
connectivity, and they handle Indian languages badly.

## The solution
SnapAssist records a meeting, transcribes it, extracts a summary with decisions and action items,
drafts the follow-up email, and answers questions about the discussion — in English and Indian
languages — with every model running on the device's **Hexagon NPU**. No audio, notes or drafts
leave the laptop. The whole thing is demonstrated in airplane mode.

## Why it has to run on Snapdragon / on-device
- **Privacy** — regulated or sensitive audio never leaves the machine.
- **Offline** — works with no connectivity, on the ~85 TOPS NPU.
- **Cost** — no per-minute cloud transcription or API fees.
- **Latency** — real-time transcription and summarisation on the NPU.

## How it's built
A reusable core, `edge_infer`, runs any exported model through ONNX Runtime's **QNN Execution
Provider** on the NPU, with a CPU fallback for development and a benchmark harness. On top of it,
the meeting pipeline chains Qualcomm AI Hub models:

| Stage | Model (AI Hub) | Runtime |
|-------|----------------|---------|
| Transcription | Whisper | ONNX + QNN EP |
| Summary / decisions / action items / drafts / Q&A | Phi-3.5-mini or Llama-3.2-3B | Genie (NPU) |
| Spoken read-back (optional) | MeloTTS | ONNX + QNN EP |

The app is a local FastAPI server and an offline web UI (no external dependencies), packaged as a
single Windows ARM64 executable. Architecture: `docs/architecture.svg`.

## What works today
The full pipeline runs offline: record/upload/paste → transcribe → summarise → draft → save. The
web UI shows a live compute badge, an offline indicator and per-request latency. CPU baseline on a
dev laptop: Whisper-tiny ~11–15 s for an 11 s clip; summary ~11–21 s with a 0.5 B dev model. The
on-device figures — expected several times faster, with far better summaries from Phi-3.5/Llama-3.2
via Genie — are measured with `edge_infer.benchmark` during device bring-up and inserted here.

## Innovation
Privacy-by-construction + Indic support + a reusable open-source NPU runtime — a combination no
shipping meeting assistant offers on this hardware.

## Impact
Professionals bound by confidentiality, and the many users in India on limited or costly
connectivity, get a capable meeting assistant that simply works, privately, with no subscription.

---

### Before submitting
- [ ] Run on the HP Snapdragon PC and paste the NPU-vs-CPU benchmark numbers above.
- [ ] Record the 3-minute demo video (script: `docs/DEMO_SCRIPT.md`) and link it here.
- [ ] Add the public code repository URL here.
