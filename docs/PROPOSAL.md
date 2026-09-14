# SnapAssist — Proposal (Snapdragon AI Lab Build & Present Challenge)

> 1-page submission outline. Fill bracketed parts after Phase 1 benchmarks.

## Problem
Professionals, students, and teams drown in meetings. Existing assistants (Otter, Fireflies,
Copilot) send audio and notes to the cloud — a non-starter for legal, medical, HR, and any
privacy-sensitive or low-connectivity setting, and they handle Indian languages poorly.

## Solution
**SnapAssist** — a private, offline AI meeting & notes assistant for Snapdragon HP PCs. It
records, transcribes, summarizes, extracts action items, and drafts follow-up emails — in
English and Indian languages — with **everything running on the Hexagon NPU**. No internet, no
subscription, no data leaving the device. Demonstrated live in airplane mode.

## Why Snapdragon / on-device (not cloud)
- **Privacy:** audio + notes never leave the laptop.
- **Offline:** works with zero connectivity (rural, travel, secure facilities).
- **Cost:** no per-minute cloud/API fees.
- **Latency:** real-time captions from the 80–85 TOPS NPU.

## Technical implementation
- Reusable `edge_infer` core: ONNX Runtime + **QNN Execution Provider** → Hexagon NPU, with CPU
  fallback and a benchmark harness.
- Pipeline (all **Qualcomm AI Hub** models): **Whisper** (ASR) → **Phi-3.5-mini / Llama-3.2-3B**
  via **Genie** (summary/drafts/Q&A) → **MeloTTS** (optional spoken output).
- Working prototype today: full offline pipeline (record → transcribe → summarize → draft) in a
  local web app; CPU baseline measured (whisper-tiny ~11s for an 11s clip; summary ~12–21s).
- On-device target (Phase 1b, HP Snapdragon PC): **[X]× lower latency on NPU vs CPU**,
  **[Y] tokens/sec** via Genie — to be measured with `edge_infer.benchmark` and inserted here.

## Deployment & accessibility
- Single Windows ARM64 installer; runs offline out of the box.
- Spoken responses + Indic language support widen access.

## Innovation
- Privacy-first + Indic-first + a reusable open-source NPU runtime — a combination no shipping
  assistant offers on this hardware.

## Impact
- For privacy-bound professionals and the millions in India with limited/expensive connectivity.

## Architecture diagram
`[insert docs/architecture.png — see README diagram]`

## Links
- Demo video: `[link]`   •   Code: `[repo]`   •   `edge_infer` (standalone): `[repo]`
