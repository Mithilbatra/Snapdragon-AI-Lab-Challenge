# SnapAssist — 3-minute demo script (Build & Present)

Goal: make the judges *feel* the three differentiators — offline, private, on-NPU — in under
three minutes, with one undeniable "wow".

## Setup (before you present)
- Device: HP OmniBook (Snapdragon X/X2), backends set to NPU + Genie.
- `python -m app.server` running; browser at `http://127.0.0.1:8000`.
- A ~2-minute meeting recording ready (or record live).
- **Wi-Fi + Bluetooth OFF (airplane mode) — show the taskbar so the audience sees it.**

## Beat 1 — "Everything you'll see happens on this laptop, with no internet." (20s)
- Point to the airplane-mode icon and the UI badge: **On-device · OFFLINE ✓** and **Hexagon NPU**.
- One line: "No audio, no notes, no email ever leaves this machine."

## Beat 2 — Record → transcript (40s)
- Click **Record**, play/speak ~30–60s of a meeting, click **Stop**, **Transcribe audio**.
- Call out the latency chip: "Transcribed on the Hexagon NPU in **[X]s**, offline."

## Beat 3 — Summary + action items (40s)
- Click **Summarize**. Read out the auto-extracted **decisions** and **action items**.
- "This is Phi-3.5 running on the NPU via Genie — still no cloud."

## Beat 4 — The Indic "wow" (30s)
- Switch **Language → Hindi**, click **Summarize** again.
- Show the same meeting summarized in Hindi. "Copilot can't do this offline; we can."

## Beat 5 — Draft email + privacy close (30s)
- Click **Draft follow-up email** — a ready-to-send follow-up appears.
- Close: "Private by construction, works with zero connectivity, powered end-to-end by Qualcomm
  AI Hub models on Snapdragon. That's SnapAssist."

## Backup (if the mic misbehaves)
- Use **Paste transcript** with a prepared transcript, or **Upload WAV** with the recorded file.
  Every path exercises the same on-device pipeline.

## What to say if a judge asks "why not just use the cloud / Copilot?"
- Privacy (regulated audio can't leave the device), offline (no connectivity needed),
  cost (no per-minute fees), and latency (NPU real-time) — with Indic support Copilot lacks.
