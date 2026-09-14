"""Registry of the Qualcomm AI Hub models SnapAssist uses.

Names map to `qai_hub_models` reference implementations. `scripts/fetch_models.py` reads this
registry to download / compile the ONNX (+ QNN context) artifacts into `models/`.

Keeping this as data (not scattered strings) makes the model set auditable for the proposal
and easy to swap (e.g. Whisper-Base -> Distil-Whisper, Phi-3.5 -> Llama-3.2-3B).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str            # local name used by the app
    aihub_id: str       # qai_hub_models module / model id
    task: str           # asr | llm | tts | ocr
    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {
    "asr": ModelSpec(
        key="whisper_base",
        aihub_id="whisper_base_en",         # multilingual variant for Indic: whisper_small
        task="asr",
        notes="Streaming transcription. Swap to distil_whisper for lower latency.",
    ),
    "llm": ModelSpec(
        key="phi_3_5_mini",
        aihub_id="phi_3_5_mini_instruct",   # alt: llama_v3_2_3b_chat
        task="llm",
        notes="Summaries, action items, draft replies, Q&A. Runs via Genie on QNN.",
    ),
    "tts": ModelSpec(
        key="melotts",
        aihub_id="melotts",                 # EN + Indic voices
        task="tts",
        notes="Optional spoken responses / accessibility.",
    ),
    # Optional Phase 3+:
    # "ocr": ModelSpec("trocr", "trocr", "ocr", "scanned meeting docs / whiteboards"),
}


def get(task: str) -> ModelSpec:
    return REGISTRY[task]
