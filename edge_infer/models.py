"""The Qualcomm AI Hub models SnapAssist deploys, kept as data so the set is easy to audit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str          # local filename stem
    aihub_id: str     # qai_hub_models module id
    task: str         # asr | llm | tts | ocr
    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {
    "asr": ModelSpec(
        "whisper_base", "whisper_base_en", "asr",
        "Transcription. Use whisper_small for multilingual/Indic input.",
    ),
    "llm": ModelSpec(
        "phi_3_5_mini", "phi_3_5_mini_instruct", "llm",
        "Summaries, action items, drafts. Runs via Genie. Alt: llama_v3_2_3b_chat.",
    ),
    "tts": ModelSpec(
        "melotts", "melotts", "tts",
        "Optional spoken output (English + Indic voices).",
    ),
}


def get(task: str) -> ModelSpec:
    return REGISTRY[task]
