"""Download / export the Qualcomm AI Hub models declared in edge_infer/models.py into models/.

Two paths, depending on how you deploy each model:
  1) ONNX for QNN EP (Whisper, MeloTTS): export via qai_hub_models, then run with EdgeSession.
  2) Genie LLM (Phi-3.5 / Llama-3.2): produce QNN context binaries + a Genie bundle.

This is a scaffold — wire the exact export calls once you have a Qualcomm AI Hub account and the
QAIRT SDK. See: https://github.com/quic/ai-hub-models and the AI Hub compile docs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_infer.models import REGISTRY  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    print("Models to fetch (from edge_infer/models.py):")
    for task, spec in REGISTRY.items():
        print(f"  [{task:4}] {spec.key:16} <- aihub:{spec.aihub_id}  ({spec.notes})")

    print(f"\nTarget dir: {MODELS_DIR}")
    print(
        "\nTODO — implement per-model export. Typical flow with qai_hub_models:\n"
        "    from qai_hub_models.models.whisper_base_en import Model\n"
        "    m = Model.from_pretrained()\n"
        "    # compile to ONNX + QNN context via qai_hub.submit_compile_job(...)\n"
        "    # download the .onnx / context .bin into models/\n"
        "\nFor the LLM, follow the Genie on-device bundle instructions from AI Hub.\n"
        "Until wired, drop a small pre-exported .onnx in models/ and test with verify_npu.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
