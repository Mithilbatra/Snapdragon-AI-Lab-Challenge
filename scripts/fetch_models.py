"""Export the Qualcomm AI Hub models into models/.

Whisper and MeloTTS export to ONNX (run on the NPU via the QNN EP). The LLM ships as a Genie
bundle, which follows a different flow — see docs/DEVICE_SETUP.md. Requires a configured
`qai-hub` account and network access, so this is normally run on the device.

    python scripts/fetch_models.py            # all ONNX models
    python scripts/fetch_models.py asr        # just one task
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_infer.models import REGISTRY  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def export_onnx(aihub_id: str) -> int:
    cmd = [sys.executable, "-m", f"qai_hub_models.models.{aihub_id}.export",
           "--target-runtime", "onnx", "--output-dir", str(MODELS)]
    print("→", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    tasks = sys.argv[1:] or list(REGISTRY)

    try:
        import qai_hub_models  # noqa: F401
    except ImportError:
        print("qai-hub-models is not installed. On the device: pip install qai-hub-models, "
              "then `qai-hub configure` with your API token.")
        return 1

    failures = 0
    for task in tasks:
        spec = REGISTRY[task]
        if spec.task == "llm":
            print(f"[{task}] {spec.key}: LLM uses a Genie bundle, not ONNX export. "
                  "Follow docs/DEVICE_SETUP.md → Phase 1b step 2.")
            continue
        if export_onnx(spec.aihub_id) != 0:
            print(f"[{task}] export failed for {spec.aihub_id}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
