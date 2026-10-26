"""Phase 0 gate: confirm the Hexagon NPU is usable before building on it.

    python scripts/verify_npu.py                 # environment check
    python scripts/verify_npu.py models/foo.onnx # also run a model and confirm NPU placement
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_infer import has_npu, npu_report  # noqa: E402


def main() -> int:
    report = npu_report()
    print("SnapAssist NPU check")

    if not report["onnxruntime_available"]:
        print("  onnxruntime is not installed.")
        print("  Device:  pip install onnxruntime-qnn")
        print("  Dev box: pip install onnxruntime")
        return 1

    print(f"  onnxruntime {report['onnxruntime_version']}")
    print(f"  providers   {report['available_providers']}")

    if not has_npu():
        print("\n  QNNExecutionProvider unavailable — not a Snapdragon device, or onnxruntime-qnn "
              "is missing. CPU fallback works for development; the submission must run on the NPU.")
        return 1

    print("  QNNExecutionProvider available — the NPU can be targeted.")

    if len(sys.argv) < 2:
        print("\n  Pass a model path to confirm real execution:")
        print("    python scripts/fetch_models.py")
        print("    python scripts/verify_npu.py models/whisper_base_encoder.onnx")
        return 0

    from edge_infer import EdgeSession
    session = EdgeSession(sys.argv[1])
    print(f"\n  loaded {sys.argv[1]}  → provider {session.active_provider}")
    if session.on_npu:
        print("  Running on the Hexagon NPU. Phase 0 passed.")
        return 0
    print("  Fell back to CPU despite QNN being available — check op support / quantisation.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
