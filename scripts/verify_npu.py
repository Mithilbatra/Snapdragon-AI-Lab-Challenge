"""Phase 0 gate: prove the Hexagon NPU is actually usable before building anything else.

This is the single highest-risk assumption in the whole project. If QNN EP is not available
or a tiny model won't run on it, stop and fix the environment (onnxruntime-qnn install,
QAIRT SDK, ARM64 Python) before writing another line of app code.

Usage:
    python scripts/verify_npu.py                 # environment check only
    python scripts/verify_npu.py path/to.onnx    # also run a real model on the NPU
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_infer import available_providers, has_npu, npu_report  # noqa: E402


def main() -> int:
    print("=" * 60)
    print("SnapAssist — NPU verification (Phase 0)")
    print("=" * 60)

    report = npu_report()
    if not report["onnxruntime_available"]:
        print("onnxruntime         : NOT INSTALLED")
        print("   On the Snapdragon HP PC:  pip install onnxruntime-qnn")
        print("   On a dev machine:         pip install onnxruntime  (CPU only)")
        return 1
    print(f"onnxruntime version : {report['onnxruntime_version']}")
    print(f"ort.get_device()    : {report['device']}")
    print(f"available providers : {report['available_providers']}")
    print()

    if not has_npu():
        print("❌ QNNExecutionProvider NOT available.")
        print("   You are likely on a non-Snapdragon machine, or `onnxruntime-qnn` is not installed.")
        print("   On the Snapdragon HP PC:  pip install onnxruntime-qnn")
        print("   (CPU fallback still works for development, but the submission must run on NPU.)")
        return 1

    print("✅ QNNExecutionProvider is available — the Hexagon NPU can be targeted.")

    model_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if not model_arg:
        print("\nNo model given — environment check only. To validate real execution:")
        print("   python scripts/fetch_models.py   # download a model")
        print("   python scripts/verify_npu.py models/<something>.onnx")
        return 0

    from edge_infer import EdgeSession  # imported late so the env check runs even without a model

    print(f"\nLoading {model_arg} on the NPU ...")
    sess = EdgeSession(model_arg, prefer_npu=True)
    print(f"active provider     : {sess.active_provider}")
    print(f"inputs              : {sess.input_names()}")
    print(f"outputs             : {sess.output_names()}")
    if sess.on_npu:
        print("\n✅ Model is running on the Hexagon NPU. Phase 0 PASSED — you may build.")
        return 0
    print("\n⚠️  Model fell back to CPU despite QNN being available — investigate op support / "
          "quantization before proceeding.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
