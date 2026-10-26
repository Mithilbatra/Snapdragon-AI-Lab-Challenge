"""Benchmark a model on the NPU vs CPU and print the table for the proposal.

Run on the Snapdragon device once the ONNX models are exported:

    python scripts/benchmark.py models/whisper_tiny_encoder.onnx
    python scripts/benchmark.py models/whisper_tiny_encoder.onnx --shape 1,80,3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_infer.benchmark import compare_npu_vs_cpu, print_comparison  # noqa: E402
from edge_infer.runtime import EdgeSession  # noqa: E402


def _input_shape(model_path: str, override: str | None) -> tuple[str, tuple[int, ...]]:
    if override:
        session = EdgeSession(model_path, prefer_npu=False)
        return session.input_names()[0], tuple(int(x) for x in override.split(","))
    session = EdgeSession(model_path, prefer_npu=False)
    spec = session._session.get_inputs()[0]
    # Replace dynamic/symbolic dims with 1 so we can generate a concrete tensor.
    shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in spec.shape)
    return spec.name, shape


def main() -> int:
    parser = argparse.ArgumentParser(description="NPU vs CPU benchmark")
    parser.add_argument("model", help="path to an .onnx model")
    parser.add_argument("--shape", help="comma-separated input shape, e.g. 1,80,3000")
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    name, shape = _input_shape(args.model, args.shape)
    print(f"input {name} {shape}")

    def make_feeds():
        return {name: np.random.randn(*shape).astype(np.float32)}

    results = compare_npu_vs_cpu(args.model, make_feeds, runs=args.runs)
    print_comparison(results)
    if len(results) == 1:
        print("\nOnly one provider ran — QNN unavailable (not a Snapdragon device?).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
