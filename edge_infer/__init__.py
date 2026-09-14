"""edge_infer — run ONNX / Qualcomm AI Hub models on the Snapdragon Hexagon NPU.

A thin, reusable runtime that:
  * creates ONNX Runtime sessions on the QNN Execution Provider (Hexagon NPU),
  * transparently falls back to CPU on non-Snapdragon dev machines,
  * reports which device actually executed the graph, and
  * benchmarks NPU vs CPU (latency / throughput).

This package is intentionally app-agnostic so it can be reused across projects and
open-sourced on its own.
"""

from .runtime import (
    EdgeSession,
    available_providers,
    has_npu,
    npu_report,
)

__all__ = ["EdgeSession", "available_providers", "has_npu", "npu_report"]
__version__ = "0.1.0"
