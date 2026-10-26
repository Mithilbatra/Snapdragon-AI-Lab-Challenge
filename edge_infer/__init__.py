"""Run ONNX / Qualcomm AI Hub models on the Hexagon NPU, with a CPU fallback for development."""

from .runtime import EdgeSession, available_providers, has_npu, npu_report

__all__ = ["EdgeSession", "available_providers", "has_npu", "npu_report"]
__version__ = "0.1.0"
