"""ONNX Runtime sessions that target the Snapdragon Hexagon NPU, falling back to CPU."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None
    ORT_AVAILABLE = False

QNN_EP = "QNNExecutionProvider"
CPU_EP = "CPUExecutionProvider"

_INSTALL_HINT = (
    "onnxruntime is not installed. On Snapdragon run `pip install onnxruntime-qnn`, "
    "elsewhere `pip install onnxruntime`."
)


def available_providers() -> list[str]:
    return list(ort.get_available_providers()) if ORT_AVAILABLE else []


def has_npu() -> bool:
    return QNN_EP in available_providers()


def npu_report() -> dict[str, Any]:
    """Snapshot used by verify_npu.py and the app's status badge."""
    return {
        "onnxruntime_available": ORT_AVAILABLE,
        "onnxruntime_version": ort.__version__ if ORT_AVAILABLE else None,
        "available_providers": available_providers(),
        "npu_available": has_npu(),
        "device": ort.get_device() if ORT_AVAILABLE else "none",
    }


def _qnn_provider_options() -> dict[str, str]:
    # HTP is the NPU backend on Snapdragon X. Burst mode favours interactive latency;
    # fp16 keeps accuracy acceptable while staying on the NPU.
    return {
        "backend_path": "QnnHtp.dll",
        "htp_performance_mode": "burst",
        "enable_htp_fp16_precision": "1",
        "htp_graph_finalization_optimization_mode": "3",
    }


@dataclass
class EdgeSession:
    """A model session pinned to the NPU when it's available, otherwise CPU.

    >>> s = EdgeSession("models/whisper_base_encoder.onnx")
    >>> s.on_npu
    True
    >>> out = s.run({"audio": mel})
    """

    model_path: str
    prefer_npu: bool = True
    context_cache_dir: str | None = "models/.qnn_cache"
    qnn_options: dict[str, str] = field(default_factory=_qnn_provider_options)

    def __post_init__(self) -> None:
        if not ORT_AVAILABLE:
            raise RuntimeError(_INSTALL_HINT)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"{self.model_path} not found. Run scripts/fetch_models.py to download it."
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers: list = []
        if self.prefer_npu and has_npu():
            if self.context_cache_dir:
                os.makedirs(self.context_cache_dir, exist_ok=True)
                cache = os.path.join(self.context_cache_dir,
                                     os.path.basename(self.model_path) + ".bin")
                opts.add_session_config_entry("ep.context_enable", "1")
                opts.add_session_config_entry("ep.context_file_path", cache)
            providers.append((QNN_EP, self.qnn_options))
        providers.append(CPU_EP)

        self._session = ort.InferenceSession(self.model_path, opts, providers=providers)
        self.active_provider = self._session.get_providers()[0]

    @property
    def on_npu(self) -> bool:
        return self.active_provider == QNN_EP

    def input_names(self) -> list[str]:
        return [i.name for i in self._session.get_inputs()]

    def output_names(self) -> list[str]:
        return [o.name for o in self._session.get_outputs()]

    def run(self, feeds: dict[str, Any], outputs: list[str] | None = None):
        return self._session.run(outputs, feeds)
