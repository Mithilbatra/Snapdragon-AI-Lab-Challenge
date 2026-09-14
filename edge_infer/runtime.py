"""Core runtime: build ONNX Runtime sessions on the Hexagon NPU (QNN EP), with CPU fallback.

The QNN Execution Provider is what routes an ONNX graph onto the Snapdragon Hexagon NPU.
On a Snapdragon Windows ARM64 machine with `onnxruntime-qnn` installed, "QNNExecutionProvider"
appears in ort.get_available_providers(). On any other machine it won't, and we fall back to CPU
so the code is still runnable during development.

Key design points:
  * `EdgeSession` wraps ort.InferenceSession and records the provider that was actually used.
  * QNN options (backend .dll, HTP performance mode, context-binary caching) are set here in one
    place so callers never have to know about them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:  # onnxruntime absent (e.g. fresh dev box) — degrade gracefully.
    ort = None
    ORT_AVAILABLE = False

_ORT_HINT = ("onnxruntime is not installed. On Snapdragon: `pip install onnxruntime-qnn`; "
             "elsewhere: `pip install onnxruntime`.")

QNN_EP = "QNNExecutionProvider"
CPU_EP = "CPUExecutionProvider"


def available_providers() -> list[str]:
    """All execution providers this ONNX Runtime build exposes (empty if ORT absent)."""
    return list(ort.get_available_providers()) if ORT_AVAILABLE else []


def has_npu() -> bool:
    """True if the QNN EP (Hexagon NPU) is available in this environment."""
    return QNN_EP in available_providers()


def npu_report() -> dict[str, Any]:
    """Human-readable snapshot for `verify_npu.py` and the app's NPU badge."""
    return {
        "onnxruntime_available": ORT_AVAILABLE,
        "onnxruntime_version": ort.__version__ if ORT_AVAILABLE else None,
        "available_providers": available_providers(),
        "npu_available": has_npu(),
        "device": ort.get_device() if ORT_AVAILABLE else "none",
    }


def _default_qnn_options() -> dict[str, str]:
    """QNN EP provider options tuned for laptop LLM/ASR workloads.

    `backend_path` selects the HTP (Hexagon Tensor Processor) backend.
    `htp_performance_mode=burst` favors latency for interactive use.
    Context-binary caching avoids recompiling large graphs (LLMs, Whisper) on every launch.
    """
    return {
        "backend_path": "QnnHtp.dll",          # HTP = the NPU backend on Snapdragon X
        "htp_performance_mode": "burst",
        "enable_htp_fp16_precision": "1",
        # Cache the compiled QNN context so >1GB models don't recompile each run:
        "htp_graph_finalization_optimization_mode": "3",
    }


@dataclass
class EdgeSession:
    """An ONNX Runtime session pinned to the NPU when possible, else CPU.

    Example:
        sess = EdgeSession("models/whisper_base_encoder.onnx")
        print(sess.active_provider)          # "QNNExecutionProvider" on Snapdragon
        out = sess.run({"audio": mel})
    """

    model_path: str
    prefer_npu: bool = True
    qnn_options: dict[str, str] = field(default_factory=_default_qnn_options)
    context_cache_dir: str | None = "models/.qnn_cache"

    _session: "ort.InferenceSession" = field(init=False, repr=False)
    active_provider: str = field(init=False)

    def __post_init__(self) -> None:
        if not ORT_AVAILABLE:
            raise RuntimeError(_ORT_HINT)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                "Run `python scripts/fetch_models.py` to download AI Hub artifacts."
            )

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers: list = []
        if self.prefer_npu and has_npu():
            opts = dict(self.qnn_options)
            if self.context_cache_dir:
                os.makedirs(self.context_cache_dir, exist_ok=True)
                # Persist compiled context binaries between runs:
                so.add_session_config_entry("ep.context_enable", "1")
                so.add_session_config_entry("ep.context_file_path",
                                            os.path.join(self.context_cache_dir,
                                                         os.path.basename(self.model_path) + ".bin"))
            providers.append((QNN_EP, opts))
        providers.append(CPU_EP)  # always keep CPU as the tail fallback

        self._session = ort.InferenceSession(self.model_path, sess_options=so, providers=providers)
        # The first provider ORT actually placed nodes on:
        self.active_provider = self._session.get_providers()[0]

    @property
    def on_npu(self) -> bool:
        return self.active_provider == QNN_EP

    def input_names(self) -> list[str]:
        return [i.name for i in self._session.get_inputs()]

    def output_names(self) -> list[str]:
        return [o.name for o in self._session.get_outputs()]

    def run(self, feeds: dict[str, Any], output_names: list[str] | None = None):
        """Run inference. `feeds` maps input name -> numpy array."""
        return self._session.run(output_names or self.output_names(), feeds)
