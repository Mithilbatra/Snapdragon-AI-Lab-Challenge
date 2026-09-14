"""Speech-to-text: Whisper.

Two backends behind ONE interface:
  * backend="cpu" — reference implementation via HuggingFace transformers. Runs on any dev
    machine (Mac/x64) so the whole pipeline is testable today.
  * backend="npu" — Whisper ONNX on the Hexagon NPU via edge_infer.EdgeSession. Used on the
    Snapdragon HP PC. Wire in Phase 1b once AI Hub artifacts are exported to models/.

Switching backends is a one-liner; nothing else in the app changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .audio import load_wav


@dataclass
class Transcript:
    text: str
    language: str = "en"
    segments: list | None = None


class Whisper:
    def __init__(self, backend: str = "cpu", model_size: str = "tiny", models_dir: str = "models"):
        self.backend = backend
        self.model_size = model_size
        self.models_dir = models_dir
        self._impl = None  # lazy-loaded

    # --- public API -------------------------------------------------------
    @property
    def on_npu(self) -> bool:
        return self.backend == "npu" and getattr(self._impl, "on_npu", False)

    def transcribe_file(self, wav_path: str, language: str = "en") -> Transcript:
        audio, rate = load_wav(wav_path, target_rate=16000)
        if self.backend == "npu":
            return self._transcribe_npu(audio, rate, language)
        return self._transcribe_cpu(audio, rate, language)

    # --- CPU reference backend (transformers) -----------------------------
    def _ensure_cpu(self):
        if self._impl is not None:
            return
        from transformers import pipeline  # imported lazily to keep startup fast
        model_id = f"openai/whisper-{self.model_size}"
        self._impl = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            chunk_length_s=30,        # enables >30s clips (meetings) via chunking
            return_timestamps=True,
        )

    def _transcribe_cpu(self, audio, rate, language) -> Transcript:
        self._ensure_cpu()
        gen_kwargs = {"language": language, "task": "transcribe"} if language else {}
        out = self._impl({"array": audio, "sampling_rate": rate}, generate_kwargs=gen_kwargs)
        segments = out.get("chunks") if isinstance(out, dict) else None
        return Transcript(text=out["text"].strip(), language=language or "auto", segments=segments)

    # --- NPU backend (edge_infer) -----------------------------------------
    def _transcribe_npu(self, audio, rate, language) -> Transcript:
        # Phase 1b (on device): build log-mel -> EdgeSession(encoder) -> EdgeSession(decoder) decode.
        # Reuse qai_hub_models Whisper pre/post-processing; only the two heavy graphs go on the NPU.
        raise NotImplementedError(
            "NPU backend: wire EdgeSession encoder/decoder on the Snapdragon device. "
            "Use backend='cpu' for development."
        )


class StreamingTranscriber:
    """Phase 3: sliding-window mic capture -> partial transcripts for live captions."""

    def __init__(self, whisper: Whisper, window_s: float = 5.0, stride_s: float = 1.0):
        self.whisper = whisper
        self.window_s = window_s
        self.stride_s = stride_s

    def stream(self):
        raise NotImplementedError("Phase 3: sounddevice callback -> rolling buffer -> partials.")
