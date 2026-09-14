"""Text-to-speech: MeloTTS on the NPU (optional spoken responses / accessibility).

Runs on EdgeSession like the ASR models. Phase 3 / nice-to-have.
"""

from __future__ import annotations

from edge_infer import EdgeSession
from edge_infer.models import get


class MeloTTS:
    def __init__(self, models_dir: str = "models"):
        spec = get("tts")
        self.session = EdgeSession(f"{models_dir}/{spec.key}.onnx")

    @property
    def on_npu(self) -> bool:
        return self.session.on_npu

    def speak(self, text: str, language: str = "en", out_wav: str = "out.wav") -> str:
        """TODO: text -> phonemes -> MeloTTS.run -> waveform -> write out_wav."""
        raise NotImplementedError("Phase 3: wire MeloTTS pre/post-processing around EdgeSession.")
