"""Optional spoken output via MeloTTS on the NPU. Used for accessibility read-back."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from edge_infer import EdgeSession
from edge_infer.models import get


class MeloTTS:
    def __init__(self, models_dir: str = "models"):
        self.session = EdgeSession(f"{models_dir}/{get('tts').key}.onnx")

    @property
    def on_npu(self) -> bool:
        return self.session.on_npu

    def speak(self, text: str, out_wav: str, rate: int = 24000) -> str:
        """Synthesise ``text`` to ``out_wav``. Phonemisation matches the MeloTTS export."""
        from qai_hub_models.models.melotts.app import tokenize  # phoneme front-end
        tokens = tokenize(text)
        audio = self.session.run({"tokens": np.array([tokens], dtype=np.int64)})[0]
        sf.write(out_wav, audio.squeeze().astype(np.float32), rate)
        return out_wav
