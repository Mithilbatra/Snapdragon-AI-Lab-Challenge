"""Whisper transcription.

The CPU backend uses transformers and runs anywhere, which is what we develop against. The NPU
backend feeds Whisper's encoder and decoder graphs through edge_infer.EdgeSession so they execute
on the Hexagon NPU; feature extraction and tokenisation stay on transformers (they're cheap and
already correct). Pick the backend with the ``backend`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import load_wav

_HF_ID = {"tiny": "openai/whisper-tiny", "base": "openai/whisper-base",
          "small": "openai/whisper-small"}


@dataclass
class Transcript:
    text: str
    language: str = "en"
    segments: list | None = None


class Whisper:
    def __init__(self, backend: str = "cpu", model_size: str = "tiny", models_dir: str = "models"):
        if backend not in ("cpu", "npu"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.model_size = model_size
        self.models_dir = models_dir
        self._pipe = None
        self._npu = None

    @property
    def on_npu(self) -> bool:
        return self.backend == "npu" and self._npu is not None and self._npu.on_npu

    def transcribe_file(self, wav_path: str, language: str = "en") -> Transcript:
        audio, rate = load_wav(wav_path)
        return self.transcribe(audio, rate, language)

    def transcribe(self, audio: np.ndarray, rate: int, language: str = "en") -> Transcript:
        if self.backend == "npu":
            return self._transcribe_npu(audio, rate, language)
        return self._transcribe_cpu(audio, rate, language)

    # CPU -------------------------------------------------------------------
    def _transcribe_cpu(self, audio: np.ndarray, rate: int, language: str) -> Transcript:
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=_HF_ID.get(self.model_size, self.model_size),
                chunk_length_s=30,
                return_timestamps=True,
            )
        gen = {"language": language, "task": "transcribe"} if language else {}
        out = self._pipe({"array": audio, "sampling_rate": rate}, generate_kwargs=gen)
        return Transcript(out["text"].strip(), language or "auto", out.get("chunks"))

    # NPU -------------------------------------------------------------------
    def _transcribe_npu(self, audio: np.ndarray, rate: int, language: str) -> Transcript:
        # Validate on the device: the encoder/decoder .onnx files must be exported from AI Hub
        # (see scripts/fetch_models.py) with the standard input_features / input_ids +
        # encoder_hidden_states signature. If your export splits the KV cache, match those names.
        enc, dec, processor = self._load_npu()

        features = processor(audio, sampling_rate=rate, return_tensors="np").input_features
        hidden = enc.run({"input_features": features.astype(np.float32)})[0]

        tok = processor.tokenizer
        ids = tok.convert_tokens_to_ids(
            ["<|startoftranscript|>", f"<|{language}|>", "<|transcribe|>", "<|notimestamps|>"]
        )
        eot = tok.eos_token_id
        for _ in range(224):  # Whisper's max decode length for a 30s window
            logits = dec.run({
                "input_ids": np.array([ids], dtype=np.int64),
                "encoder_hidden_states": hidden,
            })[0]
            nxt = int(logits[0, -1].argmax())
            if nxt == eot:
                break
            ids.append(nxt)

        text = tok.decode(ids, skip_special_tokens=True).strip()
        return Transcript(text, language)

    def _load_npu(self):
        if self._npu is None:
            from transformers import WhisperProcessor

            from edge_infer import EdgeSession
            stem = f"{self.models_dir}/whisper_{self.model_size}"
            self._npu = EdgeSession(f"{stem}_encoder.onnx")
            self._npu_decoder = EdgeSession(f"{stem}_decoder.onnx")
            self._processor = WhisperProcessor.from_pretrained(
                _HF_ID.get(self.model_size, self.model_size)
            )
        return self._npu, self._npu_decoder, self._processor


class StreamingTranscriber:
    """Rolling-window transcription off the microphone, for live captions."""

    def __init__(self, whisper: Whisper, window_s: float = 6.0, stride_s: float = 2.0,
                 rate: int = 16000):
        self.whisper = whisper
        self.window = int(window_s * rate)
        self.stride = int(stride_s * rate)
        self.rate = rate

    def stream(self, on_partial):
        """Capture from the default input device and call ``on_partial(text)`` per window."""
        import sounddevice as sd

        buffer = np.zeros(0, dtype=np.float32)
        with sd.InputStream(samplerate=self.rate, channels=1, dtype="float32") as mic:
            while True:
                chunk, _ = mic.read(self.stride)
                buffer = np.concatenate([buffer, chunk[:, 0]])[-self.window:]
                if len(buffer) >= self.stride:
                    on_partial(self.whisper.transcribe(buffer, self.rate).text)
