"""Audio loading utilities — dependency-light so the dev pipeline needs no ffmpeg.

Reads 16-bit PCM WAV via the stdlib and returns a float32 mono array at the target rate.
Enough for the JFK/LibriSpeech-style samples and typical meeting recordings exported to WAV.
"""

from __future__ import annotations

import wave

import numpy as np


def load_wav(path: str, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Load a WAV file -> (float32 mono samples in [-1, 1], sample_rate).

    Downmixes to mono and linearly resamples to `target_rate` if needed.
    """
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())

    if sampwidth != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV supported (got {sampwidth * 8}-bit). "
                         "Export/convert to 16-bit PCM WAV.")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if rate != target_rate:
        audio = _resample_linear(audio, rate, target_rate)
        rate = target_rate

    return audio, rate


def _resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Simple linear resampling. Fine for speech; swap for a polyphase filter if quality matters."""
    if src_rate == dst_rate:
        return x
    duration = x.shape[0] / src_rate
    dst_n = int(round(duration * dst_rate))
    src_t = np.linspace(0, duration, num=x.shape[0], endpoint=False)
    dst_t = np.linspace(0, duration, num=dst_n, endpoint=False)
    return np.interp(dst_t, src_t, x).astype(np.float32)
