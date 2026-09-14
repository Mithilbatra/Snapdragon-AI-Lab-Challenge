"""WAV loading without an ffmpeg dependency, so the app stays self-contained."""

from __future__ import annotations

import wave

import numpy as np


def load_wav(path: str, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV as float32 mono at ``target_rate``."""
    with wave.open(path, "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM WAV, got {wav.getsampwidth() * 8}-bit")
        channels = wav.getnchannels()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != target_rate:
        audio = resample(audio, rate, target_rate)
    return audio, target_rate


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resampling — adequate for speech going into an ASR model."""
    if src_rate == dst_rate:
        return audio
    duration = len(audio) / src_rate
    src_t = np.linspace(0, duration, len(audio), endpoint=False)
    dst_t = np.linspace(0, duration, int(round(duration * dst_rate)), endpoint=False)
    return np.interp(dst_t, src_t, audio).astype(np.float32)
