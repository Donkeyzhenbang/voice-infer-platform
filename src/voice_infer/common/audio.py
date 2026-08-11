"""音频工具：重采样、格式转换。"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np


def resample_audio(
    audio: np.ndarray,
    src_rate: int,
    dst_rate: int,
) -> np.ndarray:
    """音频重采样（线性插值，适用于非整数倍率）。

    对于高精度需求，建议用 scipy.signal.resample_poly。
    """
    if src_rate == dst_rate:
        return audio

    import scipy.signal

    # 使用 scipy 的 polyphase 重采样（高质量）
    gcd = np.gcd(src_rate, dst_rate)
    up = dst_rate // gcd
    down = src_rate // gcd
    return scipy.signal.resample_poly(audio, up, down).astype(np.float32)


def pcm_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = 16000,
    sample_width: int = 2,  # int16
) -> bytes:
    """float32 numpy PCM → WAV bytes (用于浏览器 AudioContext 解码)。"""
    if sample_width == 2:
        audio_int = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    else:
        raise ValueError(f"Unsupported sample_width: {sample_width}")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(audio_int.tobytes())
    return buf.getvalue()


def wav_bytes_to_pcm(data: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes → (float32 numpy array, sample_rate)。"""
    with wave.open(io.BytesIO(data), "rb") as w:
        sample_rate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
        audio_int = np.frombuffer(raw, dtype=np.int16)
        audio = audio_int.astype(np.float32) / 32768.0
    return audio, sample_rate


def int16_to_float32(data: bytes) -> np.ndarray:
    """int16 PCM bytes → float32 numpy array（直接转，保留原始幅度）。"""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_int16(audio: np.ndarray) -> bytes:
    """float32 numpy array → int16 PCM bytes。"""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
