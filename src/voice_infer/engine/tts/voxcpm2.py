"""TTS 引擎：VoxCPM2 + prompt_cache 预编码，音色一致 + 推理加速。

修复（2026-08-11）：
- atempo 流式语速补偿（抵消 VoxCPM2 克隆语速偏快 ~12%）
- cancel 信号传入 _stream() 每 chunk 检查，打断立即停止
- flush 吞音修复（尾部>blocksize 先整块吐完再 pad）
- build_prompt_cache 前对参考音频做峰值归一化预处理
- @torch.inference_mode() 省显存
"""

from __future__ import annotations

import logging
import re
import math
import subprocess
import threading
import queue
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Optional

import numpy as np
import torch
from scipy.signal import resample_poly

from voice_infer.common.schema import AudioChunk
from voice_infer.engine.interfaces import TTSEngine

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "default"


@dataclass(frozen=True)
class VoiceSpec:
    voice_id: str
    ref_wav: str
    ref_text: str


class _AtempoStretcher:
    """ffmpeg atempo 流式变速（保调）：16kHz mono f32 进/出。

    VoxCPM2 克隆语速比参考音快 ~12%，rate=0.886 补偿。
    """

    def __init__(self, sample_rate: int, rate: float):
        self._q: queue.Queue = queue.Queue()
        self._p = subprocess.Popen(
            [
                "ffmpeg", "-v", "error",
                "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
                "-af", f"atempo={rate}",
                "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "pipe:1",
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )
        self._buf = b""
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while True:
            data = self._p.stdout.read(65536)
            if not data:
                self._q.put(None)
                return
            self._q.put(data)

    def feed(self, audio_f32: np.ndarray) -> np.ndarray:
        """喂一块 f32，返回当前可得的拉伸输出（可能为空）。"""
        self._p.stdin.write(audio_f32.astype(np.float32).tobytes())
        self._p.stdin.flush()
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                self._buf += item
        out, self._buf = self._buf, b""
        return np.frombuffer(out, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """收尾：关闭 stdin 后读干。"""
        try:
            self._p.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        while True:
            try:
                item = self._q.get(timeout=10)
            except queue.Empty:
                logger.warning("atempo flush timeout")
                break
            if item is None:
                break
            self._buf += item
        self._p.wait()
        out, self._buf = self._buf, b""
        return np.frombuffer(out, dtype=np.float32)

    def close(self):
        """打断废弃。"""
        try:
            self._p.kill()
        except OSError:
            pass


class VoxCPM2TTS(TTSEngine):
    def __init__(self, model_path, device="cuda", sample_rate=16000,
                 cfg_value=2.0, inference_timesteps=10, atempo_rate=1.0,
                 gen_kwargs=None, voices=None):
        self.model_path = model_path
        self.device = device
        self.sample_rate = sample_rate
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.atempo_rate = atempo_rate  # 1.0=不变速，0.886≈抵消 +12%
        self.gen_kwargs = gen_kwargs or {}
        self.voices: dict[str, VoiceSpec] = {}
        self.voice_prompts: dict[str, dict] = {}
        self._pending_voices: list[VoiceSpec] = []
        self._model = None
        self._model_sr = 48000
        self._resample_up = 1
        self._resample_down = 3
        self._blocksize = 512  # 32ms @ 16kHz
        self._loaded = False

        for vid, vc in (voices or {}).items():
            self.register_voice(vc)

    # ── 音色注册 ──────────────────────────────────────────────

    def register_voice(self, spec: VoiceSpec):
        self.voices[spec.voice_id] = spec
        if self._loaded:
            self._encode_voice(spec)
        else:
            self._pending_voices.append(spec)

    def _encode_voice(self, spec: VoiceSpec):
        """预编码参考音频为 GPU latent features。

        default 音色的 ref.wav 是机器生成的，跳过编码——回退 zero-shot 干净输出。
        只有真人录制的声音才走 Ultimate Cloning。
        """
        try:
            # default 音色的参考音频是机器生成的，不编码（避免 artifacts 反馈循环）
            if spec.voice_id == DEFAULT_VOICE:
                logger.info("Skipping '%s' voice encoding (machine-generated ref, using zero-shot)", DEFAULT_VOICE)
                return

            ref_wav = spec.ref_wav
            ref_text = spec.ref_text.strip()
            logger.info("Encoding voice '%s': ref=%s", spec.voice_id, ref_wav)

            # 预处理参考音频：归一化
            ref_wav_norm = self._preprocess_ref_audio(ref_wav)

            if ref_text:
                cache = self._model.tts_model.build_prompt_cache(
                    prompt_text=ref_text,
                    prompt_wav_path=ref_wav_norm,
                    reference_wav_path=ref_wav_norm,
                    trim_silence_vad=True,
                )
            else:
                cache = self._model.tts_model.build_prompt_cache(
                    reference_wav_path=ref_wav_norm,
                    trim_silence_vad=True,
                )
            self.voice_prompts[spec.voice_id] = cache
            logger.info("Voice '%s' encoded", spec.voice_id)
        except Exception as e:
            logger.error("Failed to encode voice '%s': %s", spec.voice_id, e)

    def _preprocess_ref_audio(self, wav_path: str) -> str:
        """归一化参考音频到 -1dB 峰值，输出临时文件。

        不修改原文件——只在 build_prompt_cache 前做一次性预处理。
        """
        import soundfile as sf
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        peak = max(abs(audio).max(), 0.001)
        target_peak = 0.891  # -1dB
        gain = target_peak / peak
        if abs(gain - 1.0) < 0.05:  # 已经够响，不用处理
            return wav_path
        audio = audio * gain
        tmp = wav_path + ".norm.wav"
        sf.write(tmp, audio.astype(np.float32), sr)
        return tmp

    def list_voices(self):
        return [{"id": v.voice_id, "type": "builtin" if v.voice_id == DEFAULT_VOICE else "custom",
                 "has_cache": v.voice_id in self.voice_prompts}
                for v in self.voices.values()]

    # ── 模型加载 ──────────────────────────────────────────────

    def load_model(self):
        if self._model is not None: return
        from voxcpm import VoxCPM
        self._model = VoxCPM.from_pretrained(
            self.model_path, device=self.device,
            load_denoiser=False, optimize=False, local_files_only=True)
        self._model_sr = int(self._model.tts_model.sample_rate)
        logger.info("VoxCPM2 loaded (sr=%d)", self._model_sr)

        g = math.gcd(self._model_sr, self.sample_rate)
        self._resample_up = self.sample_rate // g
        self._resample_down = self._model_sr // g

        self._loaded = True

        for spec in self._pending_voices:
            self._encode_voice(spec)
        self._pending_voices.clear()

        if self.atempo_rate != 1.0:
            logger.info("atempo enabled: rate=%.3f (duration ×%.3f)",
                        self.atempo_rate, 1.0 / self.atempo_rate)

        self._warmup()

    def _warmup(self):
        logger.info("Warming up TTS...")
        try:
            for _ in self._stream("你好，这是一条预热测试。"):
                pass
            logger.info("TTS warmup complete")
        except Exception as e:
            logger.warning("TTS warmup failed: %s", e)

    # ── 推理 ──────────────────────────────────────────────────

    async def synthesize(self, text, voice_id, session_id, turn_id="", cancelled=None):
        """cancelled: 可选 asyncio.Event，set 后立即停止生成。"""
        cid = 0
        for audio_chunk in self._stream(text, voice_id, cancelled=cancelled):
            a = np.asarray(audio_chunk).astype(np.float32)
            if a.ndim > 1: a = a.squeeze()
            if a.size == 0: continue
            ab = (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            first = (cid == 0); cid += 1
            yield AudioChunk(session_id=session_id, turn_id=turn_id, chunk_id=cid,
                             audio=ab, sample_rate=self.sample_rate,
                             is_first=first, is_final=False)
        yield AudioChunk(session_id=session_id, turn_id=turn_id, chunk_id=cid + 1,
                         audio=b"", sample_rate=self.sample_rate,
                         is_first=False, is_final=True)

    @torch.inference_mode()
    def _stream(self, text: str, voice_id: str = DEFAULT_VOICE,
                cancelled: Optional["asyncio.Event"] = None) -> Iterator[np.ndarray]:
        """流式合成，产出 float32 16kHz blocksize 音频块。

        cancelled: 外部传入的取消信号，每 chunk 检查，set 后立即停止。
        """
        # 文本清洗
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        text = re.sub(r"[（(][^（）()]{1,20}[)）]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return

        torch.manual_seed(42); torch.cuda.manual_seed_all(42)

        cache = self.voice_prompts.get(voice_id) or self.voice_prompts.get(DEFAULT_VOICE)

        if cache is not None:
            gen = self._model.tts_model._generate_with_prompt_cache(
                target_text=text, prompt_cache=cache, min_len=2, max_len=2000,
                inference_timesteps=self.inference_timesteps, cfg_value=self.cfg_value,
                retry_badcase=False, streaming=True,
                **self.gen_kwargs,
            )
        else:
            logger.warning("No prompt cache for '%s', zero-shot", voice_id)
            gen = self._model.generate_streaming(
                text=text, cfg_value=self.cfg_value,
                inference_timesteps=self.inference_timesteps,
            )

        needs_resample = self._model_sr != self.sample_rate
        # atempo 只用于 prompt_cache 模式（克隆语速偏快 ~12%），zero-shot 不加速
        use_atempo = self.atempo_rate != 1.0 and cache is not None
        stretcher = _AtempoStretcher(self.sample_rate, self.atempo_rate) if use_atempo else None
        cancelled_clean = False
        pending = np.empty(0, dtype=np.int16)
        total_out = 0

        def _check_cancel():
            return cancelled is not None and cancelled.is_set()

        try:
            for item in gen:
                if _check_cancel():
                    cancelled_clean = True
                    return

                wav = item[0] if isinstance(item, tuple) else item
                audio = np.atleast_1d(
                    np.asarray(wav.squeeze(0).cpu() if hasattr(wav, 'cpu') else
                               np.asarray(wav).squeeze(), dtype=np.float32).squeeze()
                )
                if audio.size == 0: continue
                if needs_resample:
                    audio = resample_poly(audio, self._resample_up, self._resample_down).astype(np.float32)

                # atempo 变速
                if stretcher is not None:
                    audio = stretcher.feed(audio)
                    if audio.size == 0: continue

                # int16 缓冲 + blocksize 输出（修复吞音）
                pending = np.concatenate([pending, np.clip(audio * 32767, -32768, 32767).astype(np.int16)])
                while len(pending) >= self._blocksize:
                    yield (pending[:self._blocksize].astype(np.float32) / 32767.0)
                    pending = pending[self._blocksize:]
                    total_out += self._blocksize
        finally:
            gen.close()
            # atempo flush
            if stretcher is not None:
                if cancelled_clean:
                    stretcher.close()
                else:
                    tail = stretcher.flush()
                    if tail.size:
                        pending = np.concatenate(
                            [pending, np.clip(tail * 32767, -32768, 32767).astype(np.int16)])

        # flush 尾巴：先整块吐完再 pad（吞音修复）
        if len(pending) > 0:
            while len(pending) >= self._blocksize:
                yield (pending[:self._blocksize].astype(np.float32) / 32767.0)
                pending = pending[self._blocksize:]
                total_out += self._blocksize
            if len(pending) > 0:
                total_out += len(pending)
                padded = np.pad(pending, (0, self._blocksize - len(pending)))
                yield (padded.astype(np.float32) / 32767.0)
