"""TTS 引擎：VoxCPM2 + prompt_cache 预编码，音色一致 + 推理加速。"""

from __future__ import annotations

import logging
import re
import math
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

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


class VoxCPM2TTS(TTSEngine):
    def __init__(self, model_path, device="cuda", sample_rate=16000,
                 cfg_value=2.0, inference_timesteps=10, voices=None):
        self.model_path = model_path
        self.device = device
        self.sample_rate = sample_rate
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.voices: dict[str, VoiceSpec] = {}
        self.voice_prompts: dict[str, dict] = {}  # voice_id → prompt_cache (GPU tensors)
        self._pending_voices: list[VoiceSpec] = []  # 模型未加载时暂存
        self._model = None
        self._model_sr = 48000
        self._resample_up = 1
        self._resample_down = 3  # 48000→16000: gcd=16000, up=1, down=3
        self._blocksize = 512  # 512 sample ≈ 32ms @ 16kHz
        self._loaded = False

        # 先登记 voices
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
        """调用 build_prompt_cache 预编码参考音频为 GPU latent features。"""
        try:
            ref_wav = spec.ref_wav
            ref_text = spec.ref_text.strip()
            logger.info("Encoding voice '%s': ref=%s", spec.voice_id, ref_wav)

            if ref_text:
                # Ultimate Cloning：同一音频双路使用
                cache = self._model.tts_model.build_prompt_cache(
                    prompt_text=ref_text,
                    prompt_wav_path=ref_wav,
                    reference_wav_path=ref_wav,
                )
            else:
                # 仅音色克隆
                cache = self._model.tts_model.build_prompt_cache(
                    reference_wav_path=ref_wav,
                )
            self.voice_prompts[spec.voice_id] = cache
            logger.info("Voice '%s' encoded to GPU cache", spec.voice_id)
        except Exception as e:
            logger.error("Failed to encode voice '%s': %s", spec.voice_id, e)

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
        logger.info("VoxCPM2 model loaded (sr=%d)", self._model_sr)

        # 计算有理重采样比
        g = math.gcd(self._model_sr, self.sample_rate)
        self._resample_up = self.sample_rate // g
        self._resample_down = self._model_sr // g

        self._loaded = True

        # 预编码所有登记的音色
        for spec in self._pending_voices:
            self._encode_voice(spec)
        self._pending_voices.clear()

        # 预热：消除首句冷启动延迟
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

    async def synthesize(self, text, voice_id, session_id, turn_id=""):
        cid = 0
        for audio_chunk in self._stream(text, voice_id):
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

    def _stream(self, text: str, voice_id: str = DEFAULT_VOICE) -> Iterator[np.ndarray]:
        """流式合成，产出 float32 16kHz blocksize 音频块。"""
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
            )
        else:
            logger.warning("No prompt cache for '%s', falling back to zero-shot", voice_id)
            gen = self._model.generate_streaming(
                text=text, cfg_value=self.cfg_value,
                inference_timesteps=self.inference_timesteps,
            )

        needs_resample = self._model_sr != self.sample_rate
        buf = np.empty(0, dtype=np.float32)

        try:
            for item in gen:
                # _generate_with_prompt_cache streaming: (wav, _, _)   generate_streaming: raw tensor
                wav = item[0] if isinstance(item, tuple) else item
                audio = np.atleast_1d(
                    np.asarray(wav.squeeze(0).cpu() if hasattr(wav, 'cpu') else
                               np.asarray(wav).squeeze(), dtype=np.float32).squeeze()
                )
                if audio.size == 0: continue
                if needs_resample:
                    audio = resample_poly(audio, self._resample_up, self._resample_down).astype(np.float32)

                # blocksize 缓冲
                buf = np.concatenate([buf, audio])
                while len(buf) >= self._blocksize:
                    yield buf[:self._blocksize]
                    buf = buf[self._blocksize:]
        finally:
            gen.close()
            if len(buf) > 0:
                yield buf
