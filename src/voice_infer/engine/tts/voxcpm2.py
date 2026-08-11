"""TTS 引擎：VoxCPM2 内置默认声音，干净无杂音。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
import torch

from voice_infer.common.schema import AudioChunk
from voice_infer.engine.interfaces import TTSEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceSpec:
    voice_id: str
    ref_wav: str
    ref_text: str


class VoxCPM2TTS(TTSEngine):
    def __init__(self, model_path, device="cuda", sample_rate=16000,
                 cfg_value=1.0, inference_timesteps=10, voices=None):
        self.model_path = model_path
        self.device = device
        self.sample_rate = sample_rate
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.voices: dict[str, VoiceSpec] = dict(voices or {})
        self._model = None
        self._model_sr = 48000

    def register_voice(self, spec: VoiceSpec):
        self.voices = {**self.voices, spec.voice_id: spec}

    def list_voices(self):
        return [{"id": v.voice_id, "type": "builtin" if v.voice_id == "default" else "custom"}
                for v in self.voices.values()]

    def load_model(self):
        if self._model is not None: return
        from voxcpm import VoxCPM
        self._model = VoxCPM.from_pretrained(
            self.model_path, device=self.device,
            load_denoiser=False, optimize=False, local_files_only=True)
        self._model_sr = int(self._model.tts_model.sample_rate)
        logger.info("VoxCPM2 loaded")

    async def synthesize(self, text, voice_id, session_id, turn_id=""):
        from voice_infer.common.audio import resample_audio

        # 确定性：固定种子 + CUDA 确定性模式，消除音色随机漂移
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 不用任何参考音频——内置默认声音，干净无 artifacts
        # cfg_value=1.0 不给无条件分支加权，音色最稳定（无参考时不宜 >1.5）
        gen = self._model.generate_streaming(
            text=text, cfg_value=1.0,
            inference_timesteps=self.inference_timesteps)

        cid = 0
        for wav in gen:
            a = np.asarray(wav.squeeze()).astype(np.float32)
            if a.ndim > 1: a = a.squeeze()
            a16 = resample_audio(a, self._model_sr, self.sample_rate)
            if a16.size == 0: continue
            ab = (np.clip(a16, -1, 1) * 32767).astype(np.int16).tobytes()
            first = (cid == 0); cid += 1
            yield AudioChunk(session_id=session_id, turn_id=turn_id, chunk_id=cid,
                             audio=ab, sample_rate=self.sample_rate,
                             is_first=first, is_final=False)
        yield AudioChunk(session_id=session_id, turn_id=turn_id, chunk_id=cid+1,
                         audio=b"", sample_rate=self.sample_rate,
                         is_first=False, is_final=True)
