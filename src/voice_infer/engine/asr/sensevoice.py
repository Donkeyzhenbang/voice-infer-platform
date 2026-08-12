"""ASR 引擎：SenseVoiceSmall（FunASR）。"""

from __future__ import annotations

import logging
import re

from voice_infer.common.schema import AudioSegment, Transcription
from voice_infer.engine.interfaces import ASREngine

logger = logging.getLogger(__name__)

# SenseVoice 情绪标签正则
_EMOTION_RE = re.compile(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|DISGUSTED|FEARFUL|SURPRISED)\|>")


class SenseVoiceASR(ASREngine):
    """基于 FunASR SenseVoiceSmall 的语音转文字。

    非自回归，整段一次前向出全文。
    4s 中文语音约 0.1s（GPU）。
    """

    def __init__(self, model: str = "iic/SenseVoiceSmall", device: str = "cuda", language: str = "zh"):
        self.model_path = model
        self.device = device
        self.language = language
        self._model = None

    async def _ensure_model(self):
        if self._model is None:
            self.load_model()

    def load_model(self):
        """同步加载模型（启动时调用，避免首次请求延迟）。"""
        if self._model is not None:
            return
        from funasr import AutoModel
        logger.info("Loading SenseVoiceSmall: %s on %s", self.model_path, self.device)
        self._model = AutoModel(model=self.model_path, device=self.device, disable_update=True)
        import numpy as np
        self._model.generate(input=np.zeros(16000, dtype=np.float32), language=self.language, use_itn=True)
        logger.info("SenseVoiceSmall ready")

    async def transcribe(self, segment: AudioSegment) -> Transcription:
        await self._ensure_model()

        result = self._model.generate(
            input=segment.audio,
            language=self.language,
            use_itn=True,  # 逆文本正则化（数字、标点等）
        )

        raw_text = ""
        emotion = "NEUTRAL"

        if result and len(result) > 0:
            item = result[0]
            raw_text = item.get("text", "")

            # 提取情绪标签
            m = _EMOTION_RE.search(raw_text)
            if m:
                emotion = m.group(1)
            # 剥掉 SenseVoice 的元标签
            raw_text = re.sub(r"<\|[^|]+\|>", "", raw_text).strip()

        logger.info("ASR: %r (emotion=%s)", raw_text, emotion)

        return Transcription(
            session_id=segment.session_id,
            segment_id=segment.segment_id,
            text=raw_text,
            emotion=emotion,
            is_final=True,
        )

    async def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        """转写 WAV bytes → 文本（供音色克隆自动生成 ref_text）。

        确保 ref_text 与实际录音逐字一致，避免 Ultimate Cloning
        prompt 续写错位导致重复脚本台词。
        """
        import io
        import wave as wave_mod
        import numpy as np

        await self._ensure_model()
        try:
            with wave_mod.open(io.BytesIO(wav_bytes), "rb") as w:
                sr = w.getframerate()
                raw = w.readframes(w.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if sr != 16000:
                from voice_infer.common.audio import resample_audio
                audio = resample_audio(audio, sr, 16000)

            result = self._model.generate(
                input=audio, language=self.language, use_itn=True,
            )
            text = ""
            if result and len(result) > 0:
                text = re.sub(r"<\|[^|]+\|>", "", result[0].get("text", "")).strip()
            logger.info("Voice ref ASR: %r", text)
            return text
        except Exception as e:
            logger.warning("Voice ref ASR failed: %s", e)
            return ""
