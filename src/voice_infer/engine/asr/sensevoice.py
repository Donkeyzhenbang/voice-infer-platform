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
