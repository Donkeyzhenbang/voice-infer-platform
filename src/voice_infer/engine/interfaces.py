"""推理引擎抽象接口。

所有引擎实现都遵循这些接口，确保实现可替换。
Phase 1: 同进程直调  Phase 2: 进程间通信  Phase 3: 消息队列
——接口不变，只换实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from voice_infer.common.schema import AudioChunk, AudioSegment, LLMResponse, Transcription


class VADEngine(ABC):
    """语音活动检测引擎。

    输入连续音频流，输出检测到的语音段。
    """

    @abstractmethod
    async def detect(self, audio_chunk: bytes, session_id: str) -> list[AudioSegment]:
        """处理一段音频数据，返回新检测到的完整语音段。

        Args:
            audio_chunk: int16 PCM bytes, 16kHz mono
            session_id: 会话标识

        Returns:
            本次新检测到的完整语音段列表（可能为空）
        """
        ...

    @abstractmethod
    async def reset(self, session_id: str) -> None:
        """重置 VAD 状态（新会话或打断后）。"""
        ...


class ASREngine(ABC):
    """语音转文字引擎。"""

    @abstractmethod
    async def transcribe(self, segment: AudioSegment) -> Transcription:
        """将一段语音转写为文字。

        Args:
            segment: VAD 输出的语音段

        Returns:
            转写结果（文本 + 情绪标签）
        """
        ...


class LLMEngine(ABC):
    """大语言模型引擎。

    流式输出回复，每句一个 LLMResponse，方便 TTS 逐句合成。
    """

    @abstractmethod
    async def generate(
        self,
        user_text: str,
        session_id: str,
        history: list[dict[str, str]] | None = None,
        instructions: str | None = None,
    ) -> AsyncIterator[LLMResponse]:
        """流式生成回复。

        Args:
            user_text: 用户说的话
            session_id: 会话标识
            history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            instructions: 系统指令（人设正文）

        Yields:
            LLMResponse: 逐句流式输出，每句是一个完整句子
        """
        ...


class TTSEngine(ABC):
    """文字转语音引擎。

    流式合成，边生成边输出 PCM 音频块。
    """

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        session_id: str,
    ) -> AsyncIterator[AudioChunk]:
        """将文本流式合成为语音。

        Args:
            text: 要合成的一句话
            voice_id: 音色标识（对应 voices 配置中的 key）
            session_id: 会话标识

        Yields:
            AudioChunk: 流式 PCM 音频块
        """
        ...
