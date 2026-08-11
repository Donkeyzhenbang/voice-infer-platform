"""数据协议定义 —— 所有 Pipeline 组件间传递的消息类型。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioSegment:
    """VAD 切出的一段完整语音。

    从连续音频流中检测到的一段用户说话内容。
    """
    session_id: str
    segment_id: str
    audio: np.ndarray       # float32, (samples,), 16kHz mono
    start_ms: int
    end_ms: int


@dataclass
class Transcription:
    """ASR 转写结果。

    包含转写文本和 SenseVoice 情绪标签。
    """
    session_id: str
    segment_id: str
    text: str
    emotion: str = "NEUTRAL"   # HAPPY, SAD, ANGRY, NEUTRAL, ...
    is_final: bool = True


@dataclass
class LLMResponse:
    """LLM 生成的回复（句子级，逐句流式输出给 TTS）。

    一次 LLM 调用可能产生多个 LLMResponse，
    每个包含一个完整句子，TTS 逐句合成。
    """
    session_id: str
    turn_id: str
    text: str               # 一个完整句子
    is_final: bool = False  # 是否本轮最后一句话


@dataclass
class AudioChunk:
    """TTS 合成的一小块 PCM 音频。

    流式输出，浏览器边收边播，降低首音延迟。
    """
    session_id: str
    turn_id: str
    chunk_id: int
    audio: bytes            # int16 PCM, 16kHz mono
    sample_rate: int = 16000
    is_first: bool = False  # 本轮第一个 chunk（触发 AudioContext 播放）
    is_final: bool = False  # 本轮最后一个 chunk（句尾）
