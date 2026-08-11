"""Pipeline：整轮音频 start/end 只发一次，避免多句 TTS 互相截断。"""

from __future__ import annotations

import asyncio, logging, uuid
from typing import AsyncIterator

from voice_infer.common.schema import AudioChunk, LLMResponse, Transcription
from voice_infer.engine.interfaces import ASREngine, LLMEngine, TTSEngine, VADEngine

logger = logging.getLogger(__name__)


class PipelineEngine:
    def __init__(self, vad, asr, llm, tts):
        self.vad = vad; self.asr = asr; self.llm = llm; self.tts = tts
        self._history: dict[str, list[dict]] = {}
        self._epoch: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _is_stale(self, sid, epoch): return self._epoch.get(sid, 0) != epoch

    async def cancel(self, sid):
        self._epoch[sid] = self._epoch.get(sid, 0) + 1
        await self.vad.reset(sid)

    async def process(self, audio_chunk: bytes, session_id: str,
                      instructions=None, voice_id="default",
                      ) -> AsyncIterator[Transcription | LLMResponse | AudioChunk]:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            segments = await self.vad.detect(audio_chunk, session_id)
            if not segments: return
            epoch = self._epoch.get(session_id, 0)
            segment = segments[0]

        try:
            if self._is_stale(session_id, epoch): return
            t = await self.asr.transcribe(segment)
            if not t.text.strip() or self._is_stale(session_id, epoch): return
            yield t

            history = self._history.get(session_id, [])
            full, turn_id = "", uuid.uuid4().hex[:12]
            audio_started = False  # 整轮只发一次 audio_start

            async for r in self.llm.generate(
                user_text=t.text.strip(), session_id=session_id,
                history=history, instructions=instructions,
            ):
                if self._is_stale(session_id, epoch): return
                yield r; full += r.text

                async for c in self.tts.synthesize(text=r.text, voice_id=voice_id, session_id=session_id, turn_id=turn_id):
                    if self._is_stale(session_id, epoch): return
                    # 只在第一句的第一个 chunk 发 audio_start
                    if c.is_first and not audio_started:
                        audio_started = True
                        yield AudioChunk(session_id=session_id, turn_id=turn_id, chunk_id=0, audio=b"", sample_rate=16000, is_first=True, is_final=False)
                    # 抑制每句的 is_first，避免前端重置播放队列
                    yield AudioChunk(session_id=c.session_id, turn_id=turn_id, chunk_id=c.chunk_id,
                                     audio=c.audio, sample_rate=c.sample_rate,
                                     is_first=False, is_final=c.is_final)

            # 整轮最后发 audio_end
            if audio_started:
                yield AudioChunk(session_id=session_id, turn_id=turn_id, chunk_id=9999, audio=b"", sample_rate=16000, is_first=False, is_final=True)

            if not self._is_stale(session_id, epoch) and full.strip():
                history.append({"role": "user", "content": t.text.strip()})
                history.append({"role": "assistant", "content": full.strip()})
                if len(history) > 60: history = history[-60:]
                self._history[session_id] = history

        finally:
            pass

    async def reset_session(self, sid):
        self._history.pop(sid, None); self._epoch.pop(sid, None)
        await self.vad.reset(sid)
