"""Pipeline：LLM 流式输出 → 后台 TTS 并发生成，首音延迟最小化。"""

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
        self._cancel_events: dict[str, asyncio.Event] = {}

    def _is_stale(self, sid, epoch): return self._epoch.get(sid, 0) != epoch

    async def cancel(self, sid):
        self._epoch[sid] = self._epoch.get(sid, 0) + 1
        ev = self._cancel_events.get(sid)
        if ev: ev.set()
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

        cancel_ev = asyncio.Event()
        self._cancel_events[session_id] = cancel_ev

        try:
            if self._is_stale(session_id, epoch): return
            t = await self.asr.transcribe(segment)
            if not t.text.strip() or self._is_stale(session_id, epoch): return
            yield t

            history = self._history.get(session_id, [])
            full, turn_id = "", uuid.uuid4().hex[:12]
            audio_started = False

            # 后台 TTS 队列：LLM 每出一句立即异步提交
            tts_out: asyncio.Queue = asyncio.Queue(maxsize=256)

            async def _run_tts(text: str, seq: int):
                try:
                    async for c in self.tts.synthesize(
                        text=text, voice_id=voice_id, session_id=session_id,
                        turn_id=turn_id, cancelled=cancel_ev,
                    ):
                        if self._is_stale(session_id, epoch): return
                        await tts_out.put((seq, c))
                except Exception as e:
                    logger.error("TTS bg error: %s", e)

            tts_tasks = []
            seq = 0
            pending: dict[int, list] = {}
            next_out = 0

            async for r in self.llm.generate(
                user_text=t.text.strip(), session_id=session_id,
                history=history, instructions=instructions,
            ):
                if self._is_stale(session_id, epoch): return
                yield r; full += r.text

                # 后台提交 TTS（不阻塞 LLM）
                tts_tasks.append(asyncio.create_task(_run_tts(r.text, seq)))
                seq += 1

                # 非阻塞收 TTS 已有输出
                while True:
                    try:
                        s, c = tts_out.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    pending.setdefault(s, []).append(c)
                # 按序输出
                while next_out in pending:
                    for pc in pending.pop(next_out):
                        if self._is_stale(session_id, epoch): return
                        if pc.is_first and not audio_started:
                            audio_started = True
                            yield AudioChunk(session_id=session_id, turn_id=turn_id, chunk_id=0, audio=b"", sample_rate=16000, is_first=True, is_final=False)
                        yield AudioChunk(session_id=pc.session_id, turn_id=turn_id, chunk_id=pc.chunk_id,
                                         audio=pc.audio, sample_rate=pc.sample_rate,
                                         is_first=False, is_final=pc.is_final)
                    next_out += 1

            # LLM 结束，等所有 TTS 完成
            if tts_tasks:
                await asyncio.gather(*tts_tasks, return_exceptions=True)

            # 收尾
            while not tts_out.empty():
                s, c = tts_out.get_nowait()
                pending.setdefault(s, []).append(c)
            while next_out in pending:
                for pc in pending.pop(next_out):
                    if self._is_stale(session_id, epoch): return
                    if pc.is_first and not audio_started:
                        audio_started = True
                        yield AudioChunk(session_id=session_id, turn_id=turn_id, chunk_id=0, audio=b"", sample_rate=16000, is_first=True, is_final=False)
                    yield AudioChunk(session_id=pc.session_id, turn_id=turn_id, chunk_id=pc.chunk_id,
                                     audio=pc.audio, sample_rate=pc.sample_rate,
                                     is_first=False, is_final=pc.is_final)
                next_out += 1

            if audio_started:
                yield AudioChunk(session_id=session_id, turn_id=turn_id, chunk_id=9999, audio=b"", sample_rate=16000, is_first=False, is_final=True)

            if not self._is_stale(session_id, epoch) and full.strip():
                history.append({"role": "user", "content": t.text.strip()})
                history.append({"role": "assistant", "content": full.strip()})
                if len(history) > 60: history = history[-60:]
                self._history[session_id] = history

        finally:
            self._cancel_events.pop(session_id, None)

    async def reset_session(self, sid):
        self._history.pop(sid, None); self._epoch.pop(sid, None)
        self._cancel_events.pop(sid, None)
        await self.vad.reset(sid)
