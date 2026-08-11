"""Pipeline：串行 LLM→TTS + 三层记忆。先稳再快。"""

from __future__ import annotations

import asyncio, logging, uuid
from typing import AsyncIterator

from voice_infer.common.schema import AudioChunk, LLMResponse, Transcription
from voice_infer.engine.interfaces import ASREngine, LLMEngine, TTSEngine, VADEngine

logger = logging.getLogger(__name__)


class PipelineEngine:
    def __init__(self, vad, asr, llm, tts, memory=None, knowledge=None):
        self.vad = vad; self.asr = asr; self.llm = llm; self.tts = tts
        self.memory = memory; self.knowledge = knowledge
        self._history: dict[str, list[dict]] = {}
        self._epoch: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._persona_id: dict[str, str] = {}

    def _is_stale(self, sid, epoch): return self._epoch.get(sid, 0) != epoch

    async def cancel(self, sid):
        self._epoch[sid] = self._epoch.get(sid, 0) + 1
        ev = self._cancel_events.get(sid)
        if ev: ev.set()
        await self.vad.reset(sid)

    def _build_instructions(self, session_id, base_instructions, user_text):
        parts = [base_instructions] if base_instructions else []
        if self.memory and self.memory.enabled:
            agent_id = self._persona_id.get(session_id, "default")
            mems = self.memory.recall(agent_id)
            if mems:
                parts.append("# 关于用户的记忆\n" + "\n".join(f"- {m}" for m in mems))
        if self.knowledge and self.knowledge.enabled:
            rag = self.knowledge.build_rag_block(user_text)
            if rag:
                parts.append(rag)
        return "\n\n".join(parts) if parts else ""

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

            enriched = self._build_instructions(session_id, instructions, t.text.strip())

            async for r in self.llm.generate(
                user_text=t.text.strip(), session_id=session_id,
                history=history, instructions=enriched or instructions,
            ):
                if self._is_stale(session_id, epoch): return
                yield r; full += r.text

                async for c in self.tts.synthesize(
                    text=r.text, voice_id=voice_id, session_id=session_id,
                    turn_id=turn_id, cancelled=cancel_ev,
                ):
                    if self._is_stale(session_id, epoch): return
                    if c.is_first and not audio_started:
                        audio_started = True
                        yield AudioChunk(session_id=session_id, turn_id=turn_id, chunk_id=0, audio=b"", sample_rate=16000, is_first=True, is_final=False)
                    yield AudioChunk(session_id=c.session_id, turn_id=turn_id, chunk_id=c.chunk_id,
                                     audio=c.audio, sample_rate=c.sample_rate,
                                     is_first=False, is_final=c.is_final)

            if audio_started:
                yield AudioChunk(session_id=session_id, turn_id=turn_id, chunk_id=9999, audio=b"", sample_rate=16000, is_first=False, is_final=True)

            if not self._is_stale(session_id, epoch) and full.strip():
                history.append({"role": "user", "content": t.text.strip()})
                history.append({"role": "assistant", "content": full.strip()})
                if len(history) > 60: history = history[-60:]
                self._history[session_id] = history

                if self.memory and self.memory.enabled:
                    agent_id = self._persona_id.get(session_id, "default")
                    asyncio.create_task(asyncio.to_thread(
                        self.memory.remember, t.text.strip(), full.strip(), agent_id))

        finally:
            self._cancel_events.pop(session_id, None)

    async def reset_session(self, sid):
        self._history.pop(sid, None); self._epoch.pop(sid, None)
        self._cancel_events.pop(sid, None)
        await self.vad.reset(sid)
