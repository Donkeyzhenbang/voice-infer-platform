"""会话管理：persona/voice 解耦。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Session:
    session_id: str
    persona_id: str = "default"
    voice_id: str = "default"
    persona_text: str = ""
    created_at: float = 0.0


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(self, session_id, persona_id="default", persona_text="",
               voice_id="default"):
        import time
        s = Session(session_id=session_id, persona_id=persona_id,
                    persona_text=persona_text, voice_id=voice_id,
                    created_at=time.time())
        self._sessions[session_id] = s
        return s

    def get(self, session_id):
        return self._sessions.get(session_id)

    def update_persona(self, session_id, persona_id, persona_text):
        s = self._sessions.get(session_id)
        if s:
            s.persona_id = persona_id
            s.persona_text = persona_text

    def update_voice(self, session_id, voice_id):
        s = self._sessions.get(session_id)
        if s:
            s.voice_id = voice_id

    def remove(self, session_id):
        self._sessions.pop(session_id, None)

    @property
    def active_count(self):
        return len(self._sessions)
