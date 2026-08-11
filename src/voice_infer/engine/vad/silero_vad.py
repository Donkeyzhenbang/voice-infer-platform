"""VAD 引擎：silero-vad（强制本地缓存，不连 GitHub）。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from voice_infer.common.schema import AudioSegment
from voice_infer.engine.interfaces import VADEngine

logger = logging.getLogger(__name__)
SR = 16000
FS = 512  # 32ms @ 16kHz


class SileroVAD(VADEngine):
    """silero-vad，直接走本地 torch.hub 缓存。"""

    def __init__(self, min_silence_ms=500, min_speech_ms=250, speech_pad_ms=400):
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.speech_pad_ms = speech_pad_ms

        # 强制本地缓存（VoxEMW 已验证 /root/.cache/torch/hub/snakers4_silero-vad_master）
        hub_dir = Path(torch.hub.get_dir()) / "snakers4_silero-vad_master"
        if not hub_dir.is_dir():
            raise RuntimeError(f"silero-vad 未下载，请先执行: torch.hub.load('snakers4/silero-vad', 'silero_vad')")

        logger.info("VAD: local cache %s", hub_dir)
        self.model, _ = torch.hub.load(str(hub_dir), "silero_vad", trust_repo=True, source="local")
        self._buf: dict[str, bytearray] = {}
        self._st: dict[str, dict] = {}
        self._seq: dict[str, int] = {}

    async def detect(self, audio_chunk: bytes, session_id: str) -> list[AudioSegment]:
        buf = self._buf.setdefault(session_id, bytearray())
        buf.extend(audio_chunk)
        st = self._st.setdefault(session_id, {"spk": False, "sil": 0, "spf": 0, "cur": bytearray(), "t0": 0})
        seq = self._seq.setdefault(session_id, 0)
        out: list[AudioSegment] = []

        while len(buf) >= FS * 2:
            fb = bytes(buf[:FS * 2]); del buf[:FS * 2]
            f32 = np.frombuffer(fb, dtype=np.int16).astype(np.float32) / 32768.0
            sp = self.model(torch.from_numpy(f32), SR).item() > 0.5

            if sp:
                st["spf"] += 1; st["sil"] = 0; st["cur"].extend(fb)
                if not st["spk"] and st["spf"] * 32 >= self.min_speech_ms:
                    st["spk"] = True; st["t0"] = seq * 32 - st["spf"] * 32
            else:
                if st["spk"]:
                    st["sil"] += 1; st["cur"].extend(fb)
                    if st["sil"] * 32 >= self.min_silence_ms:
                        seg = self._finalize(st, session_id)
                        if seg: out.append(seg)
                        st.update(spk=False, spf=0, sil=0, cur=bytearray())
                else:
                    st["spf"] = 0
            seq += 1
        self._seq[session_id] = seq
        return out

    def _finalize(self, st, sid):
        sp = st["cur"]; pad = max(0, self.speech_pad_ms // 32)
        trim = max(0, st["sil"] - pad) * FS * 2
        if trim > 0 and len(sp) > trim: sp = sp[:-trim]
        if len(sp) < 1600: return None
        a = np.frombuffer(bytes(sp), dtype=np.int16).astype(np.float32) / 32768.0
        import uuid
        return AudioSegment(session_id=sid, segment_id=uuid.uuid4().hex[:12],
                            audio=a, start_ms=st["t0"], end_ms=st["t0"] + len(a) * 1000 // SR)

    async def reset(self, session_id: str) -> None:
        for d in [self._buf, self._st, self._seq]: d.pop(session_id, None)
