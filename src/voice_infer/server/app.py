"""FastAPI 入口：双队列 WS 架构。"""

import nltk as _nltk; _nltk.download = lambda *_a, **_kw: None

import asyncio, json, logging, os, uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from voice_infer.common.config import load_config
from voice_infer.common.logging import setup_logging
from voice_infer.engine.asr.sensevoice import SenseVoiceASR
from voice_infer.engine.llm.deepseek import DeepSeekLLM
from voice_infer.engine.pipeline import PipelineEngine
from voice_infer.engine.tts.voxcpm2 import VoxCPM2TTS, VoiceSpec
from voice_infer.engine.vad.silero_vad import SileroVAD
from voice_infer.server.persona import load_persona
from voice_infer.server.session import SessionManager
from voice_infer.memory.store import create_memory_store

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class VoiceManager:
    def __init__(self, tts, cfg):
        self.tts = tts
        self.custom_dir = REPO_ROOT / "data" / "voices"
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        for d in self.custom_dir.iterdir():
            if d.is_dir() and (d / "ref.wav").is_file():
                vid = d.name
                if vid not in self.tts.voices:
                    self.tts.register_voice(VoiceSpec(vid, str(d / "ref.wav"), (d / "ref.txt").read_text("utf-8").strip()))

    def list_voices(self): return self.tts.list_voices()

    def create_voice(self, vid, wav_bytes, ref_text):
        if not vid or not vid.replace("_","").replace("-","").isalnum() or len(vid) > 64: raise ValueError("invalid voice_id")
        if ".." in vid or "/" in vid or "\\" in vid: raise ValueError("invalid voice_id")
        vdir = self.custom_dir / vid; vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "ref.wav").write_bytes(wav_bytes)
        (vdir / "ref.txt").write_text(ref_text, "utf-8")
        self.tts.register_voice(VoiceSpec(vid, str(vdir / "ref.wav"), ref_text.strip()))
        return True


async def _ws_handler(ws, pipeline, sessions, personas, voice_mgr):
    sid = uuid.uuid4().hex[:16]
    dp = personas.get("default", list(personas.values())[0])
    s = sessions.create(sid, "default", dp.get("text", ""), "default")
    logger.info("WS: %s", sid[:8])

    pcm_queue: asyncio.Queue = asyncio.Queue(maxsize=64)   # receiver → turn_worker
    out_queue: asyncio.Queue = asyncio.Queue(maxsize=128)   # turn_worker → sender

    async def receiver():
        try:
            while True:
                raw = await ws.receive()
                if "text" in raw:
                    msg = json.loads(raw["text"]); t = msg.get("type", "")
                    if t == "interrupt":
                        await pipeline.cancel(sid)
                        # 清空积压 PCM
                        while not pcm_queue.empty():
                            try: pcm_queue.get_nowait()
                            except: break
                    elif t == "persona_change":
                        pid = msg.get("persona_id", "default")
                        p = personas.get(pid, dp)
                        sessions.update_persona(sid, pid, p.get("text", ""))
                        await out_queue.put(json.dumps({"type": "persona_changed", "persona_id": pid}))
                    elif t == "voice_change":
                        sessions.update_voice(sid, msg.get("voice_id", "default"))
                        await out_queue.put(json.dumps({"type": "voice_changed", "voice_id": msg.get("voice_id", "default")}))
                elif "bytes" in raw:
                    try: pcm_queue.put_nowait(raw["bytes"])
                    except asyncio.QueueFull: pass
        except WebSocketDisconnect: pass
        finally:
            await pcm_queue.put(None)  # 通知 turn_worker 结束

    async def sender():
        try:
            while True:
                item = await out_queue.get()
                if item is None: break
                if isinstance(item, str): await ws.send_text(item)
                elif isinstance(item, bytes): await ws.send_bytes(item)
        except WebSocketDisconnect: pass

    async def turn_worker():
        while True:
            pcm = await pcm_queue.get()
            if pcm is None:
                await out_queue.put(None); break
            try:
                async for event in pipeline.process(
                    audio_chunk=pcm, session_id=sid,
                    instructions=s.persona_text, voice_id=s.voice_id,
                ):
                    if hasattr(event, 'text') and getattr(event, 'text', ''):
                        type_name = type(event).__name__
                        await out_queue.put(json.dumps({
                            "type": "transcription" if "Transcription" in type_name else "llm_token",
                            "text": event.text,
                            "is_final": getattr(event, 'is_final', True),
                            "emotion": getattr(event, 'emotion', 'NEUTRAL') if "Transcription" in type_name else None,
                        }))
                    elif hasattr(event, 'audio'):
                        if event.is_first: await out_queue.put(json.dumps({"type": "audio_start"}))
                        if event.audio: await out_queue.put(event.audio)
                        if event.is_final: await out_queue.put(json.dumps({"type": "audio_end"}))
            except Exception as e:
                logger.error("Turn error: %s", e)

    await ws.accept()
    tasks = [asyncio.create_task(receiver()), asyncio.create_task(sender()), asyncio.create_task(turn_worker())]
    try: await asyncio.gather(*tasks)
    except: pass
    finally:
        for t in tasks: t.cancel()
        await pipeline.reset_session(sid); sessions.remove(sid)


class _WSBypass:
    def __init__(self, app, handler=None, **kw): self.app = app; self.h = handler
    async def __call__(self, scope, r, s):
        if scope["type"] == "websocket" and scope.get("path","").rstrip("/") == "/ws":
            await self.h(scope, r, s)
        else: await self.app(scope, r, s)


async def _ws_asgi(scope, receive, send, **kw):
    ws = WebSocket(scope, receive, send)
    await _ws_handler(ws, **kw)


def _abs(p): path = Path(p); return str(path) if path.is_absolute() else str(REPO_ROOT / p)


def create_app(config=None):
    if config is None:
        config = load_config(REPO_ROOT / "configs/pipeline.yaml", REPO_ROOT / "configs/server.yaml", REPO_ROOT / ".env.local", REPO_ROOT)
    setup_logging(config.server.log_level); pc = config.pipeline

    logger.info("Init VAD..."); vad = SileroVAD(pc.vad.min_silence_ms, pc.vad.min_speech_ms, pc.vad.speech_pad_ms)
    logger.info("Load ASR..."); asr = SenseVoiceASR(pc.asr.model, pc.asr.device, pc.asr.language); asr.load_model()
    lc = pc.llm; logger.info("Init LLM...")
    llm = DeepSeekLLM(lc.model, lc.base_url, api_key_env=lc.api_key_env, max_tokens=lc.max_tokens, temperature=lc.temperature, system_prompt=lc.system_prompt)
    tc = pc.tts; logger.info("Load TTS...")
    voices = {vid: VoiceSpec(vid, _abs(vc.ref_wav), vc.ref_text) for vid, vc in tc.voices.items()}
    tts = VoxCPM2TTS(tc.model, tc.device, tc.sample_rate, tc.cfg_value, tc.inference_timesteps,
                     atempo_rate=tc.atempo_rate, voices=voices)
    tts.load_model(optimize=tc.optimize)
    pipeline = PipelineEngine(vad, asr, llm, tts); sessions = SessionManager()
    voice_mgr = VoiceManager(tts, config)
    api_key = os.environ.get(lc.api_key_env, "")
    memory = create_memory_store(pc, api_key)

    personas = {}
    pd = REPO_ROOT / "personas"
    if pd.is_dir():
        for mf in pd.glob("*.md"):
            try: p = load_persona(mf); personas[p["id"]] = p
            except: pass
    if not personas: personas["default"] = {"id":"default","name":"Default","label":"默认","text":"你是语音助手。用中文口语回复，简短自然。"}

    app = FastAPI(title="Voice Infer", version="0.3.1")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.get("/")
    async def index():
        hp = REPO_ROOT / "web" / "index.html"
        return HTMLResponse(hp.read_text("utf-8")) if hp.is_file() else HTMLResponse("<h1>Voice Infer</h1>")

    @app.get("/voice")
    async def voice_page():
        hp = REPO_ROOT / "web" / "voice.html"
        return HTMLResponse(hp.read_text("utf-8")) if hp.is_file() else HTMLResponse("<h1>Record</h1>")

    @app.get("/voice-processor.js")
    async def awp():
        from fastapi.responses import Response
        js = (REPO_ROOT / "web" / "voice-processor.js").read_text("utf-8")
        return Response(js, media_type="application/javascript")

    @app.get("/api/personas")
    async def api_p():
        return {"default":"default","list":[{"id":pid,"name":p["name"],"label":p.get("label",p["name"])} for pid,p in personas.items()]}

    @app.get("/api/voices")
    async def api_v(): return {"voices": voice_mgr.list_voices()}

    @app.post("/api/voices/create")
    async def api_create_voice(voice_id: str = Form("my_voice"), file: UploadFile = File(...), ref_text: str = Form("")):
        try:
            wav = await file.read()
            if len(wav) < 100 or len(wav) > 10*1024*1024: return {"ok":False,"error":"size"}
            voice_mgr.create_voice(voice_id, wav, ref_text)
            return {"ok":True,"voice_id":voice_id}
        except ValueError as e: return {"ok":False,"error":str(e)}

    @app.get("/api/memory/status")
    async def api_mem(): return {"enabled": memory.enabled}

    @app.get("/api/health")
    async def health(): return {"status":"ok","sessions":sessions.active_count,"memory":memory.enabled}

    from functools import partial
    app.add_middleware(_WSBypass, handler=partial(_ws_asgi, pipeline=pipeline, sessions=sessions, personas=personas, voice_mgr=voice_mgr))
    logger.info("Ready (memory=%s)", memory.enabled)
    return app


def main():
    import uvicorn
    config = load_config(REPO_ROOT / "configs/pipeline.yaml", REPO_ROOT / "configs/server.yaml", REPO_ROOT / ".env.local", REPO_ROOT)
    uvicorn.run(create_app(config), host=config.server.host, port=config.server.port, log_level=config.server.log_level, server_header=False)

if __name__ == "__main__": main()
