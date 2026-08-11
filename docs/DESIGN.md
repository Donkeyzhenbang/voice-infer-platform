# Voice Inference Platform — 设计文档

> 语音推理服务平台 · 渐进式架构：先跑通核心管线，再单机多卡分离，最后分布式

---

## 一、设计理念

**渐进式演进，不过度设计**：

```
Phase 1: 单进程直调          Phase 2: 多进程单机多卡        Phase 3: 多机分布式
─────────────────────      ───────────────────────      ──────────────────
                                                          
  Orchestrator              Orchestrator (CPU)            Orchestrator ×N
    │                         │ (Unix Socket)              API Gateway
    ├─ vad()                  ├─ VAD Process    (CPU)      │ (Redis/NATS)
    ├─ asr()                  ├─ ASR Process    (GPU:0)    ├─ VAD Pod ×N
    ├─ llm()                  ├─ LLM Process    (CPU/API)  ├─ ASR Pod ×N
    ├─ tts()                  ├─ TTS Process    (GPU:1)    ├─ LLM Pod ×N
    └─ memory()               └─ Memory Process (CPU)      ├─ TTS Pod ×N
                                                           └─ Avatar Pod ×N
  目标: 跑通管线              目标: GPU 隔离 + 并行        目标: 弹性伸缩
  开发周期: 1-2 天            开发周期: 2-3 天             开发周期: 按需
```

**核心原则**：
1. **数据流先于中间件** — 先让声音能流转，再考虑用什么传
2. **接口抽象，实现可换** — 组件间通过明确的协议通信，直调/进程间/队列三种实现可互换
3. **单机多卡优先** — 一台 8×GPU 机器远比分布式集群常见，先解决这个
4. **配置驱动** — 模型、参数、路由全部 YAML 管理，不硬编码

---

## 二、核心数据流

这是整个平台的心脏，不管哪种部署形态，数据流不变：

```
用户说话
  │
  ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   VAD    │───▶│   ASR    │───▶│   LLM    │───▶│   TTS    │───▶│  音频    │──▶ 浏览器播放
│ (Silero) │    │(SenseVo- │    │(DeepSeek │    │ (VoxCPM2)│    │  输出    │
│          │    │ iceSmall)│    │ v4-flash)│    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
  │                │                │                │
  │ 语音段         │ 转写文本        │ LLM 回复        │ PCM 音频
  │ PCM float32    │                │ (句子级流式)     │ 16kHz int16
  │                │                │                │
  │                ▼                ▼                │
  │           浏览器显示          浏览器打字           │
  │           (实时字幕)          (逐句)              │
  │                                                  │
  └── 用户新语音 → VAD 检测 → 取消当前 epoch ────────┘
       (打断: Pipeline 检测 _is_stale() 清理所有积压)
```

### WebSocket 双队列架构（实际实现）

```
浏览器 ←── WebSocket ──→ FastAPI Server
                              │
                    ┌─────────┴──────────┐
                    │   receiver_task    │  ← 接收用户 PCM
                    │        │           │
                    │   pcm_queue        │  ← asyncio.Queue
                    │        │           │
                    │   turn_worker      │  ← Pipeline.process()
                    │        │           │     (VAD→ASR→LLM→TTS)
                    │   out_queue        │  ← asyncio.Queue
                    │        │           │
                    │   sender_task      │  ← 发送结果到浏览器
                    └────────────────────┘

打断流程:
  用户按下"打断" → cancel() 
    → epoch += 1
    → 清空 pcm_queue
    → turn_worker 检测 _is_stale() 停止所有生成
```

### 统一数据协议

```python
# src/voice_infer/common/schema.py

@dataclass
class AudioSegment:        # VAD → ASR
    session_id: str
    segment_id: str
    audio: np.ndarray      # float32, 16kHz mono
    start_ms: int
    end_ms: int

@dataclass  
class Transcription:       # ASR → LLM + 前端
    session_id: str
    segment_id: str
    text: str
    emotion: str           # HAPPY/SAD/NEUTRAL/…
    is_final: bool

@dataclass
class LLMResponse:         # LLM → TTS（句子级流式）
    session_id: str
    turn_id: str
    text: str
    is_final: bool

@dataclass
class AudioChunk:          # TTS → 浏览器（PCM 流式）
    session_id: str
    turn_id: str
    chunk_id: int
    audio: bytes           # int16 PCM, 16kHz mono
    is_first: bool
    is_final: bool
```

---

## 三、Phase 1 架构：单进程直调 ✅ 已完成

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                          │
│                   (单进程, :8000)                         │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              PipelineEngine                       │   │
│  │  (纯 Python 函数调用，无网络开销)                   │   │
│  │                                                   │   │
│  │  双队列 WS 架构:                                    │   │
│  │    receiver → pcm_queue → turn_worker              │   │
│  │    turn_worker → out_queue → sender                │   │
│  │                                                   │   │
│  │  epoch 机制:                                       │   │
│  │    每轮对话一个 epoch，打断时递增                    │   │
│  │    VAD/ASR/LLM/TTS 每步检查 _is_stale()             │   │
│  │    lock 只保护 VAD detect（避免整体超时）            │   │
│  │                                                   │   │
│  │  整轮语义:                                          │   │
│  │    audio_start/end 只在整轮首尾各发一次              │   │
│  │    中间 TTS chunk 均设 is_first=False               │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐    │
│  │ WS Handler│  │ Session  │  │ Persona / Memory   │    │
│  │ (FastAPI) │  │ Manager  │  │ / Knowledge         │    │
│  └──────────┘  └──────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────┘
         │
         │ WebSocket (:8000)
         ▼
    ┌──────────┐
    │  浏览器   │  ← iOS 风格白底蓝泡，录音/播放/打断
    └──────────┘
```

**特点**：一个进程、零网络开销、`python -m` 一条命令启动。

### 模型清单

| 组件 | 引擎 | 模型 | 显存 |
|------|------|------|------|
| VAD | Silero VAD | `snakers4/silero-vad`（torch.hub 本地） | ~0 |
| ASR | SenseVoiceSmall | `iic/SenseVoiceSmall`（FunASR） | ~1.5 GB |
| LLM | DeepSeek v4-flash | API 调用（`api.deepseek.com`） | 0 |
| TTS | VoxCPM2 | `openbmb/VoxCPM2`（ModelScope） | ~5.5 GB |
| Memory | bge-m3 | `BAAI/bge-m3`（mem0ai，默认关闭） | ~2.5 GB |

### 组件接口

```python
# src/voice_infer/engine/interfaces.py

class VADEngine(ABC):
    @abstractmethod
    async def detect(self, audio: np.ndarray, session_id: str) -> list[AudioSegment]: ...

class ASREngine(ABC):
    @abstractmethod
    async def transcribe(self, segment: AudioSegment) -> Transcription: ...

class LLMEngine(ABC):
    @abstractmethod
    async def generate(self, text: str, session_id: str) -> AsyncIterator[LLMResponse]: ...

class TTSEngine(ABC):
    @abstractmethod
    async def synthesize(self, text: str, voice_id: str, session_id: str) -> AsyncIterator[AudioChunk]: ...
```

---

## 四、Phase 2 架构：多进程单机多卡

```
┌─────────────────────────────────────────────────────────────────┐
│                        单台 8×GPU 服务器                          │
│                                                                 │
│  ┌──────────────┐                                               │
│  │ Orchestrator │  (CPU, 端口 8000)                              │
│  │ 会话管理+路由 │                                               │
│  └──────┬───────┘                                               │
│         │                                                       │
│         │  Unix Domain Socket (localhost)                        │
│         │                                                       │
│  ┌──────▼──────┬──────────┬──────────┬──────────┬──────────┐    │
│  │  VAD Proc   │ ASR Proc │ TTS Proc │ TTS Proc │ LLM Proc │    │
│  │  (CPU)      │ (GPU:0)  │ (GPU:1)  │ (GPU:2)  │ (CPU)    │    │
│  │  ×1         │  ×1      │  ×1      │  ×2      │  ×1      │    │
│  └─────────────┴──────────┴──────────┴──────────┴──────────┘    │
│                                                                 │
│  GPU 分配示例 (configs/gpu_alloc.yaml):                          │
│    ASR:  GPU 0 (SenseVoiceSmall ~1.5GB)                         │
│    TTS:  GPU 1,2 (VoxCPM2 ~5.5GB each)                          │
│    Emb:  GPU 3 (bge-m3 ~2.5GB)                                  │
│    Free: GPU 4-7                                                │
└─────────────────────────────────────────────────────────────────┘
```

通信使用自定义二进制帧协议（不走 Redis）：

```
Header (8B): [msg_type(2B)] [reserved(2B)] [payload_len(4B)]
Payload:     JSON 元数据 + 可选二进制块(PCM/图片)
```

---

## 五、Phase 3 架构：多机分布式（后续）

单机 8 GPU 不够时，引入消息队列（Redis/NATS），Phase 2 完成后按需细化。

---

## 六、项目目录结构

```
voice-infer-platform/
├── README.md
├── pyproject.toml
│
├── configs/                     # 所有配置
│   ├── server.yaml              # 端口、日志
│   ├── pipeline.yaml            # 管线参数（VAD/ASR/LLM/TTS）
│   └── personas.yaml            # 人设注册表
│
├── src/voice_infer/             # 主包
│   ├── __init__.py
│   │
│   ├── common/                  # 公共模块
│   │   ├── config.py            # YAML 配置加载
│   │   ├── schema.py            # 数据协议定义
│   │   ├── audio.py             # 音频工具
│   │   └── logging.py           # 日志
│   │
│   ├── engine/                  # 推理引擎
│   │   ├── interfaces.py        # 抽象接口
│   │   ├── pipeline.py          # PipelineEngine 编排器
│   │   ├── vad/silero_vad.py    # VAD 实现
│   │   ├── asr/sensevoice.py    # ASR 实现
│   │   ├── llm/deepseek.py      # LLM 实现
│   │   └── tts/voxcpm2.py       # TTS 实现
│   │
│   ├── server/                  # 服务层
│   │   ├── app.py               # 入口
│   │   ├── ws.py                # WebSocket
│   │   ├── session.py           # 会话管理
│   │   └── persona.py           # 人设
│   │
│   ├── memory/store.py          # Mem0 封装 (Phase 1 关)
│   └── knowledge/store.py       # 知识库 (Phase 1 关)
│
├── web/                         # 前端
│   ├── index.html
│   ├── css/
│   └── js/
│
├── tests/
├── scripts/
├── personas/default.md
└── assets/default/
```

---

## 七、配置设计

### `configs/pipeline.yaml`

```yaml
vad:
  engine: silero
  min_silence_ms: 500
  min_speech_ms: 250

asr:
  engine: sensevoice
  model: iic/SenseVoiceSmall
  device: cuda
  language: zh

llm:
  engine: deepseek
  model: deepseek-v4-flash
  base_url: https://api.deepseek.com/v1
  api_key_env: DEEPSEEK_API_KEY
  stream: true
  system_prompt: "你是语音助手。用中文口语回复，简短自然。"

tts:
  engine: voxcpm2
  model: /root/.cache/modelscope/models/openbmb--VoxCPM2/snapshots/master
  device: cuda
  sample_rate: 16000
  cfg_value: 1.0           # 无参考音频时用 1.0，音色最稳定
  inference_timesteps: 10
  optimize: false
  atempo_rate: 0.886
  # 不使用参考音频 —— VoxCPM2 内置默认声音，干净无 artifacts
  # 参考音频会导致反馈循环劣化（详见 docs/troubleshooting.md）
```

### `configs/server.yaml`

```yaml
server:
  host: "0.0.0.0"
  port: 8000

session:
  max_history: 30
  idle_timeout: 300
```

---

## 八、实施计划

### Phase 1: 核心管线 ✅ 已完成

```
✔ 1.1 项目骨架
  ✔ pyproject.toml
  ✔ common/（config, schema, audio, logging）
  ✔ configs/ 配置文件（server.yaml, pipeline.yaml）

✔ 1.2 推理引擎
  ✔ engine/interfaces.py（抽象接口）
  ✔ engine/vad/silero_vad.py（本地缓存加载）
  ✔ engine/asr/sensevoice.py（FunASR + ModelScope）
  ✔ engine/llm/deepseek.py（thinking 禁用 + 括号动作过滤）
  ✔ engine/tts/voxcpm2.py（无参考音频，cudnn 确定性，cfg_value=1.0）
  ✔ engine/pipeline.py（epoch 取消 + 整轮 audio_start/end）

✔ 1.3 服务层
  ✔ server/app.py（FastAPI + 双队列 WebSocket + ASGI Origin 绕过）
  ✔ server/session.py（persona_id / voice_id 解耦）
  ✔ server/persona.py

✔ 1.4 前端
  ✔ web/index.html（白底蓝泡录音/播放 + 打断按钮）

✔ 1.5 验证
  ✔ 端到端：说话 → 转写 → LLM 回复 → 语音播放
  ✔ 打断功能正常
  ✔ 已部署：ssh 隧道 localhost:8000
```

### Phase 2: 多进程单机多卡

详见 [`docs/phase2-roadmap.md`](phase2-roadmap.md)。

```
□ Phase 2a: Push 模式（会话绑定 Worker）
  □ Worker 进程封装（Unix Socket + 二进制帧协议）
  □ WorkerAllocator（Least Connections）
  □ gpu_alloc.yaml 配置
  □ 多用户并发验证

□ Phase 2b: Pull 模式（按需，组件池 + 任务队列）
```

### Phase 3: 分布式（按需）

```
□ 消息队列 (Redis/NATS)
□ K8s + HPA
```

