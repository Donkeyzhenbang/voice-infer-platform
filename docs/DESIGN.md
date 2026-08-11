# Voice Inference Platform — 设计文档

> 分布式语音推理服务平台 · 从 VoxEMW 提炼核心数据流，重构为可横向扩展的微服务架构

---

## 一、设计目标

| 目标 | VoxEMW 现状 | 本平台方案 |
|------|-------------|------------|
| **并发会话** | 1（`num_pipelines: 1`） | 无上限（按 GPU 线性扩展） |
| **组件耦合** | 单进程紧耦合 | 独立微服务 + 消息队列解耦 |
| **GPU 利用** | 全部组件同卡争抢 | 按组件粒度分配 GPU，独立扩缩 |
| **故障隔离** | 单点故障全挂 | 单组件故障不影响其他，自动重试 |
| **部署方式** | 手动脚本 | Docker Compose（单机）→ K8s（集群） |
| **会话状态** | 进程内存 | Redis 集中存储 |
| **模型更新** | 重启全服务 | 滚动更新 + 灰度发布 |

---

## 二、整体架构

### 2.1 架构图

```
                          ┌─────────────────────────────────────┐
                          │            API Gateway              │
                          │  (Nginx/Traefik + WS Proxy)         │
                          │           :80 / :443                │
                          └──────────────┬──────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────┐
                          │          Orchestrator               │
                          │   会话管理 · 人设 · 路由 · 降级       │
                          │          (CPU, 可多副本)             │
                          └──────┬───────┬───────┬──────────────┘
                                 │       │       │
                    ┌────────────┼───────┼───────┼────────────┐
                    │            │  Redis / NATS              │
                    │     ┌──────▼──┐ ┌──▼──────┐ ┌─────────┐ │
                    │     │ Session │ │  Job Q  │ │  Pub/Sub │ │
                    │     │  State  │ │(pipeline)│ │ (events) │ │
                    │     └─────────┘ └─────────┘ └──────────┘ │
                    └────────────┬───────┬───────┬──────────────┘
                                 │       │       │
          ┌──────────────────────┼───────┼───────┼──────────────────┐
          │                      │       │       │                   │
    ┌─────▼──────┐   ┌──────────▼──┐ ┌──▼──────┐ ┌─▼────────┐   ┌──▼──────┐
    │ VAD Worker │   │  ASR Worker │ │   LLM   │ │   TTS    │   │ Avatar  │
    │  (CPU)     │   │   (GPU)     │ │  Proxy  │ │  Worker  │   │ Worker  │
    │            │   │             │ │  (API)  │ │  (GPU)   │   │ (GPU)   │
    │ silero-vad │   │SenseVoiceS  │ │DeepSeek │ │ VoxCPM2  │   │ AVTR-1  │
    │ ×N 副本     │   │ ×N 副本     │ │         │ │ ×N 副本   │   │ ×N 副本  │
    └────────────┘   └─────────────┘ └─────────┘ └──────────┘   └─────────┘

          ┌──────────────────────────────────────────────────────┐
          │                  存储层                              │
          │  ┌─────────┐  ┌──────────┐  ┌───────────────────┐   │
          │  │  Redis  │  │  MinIO   │  │  Qdrant / Milvus  │   │
          │  │ 状态/队列│  │ 音频/模型 │  │    向量库         │   │
          │  └─────────┘  └──────────┘  └───────────────────┘   │
          └──────────────────────────────────────────────────────┘
```

### 2.2 核心数据流

```
用户说话 (Browser)
    │
    │  WebSocket (binary PCM 16kHz mono int16)
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Orchestrator                                                 │
│  1. 接收 PCM 音频帧                                          │
│  2. 写入 Redis Stream: session:<id>:audio_in                 │
│  3. 消费 Redis Stream: session:<id>:audio_out → 浏览器播放    │
└─────────────────────────────────────────────────────────────┘
    │
    │  VAD Worker 消费 audio_in
    ▼
┌─────────────────────────────────────────────────────────────┐
│ VAD Worker                                                   │
│  - 读取: session:<id>:audio_in                               │
│  - 处理: silero-vad 检测语音段                                │
│  - 写入: session:<id>:speech_segments (完整语音段 PCM)         │
└─────────────────────────────────────────────────────────────┘
    │
    │  ASR Worker 消费 speech_segments
    ▼
┌─────────────────────────────────────────────────────────────┐
│ ASR Worker                                                   │
│  - 读取: session:<id>:speech_segments                        │
│  - 处理: SenseVoiceSmall 转写                                 │
│  - 写入: Redis Pub/Sub → session:<id>:transcription           │
│         (orchestrator 转发给浏览器显示转写文本)                │
│  - 发布: session:<id>:user_text                              │
└─────────────────────────────────────────────────────────────┘
    │
    │  LLM Proxy 消费 user_text
    ▼
┌─────────────────────────────────────────────────────────────┐
│ LLM Proxy                                                    │
│  - 读取: session:<id>:user_text + persona + memory_context    │
│  - 处理: DeepSeek API 流式生成                                │
│  - 写入: Redis Stream: session:<id>:llm_tokens (流式逐token)   │
│         (orchestrator 转发给浏览器显示打字效果)                │
│  - 发布: session:<id>:response_text (完整句子/段落)            │
└─────────────────────────────────────────────────────────────┘
    │
    │  TTS Worker 消费 response_text
    ▼
┌─────────────────────────────────────────────────────────────┐
│ TTS Worker                                                   │
│  - 读取: session:<id>:response_text + persona voice_config    │
│  - 处理: VoxCPM2 流式合成 (48kHz → 16kHz)                     │
│  - 写入: Redis Stream: session:<id>:audio_chunks (PCM chunks) │
│  - 可选: session:<id>:audio_chunks → Avatar Worker            │
└─────────────────────────────────────────────────────────────┘
    │
    │  Orchestrator 消费 audio_chunks
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Orchestrator → Browser                                       │
│  - WebSocket 推送 PCM 音频 → 浏览器播放                       │
│  - 可选: 视频帧 → 浏览器渲染数字人                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、技术选型

### 3.1 模型选型（沿用 VoxEMW 验证过的方案）

| 组件 | 模型 | 硬件 | 延迟 | 备注 |
|------|------|------|------|------|
| VAD | silero-vad | CPU | <1ms | torch.hub 加载，极轻量 |
| ASR | SenseVoiceSmall | GPU (1-2GB) | ~0.1s | ModelScope，非自回归 |
| LLM | DeepSeek v4-flash | API | ~1.4s 首 token | 流式，可切换本地 vLLM |
| TTS | VoxCPM2 | GPU (5-6GB) | ~0.1s 首音 | Ultimate Cloning，流式 |
| Embedding | bge-m3 | GPU (2-3GB) | — | 记忆+RAG 共用 |

### 3.2 基础设施选型

| 层次 | 技术 | 说明 |
|------|------|------|
| **消息队列** | Redis Streams + Pub/Sub | 轻量，单机满足；后续可换 NATS/Kafka |
| **会话状态** | Redis Hash | session_id → {persona, history, voice, …} |
| **对象存储** | MinIO (开发) / S3 (生产) | 音频片段、参考音频、模型权重 |
| **向量数据库** | Qdrant (内嵌/独立) | Memory 长期记忆检索 |
| **容器编排** | Docker Compose (单机) → K8s (集群) | 渐进式部署 |
| **服务发现** | Docker DNS (单机) → Consul (集群) | 组件间寻址 |
| **API 网关** | Traefik / Nginx | WebSocket 代理 + 负载均衡 |
| **监控** | Prometheus + Grafana | GPU 利用率、延迟、吞吐 |

### 3.3 开发语言与框架

| 组件 | 语言 | 框架 | 原因 |
|------|------|------|------|
| Orchestrator | Python | aiohttp / FastAPI | 异步 WebSocket，生态成熟 |
| VAD Worker | Python | 独立进程 | 轻量，无框架依赖 |
| ASR Worker | Python | FastAPI + funasr | GPU 推理，HTTP/gRPC 接口 |
| LLM Proxy | Python | FastAPI + httpx | API 代理，流式转发 |
| TTS Worker | Python | FastAPI + voxcpm | GPU 推理 |
| Avatar Worker | Python | FastAPI + AVTR-1 | GPU 渲染 |
| Memory Worker | Python | FastAPI + mem0ai | 异步写入，不占语音延迟 |
| Knowledge Worker | Python | FastAPI + sentence-transformers | PDF 入库 + 检索 |

---

## 四、项目目录结构

```
voice-infer-platform/
├── README.md
├── docker-compose.yml          # 单机一键部署
├── Makefile                    # 常用命令封装
│
├── configs/                    # 配置文件
│   ├── gateway.yaml            # API Gateway 配置
│   ├── orchestrator.yaml       # Orchestrator 配置
│   ├── workers.yaml            # 各 Worker 的统一配置
│   ├── personas.yaml           # 人设注册表
│   └── models.yaml             # 模型路径与参数
│
├── deploy/                     # 部署相关
│   ├── docker/
│   │   ├── Dockerfile.base     # 基础镜像（torch + 公共依赖）
│   │   ├── Dockerfile.asr      # ASR Worker 镜像
│   │   ├── Dockerfile.tts      # TTS Worker 镜像
│   │   └── Dockerfile.orch     # Orchestrator 镜像
│   └── k8s/                    # K8s 部署清单（后续）
│       ├── namespace.yaml
│       ├── orchestrator.yaml
│       └── workers.yaml
│
├── src/                        # 源代码
│   ├── __init__.py
│   │
│   ├── common/                 # 公共模块
│   │   ├── __init__.py
│   │   ├── config.py           # 配置加载（YAML + env）
│   │   ├── redis_client.py     # Redis 封装（Streams + Pub/Sub + Hash）
│   │   ├── schema.py           # 消息协议定义（Pydantic）
│   │   ├── audio_utils.py      # 音频处理工具（重采样、格式转换）
│   │   ├── logging.py          # 统一日志
│   │   └── types.py            # 公共类型定义
│   │
│   ├── orchestrator/           # 编排层
│   │   ├── __init__.py
│   │   ├── server.py           # aiohttp Web 服务入口
│   │   ├── ws_handler.py       # WebSocket 会话管理
│   │   ├── session.py          # 会话状态机
│   │   ├── persona.py          # 人设管理与注入
│   │   └── router.py           # 请求路由到 Worker
│   │
│   ├── workers/                # 推理 Worker
│   │   ├── __init__.py
│   │   ├── base.py             # Worker 基类（Redis 消费循环）
│   │   ├── vad/                # VAD Worker
│   │   │   ├── __init__.py
│   │   │   └── worker.py
│   │   ├── asr/                # ASR Worker
│   │   │   ├── __init__.py
│   │   │   └── worker.py
│   │   ├── llm/                # LLM Proxy
│   │   │   ├── __init__.py
│   │   │   └── worker.py
│   │   ├── tts/                # TTS Worker
│   │   │   ├── __init__.py
│   │   │   └── worker.py
│   │   ├── avatar/             # Avatar Worker（可选）
│   │   │   ├── __init__.py
│   │   │   └── worker.py
│   │   ├── memory/             # Memory Worker
│   │   │   ├── __init__.py
│   │   │   └── worker.py
│   │   └── knowledge/          # Knowledge Worker
│   │       ├── __init__.py
│   │       └── worker.py
│   │
│   └── web/                    # 前端页面
│       ├── index.html          # 对话主页面
│       ├── knowledge.html      # 知识库管理页
│       ├── css/
│       └── js/
│           ├── app.js          # 主逻辑
│           ├── webrtc.js       # WebRTC 管理
│           └── audio.js        # 音频播放/录音
│
├── tests/                      # 测试
│   ├── unit/                   # 单元测试
│   └── integration/            # 集成测试
│
├── scripts/                    # 运维脚本
│   ├── download_models.sh      # 模型下载
│   ├── start_dev.sh            # 开发环境启动
│   └── tunnel.sh               # SSH 隧道
│
├── personas/                   # 人设文件
│   └── default.md
│
└── assets/                     # 素材
    └── default/
        ├── ref.wav
        ├── ref.txt
        └── ref.png
```

---

## 五、核心协议设计

### 5.1 Redis 消息通道

```
┌──────────────────────────────────────────────────────────────┐
│                     Redis 数据结构                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Streams（有序持久化队列，支持 Consumer Group）                  │
│  ─────────────────────────────────────────                   │
│  session:<id>:audio_in         # 用户原始 PCM 音频帧           │
│  session:<id>:speech_segments  # VAD 切出的完整语音段           │
│  session:<id>:llm_tokens       # LLM 流式 token               │
│  session:<id>:audio_chunks     # TTS 流式 PCM chunk           │
│                                                              │
│  Pub/Sub（实时广播）                                           │
│  ────────────────                                            │
│  session:<id>:transcription    # ASR 转写结果                  │
│  session:<id>:response_text    # LLM 完整句子（送 TTS）        │
│  session:<id>:control          # 控制事件（interrupt/persona）  │
│                                                              │
│  Hash（键值状态）                                              │
│  ───────────────                                             │
│  session:<id>:state            # 会话状态（persona/voice/…）   │
│  session:<id>:history          # 对话历史（最近 N 轮）          │
│                                                              │
│  Keys（过期控制）                                              │
│  ───────────────                                             │
│  session:<id>:lock             # 会话锁（防重入, TTL 60s）     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 消息 Schema（JSON over Redis）

```python
# audio_in — 用户音频帧
{
    "session_id": "uuid",
    "seq": 0,                    # 帧序号
    "timestamp": 1692000000.0,   # 采集时间戳
    "sample_rate": 16000,
    "audio": "<base64 pcm int16>"
}

# speech_segment — VAD 切出的完整一段
{
    "session_id": "uuid",
    "segment_id": "uuid",
    "start_ms": 0,
    "end_ms": 3200,
    "audio": "<base64 pcm int16>"
}

# transcription — ASR 结果
{
    "session_id": "uuid",
    "segment_id": "uuid",
    "text": "你好你好",
    "emotion": "NEUTRAL",        # SenseVoice 情绪标签
    "is_final": true
}

# response_text — LLM 完整句子（送 TTS）
{
    "session_id": "uuid",
    "turn_id": "uuid",
    "text": "你好！有什么可以帮你的？",
    "is_final": true             # false=流式中途
}

# llm_token — LLM 流式 token（送前端显示）
{
    "session_id": "uuid",
    "turn_id": "uuid",
    "token": "你好",
    "index": 0
}

# audio_chunk — TTS 输出
{
    "session_id": "uuid",
    "turn_id": "uuid",
    "chunk_id": 0,
    "sample_rate": 16000,
    "audio": "<base64 pcm int16>",
    "is_first": true,
    "is_final": false
}

# control — 控制事件
{
    "session_id": "uuid",
    "type": "interrupt"           # interrupt | persona_change | session_end
}
```

### 5.3 浏览器 ↔ Orchestrator WebSocket 协议

```javascript
// → 上行（浏览器 → 服务端）
{ "type": "audio_frame",       "audio": "<base64>", "seq": 0 }
{ "type": "interrupt" }         // 用户打断
{ "type": "persona_change",    "persona_id": "fengge" }

// ← 下行（服务端 → 浏览器）
{ "type": "transcription",     "text": "你好你好", "is_final": true }
{ "type": "llm_token",         "token": "你好", "index": 0 }
{ "type": "audio_chunk",       "audio": "<base64>", "chunk_id": 0 }
{ "type": "status",            "persona": "fengge", "avatar": false }
```

---

## 六、Worker 基类设计

```python
# src/workers/base.py
from abc import ABC, abstractmethod
from redis.asyncio import Redis

class BaseWorker(ABC):
    """所有 Worker 的基类：消费 Redis Stream → 处理 → 写入下游"""

    def __init__(self, redis: Redis, consumer_group: str, input_stream: str):
        self.redis = redis
        self.group = consumer_group
        self.input_stream = input_stream

    async def run(self):
        """主循环：消费 → 处理 → ACK → 循环"""
        await self._ensure_group()
        while True:
            messages = await self.redis.xreadgroup(
                self.group, self.consumer_name,
                {self.input_stream: ">"}, count=1, block=5000
            )
            for stream, entries in messages:
                for msg_id, data in entries:
                    try:
                        await self.process(data)
                        await self.redis.xack(self.input_stream, self.group, msg_id)
                    except Exception as e:
                        logger.exception("处理失败: %s", e)
                        # 不 ACK，让其他 consumer 重试

    @abstractmethod
    async def process(self, data: dict) -> None:
        """处理一条消息，写入下游"""
        ...

    @property
    def consumer_name(self) -> str:
        return f"{self.group}-{os.getpid()}"
```

---

## 七、会话生命周期

```
Browser 连接        Orchestrator 创建 Session         Worker 初始化
    │                      │                              │
    ├─ WS connect ────────►│                              │
    │                      ├─ session:<id>:state 写入 Redis│
    │                      ├─ persona lookup               │
    │                      ├─ memory recall (async)        │
    │◄─ status ────────────┤                              │
    │                      │                              │
    ├─ audio_frame ───────►│                              │
    │                      ├─ XADD audio_in ──────────────►│ VAD Worker
    │                      │                              ├─ 检测语音段
    │                      │◄─ speech_segment ─────────────┤
    │                      ├─ XADD speech_segments ───────►│ ASR Worker
    │                      │                              ├─ 转写
    │◄─ transcription ─────┤◄─ Pub/Sub transcription ─────┤
    │                      ├─ XADD user_text ─────────────►│ LLM Proxy
    │                      │                              ├─ 流式生成
    │◄─ llm_token ─────────┤◄─ llm_tokens ────────────────┤
    │                      ├─ Pub/Sub response_text ──────►│ TTS Worker
    │                      │                              ├─ 流式合成
    │◄─ audio_chunk ───────┤◄─ audio_chunks ──────────────┤
    │   (浏览器播放)        │                              │
    │                      │                              │
    ├─ interrupt ─────────►│                              │
    │                      ├─ Pub/Sub control:interrupt ──►│ ALL Workers
    │                      │                              ├─ 丢弃积压数据
    │                      │                              │
    ├─ WS close ──────────►│                              │
    │                      ├─ TTL 过期清理                  │
    │                      ├─ memory save (async) ────────►│ Memory Worker
```

---

## 八、分布式扩展设计

### 8.1 扩缩策略

| Worker | 扩缩方式 | 瓶颈 | 机制 |
|--------|----------|------|------|
| Orchestrator | 水平扩展 ×N | 无状态（状态在 Redis） | Nginx upstream / Traefik sticky session |
| VAD | 水平扩展 ×N | CPU | Redis Consumer Group（自动负载均衡） |
| ASR | 水平扩展 ×N | GPU 显存 | 每副本独占 1 GPU，K8s resource limit |
| LLM Proxy | 水平扩展 ×N | API rate limit | 多 API key 轮询 / 本地 vLLM 部署 |
| TTS | 水平扩展 ×N | GPU 显存 | 每副本独占 1 GPU，模型预热 |
| Avatar | 水平扩展 ×N | GPU 显存 | 按需启动（冷启动容忍） |
| Memory | 单实例 / 主从 | 写入吞吐 | 异步写入不占语音延迟 |

### 8.2 Consumer Group 自动负载均衡

```
                    Redis Stream: speech_segments
                    ┌─────────────────────────────┐
                    │ msg1  msg2  msg3  msg4 ...  │
                    └──────┬───────┬──────────────┘
                           │       │
              ┌────────────┼───────┼────────────┐
              │            │       │            │
         ┌────▼───┐   ┌───▼───┐  ┌▼───────┐  ┌─▼───────┐
         │ ASR-1  │   │ ASR-2 │  │ ASR-3  │  │ ASR-N   │
         │ GPU:0  │   │ GPU:1 │  │ GPU:2  │  │ GPU:N-1 │
         └────────┘   └───────┘  └────────┘  └─────────┘
         
         同一个 Consumer Group "asr-workers"
         Redis 自动将消息分发给不同 consumer
         每个 consumer 处理完 XACK，失败不 ACK → 自动重试
```

### 8.3 部署形态

```
阶段一：单机 Docker Compose（开发/演示）
─────────────────────────────────────
  1× Redis, 1× Orchestrator, 1× VAD, 1× ASR, 1× TTS
  全部容器共享 1 GPU（通过 CUDA_VISIBLE_DEVICES）

阶段二：多 GPU 单机（小规模生产）
─────────────────────────────────────
  1× Redis, 2× Orchestrator, 2× VAD, 2× ASR, 2× TTS
  每个 GPU Worker 绑定不同 GPU

阶段三：K8s 集群（大规模生产）
─────────────────────────────────────
  GPU 节点池，每个 Worker 声明 GPU resource
  HPA 根据队列深度自动扩缩
  Redis Cluster / NATS 替代单机 Redis
```

---

## 九、与 VoxEMW 的差异对比

| 维度 | VoxEMW | 本平台 |
|------|--------|--------|
| **进程模型** | 3 进程（orchestrator + pipeline + avatar） | N 个独立容器（每个组件独立部署） |
| **通信方式** | 直连 WebSocket (127.0.0.1) | Redis Streams + Pub/Sub |
| **会话隔离** | `num_pipelines=1` 单槽位 | 每个 session 独立 Redis key 空间 |
| **模型加载** | 启动时全量预加载 | 按 Worker 粒度懒加载 + 预热 |
| **打断机制** | pipeline 内 cancel_scope | Redis Pub/Sub 广播 cancel 事件 |
| **前端** | 单页 HTML + WebRTC | 同方案，WebSocket 直连 |
| **配置** | 单文件 YAML | 分层 YAML（gateway/orchestrator/workers） |
| **监控** | 无 | Prometheus metrics + Grafana dashboard |

---

## 十、实施计划

### Phase 1: 骨架搭建（本次）
- [ ] 项目目录初始化
- [ ] `common/` 公共模块（config, redis, schema, audio_utils）
- [ ] Worker 基类 (`base.py`)
- [ ] `docker-compose.yml`（Redis + 空服务占位）
- [ ] `configs/` 配置文件

### Phase 2: 核心管线（单体模式，快速验证）
- [ ] Orchestrator（WebSocket + 会话管理）
- [ ] VAD Worker（silero-vad）
- [ ] ASR Worker（SenseVoiceSmall）
- [ ] LLM Proxy（DeepSeek API）
- [ ] TTS Worker（VoxCPM2）
- [ ] 前端页面（index.html + audio 播放/录音）
- [ ] 端到端跑通：说话 → 文字回复 + 语音播放

### Phase 3: 增强功能
- [ ] Persona 人设系统
- [ ] Memory 长期记忆（Mem0 + Qdrant）
- [ ] Knowledge RAG（bge-m3 + SQLite）
- [ ] 打断功能（interrupt）
- [ ] 流式 LLM token 前端显示

### Phase 4: 分布式就绪
- [ ] Consumer Group 多实例验证
- [ ] GPU 资源隔离
- [ ] 健康检查 + 自动重启
- [ ] Prometheus metrics
- [ ] Docker 镜像构建与优化

### Phase 5: Avatar（可选）
- [ ] AVTR-1 集成
- [ ] WebRTC 音画轨
- [ ] TURN 服务器

---

## 十一、待 Review 确认

1. **消息队列选择**：Redis Streams vs NATS？Redis 轻量但 NATS 更适合云原生。建议先 Redis 快速验证，后续平滑迁移。
2. **LLM 部署模式**：纯 API 代理 vs 本地部署 vLLM/SGLang？API 模式零运维但延迟不可控。
3. **前端方案**：沿用 VoxEMW 原生 HTML/JS 还是上 React/Vue？原生更轻量。
4. **会话过期策略**：Redis TTL 自动清理 vs 定时任务扫描？
5. **音频编码**：PCM int16 base64 vs 直接二进制帧？base64 调试方便但体积大 33%。
