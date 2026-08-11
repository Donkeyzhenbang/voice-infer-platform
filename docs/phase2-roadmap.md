# Phase 2 单机多卡 Roadmap

> 目标：单机多 GPU 服务多个用户实例，每用户独立会话管线。

---

## 一、VoxEMW 借鉴分析

### 1.1 音色一致性 — Ultimate Cloning + Prompt Cache

VoxEMW 的核心做法：

```
启动时（一次性）：
  真人录音 ref.wav ──▶ build_prompt_cache() ──▶ GPU latent features
                                                      │
                                            self.voice_prompts dict（常驻显存）

推理时（每句）：
  _generate_with_prompt_cache(prompt_cache=预编码cache)
    → 直接拼接 latent features，零 I/O、零重编码
    → 同一音色所有句子用同一份 cache → 跨句音色完全一致
```

**我们需要改的**：
- 当前：完全没有参考音频，音色从 `randn()` 随机涌现
- 改为：准备真人录音 → 启动时 `build_prompt_cache()` 预编码 → 推理复用
- 注意：**必须用真人录音，不能用机器生成的**（否则反馈循环劣化）

### 1.2 推理加速

| 手段 | 效果 | 我们当前状态 |
|------|------|-------------|
| **Prompt Cache 预编码** | 跳过每句 AudioVAE 编码（~200ms） | ❌ 未使用 |
| **流式生成** | TTFA 0.2-0.5s，不等整句 | ✅ 已实现 |
| **Warmup** | 预热 CUDA kernel / KV cache | ❌ 未实现 |
| **torch.compile (optimize)** | RTF 1.0→0.3-0.5（启动 +1-2min） | ❌ 未使用 |
| **CancelScope** | 打断立即停 GPU 算力 | ✅ epoch 机制等效 |
| **Atempo 语速补偿** | ffmpeg 流式保调变速 | ❌ 未实现 |
| **@torch.inference_mode()** | 省显存 + 加速 | ❌ 未使用 |
| **有理重采样 1/3** | 48k→16k 最小计算量 | ❌ 当前用 librosa |

**优先级建议**：
1. **Prompt Cache**（最大收益：音色一致性 + 速度双赢）
2. **Warmup**（消除首句冷启动延迟）
3. **有理重采样**（微小改动，零成本提速）
4. **@torch.inference_mode()**（一行装饰器）
5. **torch.compile**（需评估：启动慢 1-2min，RTF 降 3 倍）

---

## 二、单机多卡架构设计

### 2.1 硬件假设

```
1 台服务器 × 2-8 张 GPU（如 RTX 4090D 24GB）
每 GPU 显存预算：
  - VAD: ~0 GB（CPU 即可）
  - ASR: ~1.5 GB
  - TTS: ~5.5 GB
  - 余量: ~17 GB（可用于多实例或 memory 模块）
```

### 2.2 进程模型选择：Push vs Pull

#### 方案 A：Push 模式（会话绑定 Worker）

```
                  ┌─────────────┐
                  │ Orchestrator │  (CPU, :8000)
                  │ 会话管理+路由 │
                  └──┬───┬───┬──┘
                     │   │   │    Unix Socket / shared memory
          ┌──────────┘   │   └──────────┐
          ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Worker 0 │  │ Worker 1 │  │ Worker 2 │
    │ (GPU:0)  │  │ (GPU:1)  │  │ (GPU:2)  │
    │          │  │          │  │          │
    │ VAD+ASR  │  │ VAD+ASR  │  │ VAD+ASR  │
    │ +LLM(API)│  │ +LLM(API)│  │ +LLM(API)│
    │ +TTS     │  │ +TTS     │  │ +TTS     │
    └──────────┘  └──────────┘  └──────────┘
         ▲              ▲              ▲
         │              │              │
    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
    │ 会话 A   │    │ 会话 B   │    │ 会话 C   │
    └─────────┘    └─────────┘    └─────────┘

流程：
  1. 用户连接 → Orchestrator 分配空闲 Worker
  2. Worker 持有整条管线，处理该用户全部请求
  3. 用户断开 → Worker 释放，可分配给新用户
  4. 用户打断 → Worker 内部 cancel（epoch 机制直接生效）
```

**优势**：
- 架构简单：Worker 就是当前 PipelineEngine 的子进程版
- 会话亲和性：用户状态（history、memory）全在 Worker 内存，无状态同步
- 打断低延迟：cancel 在本进程内完成，无跨进程协调
- 调试友好：per-Worker 日志，问题定位清晰

**劣势**：
- 负载不均：A 用户一直说话占满 GPU，B 用户空闲浪费算力
- GPU 粒度粗：一个 Worker 至少占一个 GPU，不能"半张卡"分配
- 单 Worker 串行：同一用户的多句间天然串行（VAD→ASR→LLM→TTS），无法流水线并行
- 扩容不灵活：加 GPU 才能加用户容量

#### 方案 B：Pull 模式（组件池 + 任务队列）

```
                  ┌─────────────┐
                  │ Orchestrator │  (CPU, :8000)
                  │ 会话管理+调度 │
                  └──┬───┬───┬──┘
                     │   │   │
         ┌───────────┼───┼───┼───────────┐
         │           │   │   │           │
         ▼           ▼   ▼   ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ASR Pool │ │TTS Pool │ │TTS Pool │ │MEM Pool │
    │(GPU:0)  │ │(GPU:1)  │ │(GPU:2)  │ │(GPU:3)  │
    │×2 实例  │ │×2 实例  │ │×2 实例  │ │×1 实例  │
    └─────────┘ └─────────┘ └─────────┘ └─────────┘
         ▲           ▲           ▲           ▲
         │           │           │           │
         └───────────┴───────────┴───────────┘
                     │
              ┌──────┴──────┐
              │ 消息队列     │  (Redis/NATS/Unix MQ)
              │ asr_q/tts_q │
              └─────────────┘

流程：
  1. 用户音频 → Orchestrator → ASR 任务进 asr_q
  2. ASR Pool 空闲 Worker 拉取任务 → 转写 → 结果回写
  3. Orchestrator 收到转写 → LLM(API) → TTS 任务进 tts_q
  4. TTS Pool 空闲 Worker 拉取任务 → 合成 → 音频回写
  5. Orchestrator 推送音频到浏览器
```

**优势**：
- 负载均衡：任务级调度，自动利用最空闲的 Worker
- 组件独立扩缩：ASR 瓶颈就加 ASR Worker，TTS 瓶颈就加 TTS Worker
- 故障隔离：一个 Worker 崩溃不影响其他任务
- 流水线并行：不同用户的不同阶段可以同时在不同 GPU 上执行

**劣势**：
- 架构复杂：需要消息队列、任务序列化、状态管理
- 会话状态需外部化：history、memory 不能留在 Worker 内存
- 跨进程数据传输：PCM 音频需序列化/反序列化（共享内存可缓解）
- 调试复杂：一次请求跨多个进程，链路追踪困难
- 打断延迟高：cancel 信号需传播到队列中的多个任务

### 2.3 推荐方案：Push 优先 + Pull 预留

**Phase 2a（推荐先做）：Push 模式**

理由：
- 当前核心需求是「服务多个用户」，不是「极致 GPU 利用率」
- 1 张 4090D 可跑 2-3 个完整管线实例（7GB/实例）
- Push 模式改动最小：把 PipelineEngine 包装为子进程 + 简单的分配器
- 等用户量上来、GPU 利用率成瓶颈时，再考虑 Pull 模式

**Phase 2b（后续按需）：Pull 模式**

触发条件：
- Worker 空闲率 > 30% 或用户数 > GPU 实例数 × 2
- 需要独立扩缩 ASR/TTS 时

---

## 三、Phase 2a 详细设计（Push 模式）

### 3.1 架构图

```
┌──────────────────────────────────────────────────────────────┐
│                      Orchestrator (CPU)                       │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ WS Manager  │  │ Session Store│  │ Worker Allocator   │   │
│  │ (浏览器连接) │  │ (Redis/内存) │  │ (分配+回收+健康检查)│   │
│  └──────┬──────┘  └──────────────┘  └────────┬──────────┘   │
│         │                                    │               │
│         │         Unix Socket                │               │
│         └──────────────┬─────────────────────┘               │
│                        │                                     │
└────────────────────────┼─────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ Worker 0 │   │ Worker 1 │   │ Worker 2 │
   │ (GPU:0)  │   │ (GPU:1)  │   │ (GPU:2)  │
   │          │   │          │   │          │
   │ VAD      │   │ VAD      │   │ VAD      │
   │ ASR      │   │ ASR      │   │ ASR      │
   │ LLM(API) │   │ LLM(API) │   │ LLM(API) │
   │ TTS      │   │ TTS      │   │ TTS      │
   │ Memory*  │   │ Memory*  │   │ Memory*  │
   └──────────┘   └──────────┘   └──────────┘

* Memory 模块如果显存压力大，可单独放到 CPU Worker
```

### 3.2 通信协议

沿用 Phase 1 的 `AudioChunk`/`Transcription`/`LLMResponse`，加一层进程间传输：

```
二进制帧协议（Unix Socket，避免 TCP 开销）:

┌──────────┬──────────┬────────────┬──────────────────┐
│ msg_type │ reserved │ payload_len│    payload        │
│  (2B)    │  (2B)    │   (4B)     │  (JSON + binary)  │
└──────────┴──────────┴────────────┴──────────────────┘

msg_type:
  0x01 = PCM_AUDIO       (浏览器→Worker)
  0x02 = TRANSCRIPTION    (Worker→浏览器)
  0x03 = LLM_RESPONSE     (Worker→浏览器)
  0x04 = AUDIO_CHUNK      (Worker→浏览器)
  0x05 = CANCEL           (浏览器→Worker)
  0x06 = HEARTBEAT        (双向)
```

### 3.3 Worker 生命周期

```
                    ┌──────────┐
                    │   IDLE   │ ← 等待分配
                    └────┬─────┘
                         │ assign(session_id)
                         ▼
                    ┌──────────┐
                    │  ACTIVE  │ ← 处理会话
                    └────┬─────┘
                         │ session_end / timeout / crash
                         ▼
                    ┌──────────┐
                    │ DRAINING │ ← 清理状态（history, memory）
                    └────┬─────┘
                         │ cleanup done
                         ▼
                    ┌──────────┐
                    │   IDLE   │ ← 重新可用
                    └──────────┘
```

### 3.4 分配策略

```python
class WorkerAllocator:
    """最少会话数分配（Least Connections）"""
    
    def allocate(self, session_id: str) -> Worker:
        # 1. 找负载最低的空闲 Worker
        worker = min(self.idle_workers, key=lambda w: w.active_sessions)
        # 2. 如果全满，等待或拒绝
        if worker is None:
            raise NoWorkerAvailable
        # 3. 分配并标记
        worker.assign(session_id)
        return worker
    
    def release(self, worker: Worker):
        worker.drain()  # 清理 session 状态
        self.idle_workers.append(worker)
```

### 3.5 关键实现要点

| 要点 | 说明 |
|------|------|
| **模型共享** | VAD/ASR 模型极小，每个 Worker 独立加载一份（~1.5GB）。TTS 模型 ~5.5GB，每 Worker 一份。GPU 0 可只跑 1 个 Worker（~7GB），GPU 1-7 各跑 1-2 个 |
| **Session 迁移** | Push 模式下不需要——一个会话始终在同一 Worker |
| **优雅关闭** | Orchestrator 发 SIGTERM → Worker 完成当前句子 → drain → 退出 |
| **健康检查** | Orchestrator 定时 heartbeat，Worker 超时 → 标记 dead → 分配新 Worker |
| **Memory 共享** | 如启用 mem0ai，放在 Orchestrator 进程（CPU），Worker 通过 Unix Socket 调 RPC |

---

## 四、实施路线图

### Phase 1 补充（当前优先）

```
□ 1.6 音色一致性修复
  □ 准备真人录音参考文件（assets/voices/*.wav + *.txt）
  □ 启动时 build_prompt_cache() 预编码所有音色
  □ 推理改用 _generate_with_prompt_cache() 替代 generate_streaming()
  □ 验证跨句音色一致性

□ 1.7 推理加速
  □ Warmup：启动后跑一句预热
  □ 有理重采样 1/3 替代 librosa
  □ @torch.inference_mode() 装饰器
  □ 评估 torch.compile（optimize=True）收益

□ 1.8 TTS prompt cache 预编码
  □ 实现 VoiceManager.register_voice() 触发 build_prompt_cache()
  □ VoiceSpec 改为持有 prompt_cache 引用
```

### Phase 2a：Push 模式单机多卡

```
□ 2.1 Worker 进程封装
  □ ProcessWorker 基类（Unix Socket 通信 + 二进制帧协议）
  □ PipelineWorker(GPUWorker) 继承
  □ Worker 生命周期管理（启动/健康检查/优雅关闭）

□ 2.2 Orchestrator 改造
  □ WorkerAllocator（Least Connections 策略）
  □ Session Store（Redis 或内存 dict）
  □ WebSocket → Worker 消息路由

□ 2.3 配置与部署
  □ gpu_alloc.yaml（GPU:Worker 映射）
  □ 单机多 Worker 启动脚本
  □ 优雅关闭与重启

□ 2.4 验证
  □ 2 用户同时对话（不同 Worker）
  □ 打断功能正常
  □ Worker 崩溃不影响其他用户
```

### Phase 2b：Pull 模式（按需）

```
□ 3.1 消息队列引入（Redis/NATS）
□ 3.2 组件池化（ASR Pool / TTS Pool）
□ 3.3 会话状态外部化
□ 3.4 流水线并行调度
```

---

## 五、Push vs Pull 决策总结

| 维度 | Push（会话绑定） | Pull（任务队列） |
|------|-----------------|-----------------|
| 架构复杂度 | ⭐⭐ 低 | ⭐⭐⭐⭐ 高 |
| 代码改动量 | 小（包装现有 PipelineEngine） | 大（全链路重构） |
| GPU 利用率 | 中（会话空闲 = GPU 浪费） | 高（任务级调度） |
| 打断延迟 | ~0ms（进程内 cancel） | ~10-50ms（队列传播） |
| 调试难度 | 低（单进程日志） | 高（分布式追踪需要） |
| 适合规模 | ≤10 并发用户 | ≥50 并发用户 |
| **推荐** | **当前阶段** ✅ | 未来扩展 |
