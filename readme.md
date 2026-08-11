## 语音服务平台模型设计

单进程全链路 Pipeline：`VAD(Silero) → ASR(SenseVoiceSmall) → LLM(DeepSeek v4-flash) → TTS(VoxCPM2)`，FastAPI + WebSocket 双队列架构。

- VAD 检测用户说完 → ASR 转写 → LLM 流式生成回复 → TTS 逐句合成语音 → 浏览器播放
- WebSocket 双队列：`receiver → pcm_queue → turn_worker → out_queue → sender`
- 打断：epoch 递增 + asyncio.Event 传入 TTS，每 chunk 检查，立即停止
- 音色：default 走 zero-shot，自定义音色走 Ultimate Cloning（`build_prompt_cache` 预编码）

## 优化手段

**音色连续性**
- prompt_cache 预编码：真人录音 → 启动时 GPU latent 编码 → 推理复用，跨句音色一致
- AudioWorklet RingBuffer 播放：512 采样小块 → 10s 环形缓冲区，零调度间隙

**推理加速**
- `torch.compile`（mode=reduce-overhead）：RTF 0.6→0.2，启动 +90s
- warmup：启动后预热一句，消除首句冷启动
- `@torch.inference_mode()`：禁用 autograd，省显存
- 有理重采样 `resample_poly(up=1, down=3)`：48k→16k 最小计算量

**流式 + 延迟**
- LLM+TTS 并发生成：LLM 每句 `asyncio.create_task` 提后台 TTS，不阻塞后续句子
- 流式输出：`_stream()` 攒 ~0.5s 大块即发，首音延迟 TTFA ≈ 100ms（TTS 侧）
- `await asyncio.sleep(0)`：每 chunk 释放事件循环，interrupt/WS 即时响应
- AudioWorklet 麦克风：128 采样(~8ms)替代 ScriptProcessor 4096(~256ms)

## Memory 设计

三层 memory，我们目前只接入了第一层：

| 层级 | 名称 | 实现 | 状态 |
|------|------|------|------|
| **L1** | 短期对话历史 | `PipelineEngine._history`，内存 dict，60 轮上限 | ✅ 已启用 |
| **L2** | 长期记忆 | Mem0 + BGE-M3 + Qdrant，自动抽取用户偏好/事实 | ⚠️ 代码已就绪，未接入管线 |
| **L3** | 知识库 RAG | PDF→切块→BGE-M3 嵌入→SQLite 余弦检索→注入 LLM | ❌ 仅有配置壳 |

**当前实际生效**：仅 L1。LLM 每次请求携带最近 30 轮对话历史作为上下文。

**待接入**（参照 VoxEMW）：
- L2：会话建立时 `recall(agent_id)` → 注入 instructions；每轮结束 `remember(user, assistant)` 异步写
- L3：每轮 LLM 前检索知识库 → 命中后注入 `"# 参考资料\n..."`

## Push vs Pull（单机多卡设计）

**Push 模式（推荐先做）**：会话绑定 Worker
- Orchestrator 分配空闲 GPU Worker → Worker 持有完整管线 → 会话结束释放
- 优势：架构简单，会话亲和（history/memory 在 Worker 内存），打断低延迟
- 劣势：GPU 利用率中（会话空闲=浪费），扩容需加 GPU

**Pull 模式（后续按需）**：组件池 + 任务队列
- ASR Pool / TTS Pool 独立扩缩，任务级调度，负载均衡
- 优势：GPU 利用率高，组件独立伸缩
- 劣势：架构复杂（消息队列/状态外化/分布式追踪），打断延迟高
