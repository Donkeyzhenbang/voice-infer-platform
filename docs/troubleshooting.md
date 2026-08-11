# Voice Infer Platform — 优化记录

> 引擎：VoxCPM2 + DeepSeek v4-flash  
> 设备：RTX 4090D 24GB，CUDA 12.8  
> 日期：2026-08-11

---

## 一、音色一致性

### 1.1 Ultimate Cloning + Prompt Cache

**问题**：无参考音频时 VoxCPM2 没有音色锚点，每次从 `z = randn()` 随机涌现音色，不同句子音色不一致。

**修复**：
- 启动时 `build_prompt_cache()` 一次性编码参考音频为 GPU latent features
- 推理用 `_generate_with_prompt_cache()` 复用 cache，同一音色所有句子共享
- `register_voice()` 实时触发预编码，不等到重启

```yaml
tts:
  cfg_value: 2.0         # 文本引导强度，官方推荐值
  inference_timesteps: 10
  atempo_rate: 0.886      # 仅 prompt_cache 模式生效
```

### 1.2 default 音色的特殊处理

**问题**：`assets/default/ref.wav` 是机器生成的，用它做 `build_prompt_cache()` 等于"用有损 MP3 反复重编码"→ artifacts 反馈循环劣化。

**修复**：`default` 音色跳过 prompt_cache 编码，走 zero-shot（无参考，纯文本引导生成）。只有真人录制的声音才走 Ultimate Cloning。

### 1.3 参考音频预处理

- `trim_silence_vad=True`：VoxCPM2 内置 VAD 裁切首尾静音
- 峰值归一化到 -1dB
- 前端录制：`noiseSuppression` + 噪声门(-30dB) + 归一化(-3dB)

### 1.4 cfg_value 机制

```
v_cfg = v_uncond + cfg_value × (v_cond − v_uncond)
```
- CFG 两个分支的 `cond`（音频上下文）完全相同
- 唯一区别是 `mu`（文本语义编码）
- `cfg_value` 控制文本语义引导强度，不直接控制音色
- `cfg_value=2.0`：文本跟随紧，结构稳定，间接让音色一致

---

## 二、推理加速

### 2.1 torch.compile

```yaml
optimize: true  # RTF 0.6→0.2，启动+90s，显存+1GB
```
编译 `base_lm.forward_step`、`residual_lm.forward_step`、`feat_encoder`、`feat_decoder.estimator`。

### 2.2 Warmup

启动后跑一句预热，初始化 CUDA kernel 缓存，消除首句冷启动延迟。

### 2.3 其他

| 优化 | 效果 |
|------|------|
| `@torch.inference_mode()` | 禁用 autograd，省显存+加速 |
| 有理重采样 `resample_poly(up=1, down=3)` | 48k→16k 最小计算量 |
| prompt_cache 预编码 | 每句跳过 ~200ms 音频编码 |

---

## 三、流式 + 延迟

### 3.1 LLM + TTS 并发生成

**之前（串行）**：LLM出句1 → TTS阻塞2-10s → LLM出句2 → TTS又阻塞 → ...

**现在（并发）**：LLM每句通过 `asyncio.create_task` 提后台TTS，`pending` dict 按序输出。

### 3.2 流式输出

`_stream()` 攒 ~0.5s 大块立即 yield，`synthesize()` 不累积整句。

延迟链：说完了→VAD(0)→ASR(60ms)→LLM首token(300ms)→TTS首块(100ms)→播放，TTFA ≈ 460ms。

### 3.3 asyncio 不阻塞

`synthesize()` 每 chunk 后 `await asyncio.sleep(0)` 释放事件循环，interrupt/WS 即时响应。

---

## 四、音频质量

### 4.1 消除 float32↔int16 乒乓转换

**之前**：float32→int16→float32→int16（4次转换，155个32ms小块）

**现在**：`_stream()` 直出 float32 → `synthesize()` 一次转 int16 → 10个大块。max boundary jump 0.11→0.02。

### 4.2 AudioWorklet + RingBuffer 播放

**之前**：每 32ms chunk 独立 `createBufferSource().start()` → 调度间隙→爆音

**现在**：AudioWorklet 环形缓冲区(10s)，主线程 `postMessage` 写入，音频线程连续读取→零间隙。

### 4.3 AudioWorklet 麦克风

**之前**：`ScriptProcessorNode` 4096采样(~256ms延迟)

**现在**：`AudioWorkletNode` 128采样(~8ms延迟)，VAD更快感知用户说完。

---

## 五、打断机制

| 层级 | 机制 |
|------|------|
| Pipeline | `_epoch[sid]` 递增，`_is_stale()` 每步检查 |
| TTS | `asyncio.Event` 传入 `_stream()`，每chunk检查 |
| 前端 | 打断按钮→WS `interrupt`→cancel()→epoch+++event.set() |

---

## 六、前端优化

| 优化 | 之前 | 之后 |
|------|------|------|
| 音色下拉框 | loadVoices()重建时重置选中 | 保留cur值 |
| 文本清洗 | 无 | 过滤括号动作（笑）（拍大腿） |
| 录音降噪 | 无 | noiseSuppression+噪声门+归一化 |

---

## 七、已知限制

| 限制 | 说明 |
|------|------|
| default无固定音色 | zero-shot每句随机涌现，需真人录音替换ref.wav |
| 无speculative VAD reopen | 短暂停顿被误判说完 |
| LLM API延迟不可控 | DeepSeek ~300-500ms首token |
| 单GPU | 无法服务多用户并发 |

---

## 八、关键经验

1. **不要用模型自己的输出做参考**：机器生成ref.wav → artifacts反馈循环
2. **无参考TTS音色随机**：VoxCPM2无内置音色，从randn()涌现。真人录音+prompt_cache才能锁定
3. **cfg_value控制文本引导非音色**：CFG两分支仅在mu不同，无参考时不宜低于2.0
4. **小块播放必然爆音**：createBufferSource调度有间隙，需AudioWorklet RingBuffer
5. **TTS不能阻塞LLM**：串行浪费LLM流式能力，需asyncio.create_task并发
6. **float32↔int16别来回转**：一次转换省CPU且消除量化累积误差
