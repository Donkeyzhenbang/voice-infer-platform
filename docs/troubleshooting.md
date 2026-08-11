# TTS 音质问题排障记录

> 引擎：VoxCPM2（openbmb/VoxCPM2）  \
> 设备：RTX 4090D 24GB，CUDA 12.8

---

## 问题总览

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| 1 | TTS 输出开头播放参考音频 | `prompt_wav_path` 传入了 ref.wav，VoxCPM2 会先播放参考音频再生成 | 去掉 `prompt_wav_path` |
| 2 | 尾音有"哦""啊"等多余音节 + 背景杂音 | `reference_wav_path` 用了机器生成的 ref.wav，模型克隆机器音 → 反馈循环劣化 | 去掉所有参考音频，用内置默认声音 |
| 3 | 多句 TTS 只有最后一句有声音 | 每句都发 `audio_start`，前端重置播放队列 | 整轮只发一次 `audio_start/end` |
| 4 | 音色偶尔不一致 | CUDA 非确定性 + `cfg_value=2.0` 在无参考音频时过度放大文本差异 | `cudnn.deterministic=True` + `cfg_value` 降到 1.0 |

---

## 详细分析

### 问题 1：TTS 输出开头播放参考音频

**现象**：生成的语音开头有一段不属于生成内容的音频。

**原因**：VoxCPM2 的 `generate_streaming(prompt_wav_path=...)` 会把参考音频拼接在生成结果前面，用于"提示"模型。这不是我们需要的——我们只需要生成的语音，不需要播放参考音频。

**修复**：彻底去掉 `prompt_wav_path` 参数。不用参考音频提示，直接用内置默认声音生成。

### 问题 2：尾音"哦啊"和背景杂音

**现象**：生成的语音末尾偶尔出现"哦""啊"等无意义音节，同时有明显的背景噪音。

**原因**：
- 我们用 VoxCPM2 自身生成了一个 `ref.wav` 作为参考音频
- 机器生成的音频本身就有尾音残留和 mild artifacts
- 把这个不完美的输出当作"参考"再输入给模型 → 形成劣化反馈循环
- 扩散模型在参考音频的影响下，不知道在哪里停，产生多余音节

**修复**：彻底去掉 `reference_wav_path`。VoxCPM2 内置默认声音（独立测试验证：2.4s 干净输出，RMS=0.14，无杂音）。

### 问题 3：多句 TTS 截断

**现象**：LLM 输出多句话时，前端只播放最后一句，前面的句子没声音。

**原因**：
- 每一句 TTS 的 `is_first=True` chunk 都会告诉前端"新音频开始了"
- 前端的 `audio` 元素接到新 `src` 会重置播放队列
- 最后一句覆盖了前面所有句子的播放

**修复**：Pipeline 层做"整轮"语义封装——只在第一句的第一个 TTS chunk 前发 `audio_start`，整轮结束只发一次 `audio_end`。中间的 TTS chunk 全部设 `is_first=False`。

### 问题 4：音色偶尔不一致

**现象**：同一会话中，不同句子的语音音色有时明显不同。

**原因**（两个叠加因素）：

1. **CUDA 非确定性**：`torch.manual_seed(42)` 不够——GPU 上的浮点 reduce/scan 操作默认不保证运算顺序一致性。每次推理的 noise sample 虽然种子相同，但后续计算路径因并行度不同可能产生微小差异，在扩散模型中累积放大。

2. **cfg_value 偏高**：无参考音频时，VoxCPM2 只有一个"默认声音"分布。CFG（Classifier-Free Guidance）公式为：
   ```
   output = unconditional + cfg_value × (conditional - unconditional)
   ```
   当 `cfg_value=2.0` 时，模型被强力推向文本条件方向。不同文本 → 不同推力方向 → 音色漂移。

**修复**：
- `torch.backends.cudnn.deterministic = True` + `cudnn.benchmark = False`
- `cfg_value` 从 2.0 降到 1.0（CFM 模块原生默认值，无条件分支不加权，音色最稳定）

---

## 关键经验

1. **不要用模型自己的输出做参考**：这是"用有损 MP3 重新编码"式的反馈循环。
2. **无参考 TTS 的 cfg_value 不宜高**：参考音频缺失时，CFG 只在文本和"空提示"之间插值，值越高越不稳定。
3. **CUDA 确定性很重要**：扩散模型的随机性叠加 GPU 并行非确定性 = 每次输出微妙不同。`cudnn.deterministic=True` 是必要的。
4. **前端播放队列不假设后端语义**：浏览器 `<audio>` 的 `src` 切换是破坏性的。需要后端保证"一轮对话 = 一个音频流"的语义。
