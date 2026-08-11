#!/usr/bin/env python
"""最小 TTS 音色测试脚本 — 不依赖 FastAPI/WebSocket 等服务层。

用法:
    cd /root/voice-infer-platform
    source /root/VoxEMW/.venv/bin/activate
    PYTHONPATH=src python scripts/test_tts.py

输出: tests/audio/ 下生成 wav 文件 + tests/audio/metrics.json
"""

from __future__ import annotations

import json
import logging
import math
import time
import wave
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("test_tts")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SR = 16000  # 目标采样率
VOICE_ID = "default"

# ── 测试用例 ──────────────────────────────────────────

TEST_CASES = [
    ("short_10", "你好，今天天气不错。"),
    ("normal_30", "今天天气真好，我想出去走走。阳光明媚，微风轻拂，春天真的来了。"),
    ("long_100", "人工智能技术正在深刻改变我们的生活方式。从智能手机到自动驾驶汽车，从医疗诊断到金融分析，"
     "人工智能的应用已经渗透到各个领域。深度学习作为人工智能的核心技术之一，通过多层神经网络模拟人脑的学习过程，"
     "在图像识别、语音识别、自然语言处理等方面取得了突破性进展。"),
    ("question", "你觉得人工智能会取代人类的工作吗？我们应该怎么应对这种变化呢？"),
    ("happy", "太棒了！我们终于成功了！这真是令人振奋的好消息啊！"),
    ("mixed_zh_en", "让我们来学习一下 machine learning 的基本概念。首先需要了解 neural network 的工作原理。"
     "Python 是最流行的 AI 编程语言。"),
]

# ── 连续长文本测试（模拟真实对话） ─────────────────────

LONG_TEXT = (
    "大家好，欢迎收听今天的节目。我们接下来要讨论的话题是人工智能的未来发展。"
    "首先我想说的是，人工智能技术在过去几年里取得了令人瞩目的进步。"
    "从最初的简单规则系统到现在的大语言模型，技术演进的步伐从未停止。"
    "以 ChatGPT 为代表的对话式人工智能已经能够理解复杂的自然语言指令。"
    "在医疗领域，人工智能辅助诊断系统已经能够准确识别多种疾病的影像特征。"
    "在金融行业，智能风控系统每天处理数以亿计的交易数据。"
    "当然，人工智能也带来了一些挑战，比如就业结构的变化和隐私保护的问题。"
    "我们需要以开放和审慎的态度来面对这些变化。"
    "谢谢大家的收听，我们下期再见。"
)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int = SR):
    """保存 float32 [-1,1] 音频为 16-bit WAV。"""
    pcm = np.clip(audio, -1.0, 1.0)
    i16 = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(i16.tobytes())


def load_tts():
    """加载 TTS 引擎（复用项目现有代码）。"""
    from voice_infer.engine.tts.voxcpm2 import VoxCPM2TTS, VoiceSpec

    config = {
        "model_path": "/root/.cache/modelscope/models/openbmb--VoxCPM2/snapshots/master",
        "device": "cuda",
        "sample_rate": SR,
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "voices": {},
    }

    # 尝试加载已有音色
    voices = {}
    default_ref = REPO / "assets" / "default" / "ref.wav"
    if default_ref.is_file():
        ref_text_path = REPO / "assets" / "default" / "ref.txt"
        ref_text = ref_text_path.read_text("utf-8").strip() if ref_text_path.is_file() else ""
        voices["default"] = VoiceSpec("default", str(default_ref), ref_text)

    my_ref = REPO / "data" / "voices" / "my_voice" / "ref.wav"
    if my_ref.is_file():
        ref_text_path = REPO / "data" / "voices" / "my_voice" / "ref.txt"
        ref_text = ref_text_path.read_text("utf-8").strip() if ref_text_path.is_file() else ""
        voices["my_voice"] = VoiceSpec("my_voice", str(my_ref), ref_text)

    tts = VoxCPM2TTS(**config)
    for v in voices.values():
        tts.register_voice(v)
    tts.load_model()
    return tts


def test_one(tts, name: str, text: str, voice_id: str = VOICE_ID) -> dict:
    """合成一句文本，返回指标。"""
    logger.info("─" * 40)
    logger.info("Test: %s | text=%s | voice=%s", name, text[:60], voice_id)

    torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    t0 = time.perf_counter()
    chunks = []
    first_chunk_at = None

    for audio_chunk in tts._stream(text, voice_id):
        a = np.asarray(audio_chunk).astype(np.float32).squeeze()
        if a.size == 0: continue
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter() - t0
        chunks.append(a)

    t1 = time.perf_counter()
    total_time = t1 - t0

    if not chunks:
        logger.warning("No audio generated!")
        return {}

    audio = np.concatenate(chunks)
    duration = len(audio) / SR
    rtf = total_time / duration if duration > 0 else float("inf")
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # 保存
    wav_path = OUT / f"{name}_{voice_id}.wav"
    save_wav(wav_path, audio)
    logger.info("  TTFA=%.3fs  total=%.3fs  duration=%.2fs  RTF=%.2f  RMS=%.4f  samples=%d",
                first_chunk_at or 0, total_time, duration, rtf, rms, len(audio))
    logger.info("  saved: %s", wav_path)

    return {
        "name": name, "voice_id": voice_id,
        "ttfa_s": round(first_chunk_at, 3) if first_chunk_at else None,
        "total_s": round(total_time, 3),
        "duration_s": round(duration, 3),
        "rtf": round(rtf, 3),
        "rms": round(rms, 6),
        "samples": len(audio),
        "wav": str(wav_path),
    }


def main():
    logger.info("=" * 60)
    logger.info("TTS 音色测试脚本")
    logger.info("=" * 60)

    tts = load_tts()

    results = []

    # 1. 基础测试用例
    for name, text in TEST_CASES:
        metrics = test_one(tts, name, text, "default")
        if metrics: results.append(metrics)

    # 2. 多音色对比（如果 my_voice 存在）
    if "my_voice" in tts.voice_prompts:
        logger.info("\n=== 多音色对比 ===")
        text = "今天天气真好，我想出去走走。"
        for vid in ["default", "my_voice"]:
            m = test_one(tts, f"voice_cmp_{vid}", text, vid)
            if m:
                m["name"] = f"voice_cmp_{vid}"
                results.append(m)

    # 3. 同文本多次合成（稳定性测试）
    logger.info("\n=== 稳定性测试（同文本 × 3） ===")
    text = "人工智能正在改变世界。"
    for i in range(3):
        m = test_one(tts, f"stability_{i+1}", text, "default")
        if m:
            m["name"] = f"stability_{i+1}"
            results.append(m)

    # 4. 连续长文本（整段一次性合成）
    logger.info("\n=== 长文本整段合成 ===")
    m = test_one(tts, "long_continuous", LONG_TEXT, "default")
    if m: results.append(m)

    # ── 保存指标 ──────────────────────────────────────

    metrics_path = OUT / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("\n=== 指标汇总 ===")
    for r in results:
        logger.info("  %-25s TTFA=%.3fs  RTF=%.2f  dur=%.2fs  RMS=%.4f",
                    r["name"], r.get("ttfa_s", 0), r["rtf"], r["duration_s"], r["rms"])
    logger.info("Metrics saved: %s", metrics_path)


if __name__ == "__main__":
    main()
