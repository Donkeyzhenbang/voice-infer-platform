#!/usr/bin/env python
"""全链路音频质量测试 — 检查 chunk 拼接、转换精度、卡顿/爆音根因。

用法:
    cd /root/voice-infer-platform
    source /root/VoxEMW/.venv/bin/activate
    PYTHONPATH=src python scripts/test_audio_quality.py
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SR = 16000

# ═══════════════════════════════════════════════════════════
# 测试 1: 后端 float32 → int16 → float32 往返精度
# ═══════════════════════════════════════════════════════════

def test_roundtrip_precision():
    """验证 float32→int16→float32 转换是否引入可听失真。"""
    print("\n=== Test 1: Roundtrip Precision ===")

    # 模拟一个 512-sample block（32ms @ 16kHz）
    np.random.seed(42)
    original = np.random.randn(512).astype(np.float32) * 0.3  # -10dB 左右的信号
    original = np.clip(original, -1.0, 1.0)

    # 当前代码的转换路径
    i16 = np.clip(original * 32767, -32768, 32767).astype(np.int16)
    back = i16.astype(np.float32) / 32767.0

    diff = original - back
    max_err = abs(diff).max()
    rms_err = math.sqrt(np.mean(diff ** 2))

    print(f"  max error: {max_err:.6f}  ({20*math.log10(max_err+1e-10):.1f} dB)")
    print(f"  rms error: {rms_err:.6f}  ({20*math.log10(rms_err+1e-10):.1f} dB)")

    # 检查是否有样本被截断
    clipped = (abs(original) > 0.999).sum()
    print(f"  clipped samples: {clipped}")

    return {"max_err": float(max_err), "rms_err": float(rms_err), "clipped": int(clipped)}


# ═══════════════════════════════════════════════════════════
# 测试 2: chunk 边界连续性
# ═══════════════════════════════════════════════════════════

def test_chunk_continuity():
    """验证 _stream() 产出的多个 chunk 拼接后是否在边界处有跳变。"""
    print("\n=== Test 2: Chunk Boundary Continuity ===")

    from voice_infer.engine.tts.voxcpm2 import VoxCPM2TTS

    tts = VoxCPM2TTS(
        "/root/.cache/modelscope/models/openbmb--VoxCPM2/snapshots/master",
        atempo_rate=1.0,
    )
    tts.load_model()

    # 生成一段较长文本，收集所有 chunk
    text = "今天天气真好，阳光明媚，微风轻拂，春天真的来了。"
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    chunks = []
    for c in tts._stream(text, "default"):
        a = np.asarray(c).astype(np.float32).squeeze()
        if a.size > 0:
            chunks.append(a.copy())

    if len(chunks) < 2:
        print("  WARNING: only 1 chunk, can't test boundaries")
        return {}

    # 拼接
    full = np.concatenate(chunks)

    # 检查每个 chunk 边界处的跳变
    boundaries = []
    pos = 0
    for i in range(len(chunks) - 1):
        pos += len(chunks[i])
        if pos >= len(full) - 1:
            break
        # 边界处两个相邻采样的差
        jump = abs(float(full[pos]) - float(full[pos - 1]))
        boundaries.append({"pos": pos, "jump": float(jump)})

    max_jump = max(b["jump"] for b in boundaries)
    avg_jump = sum(b["jump"] for b in boundaries) / len(boundaries)

    # 正常语音相邻采样差通常 < 0.1，如果 > 0.5 说明有明显跳变
    print(f"  chunks: {len(chunks)}")
    print(f"  max boundary jump: {max_jump:.4f}")
    print(f"  avg boundary jump: {avg_jump:.4f}")
    print(f"  suspicious jumps (>0.3): {sum(1 for b in boundaries if b['jump']>0.3)}")

    # 保存完整音频到文件
    path = OUT / "continuity_test.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(full, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"  saved: {path}")

    return {"chunks": len(chunks), "max_jump": float(max_jump), "avg_jump": float(avg_jump)}


# ═══════════════════════════════════════════════════════════
# 测试 3: int16 字节级验证
# ═══════════════════════════════════════════════════════════

def test_int16_bytes():
    """验证 synthesize() 产出的 int16 bytes 是否能正确解码。"""
    print("\n=== Test 3: Int16 Bytes Integrity ===")
    import asyncio
    from voice_infer.engine.tts.voxcpm2 import VoxCPM2TTS

    tts = VoxCPM2TTS(
        "/root/.cache/modelscope/models/openbmb--VoxCPM2/snapshots/master",
        atempo_rate=1.0,
    )
    tts.load_model()

    async def collect():
        chunks = []
        async for c in tts.synthesize("你好世界。", "default", "test", "t1"):
            if c.audio:
                chunks.append(c.audio)
        return chunks

    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    audio_bytes = asyncio.run(collect())

    if not audio_bytes:
        print("  ERROR: no audio produced!")
        return {}

    # 解码所有 int16 bytes
    all_samples = []
    for i, ab in enumerate(audio_bytes):
        samples = np.frombuffer(ab, dtype=np.int16).astype(np.float32) / 32767.0
        all_samples.append(samples)
        if len(samples) != 512:
            print(f"  WARNING: chunk {i} has {len(samples)} samples (expected 512)")

    full = np.concatenate(all_samples)
    rms = math.sqrt(np.mean(full ** 2))
    peak = abs(full).max()

    print(f"  total chunks: {len(audio_bytes)}")
    print(f"  total samples: {len(full)}")
    print(f"  duration: {len(full)/SR:.2f}s")
    print(f"  RMS: {rms:.4f}  peak: {peak:.3f}")

    # 保存
    path = OUT / "int16_test.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(full, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"  saved: {path}")

    return {"chunks": len(audio_bytes), "samples": int(len(full)), "rms": float(rms), "peak": float(peak)}


# ═══════════════════════════════════════════════════════════
# 测试 4: 静音段检测（找爆音/click）
# ═══════════════════════════════════════════════════════════

def test_click_detection(wav_path: str | None = None):
    """检测音频中的突变（可能的 click/pop）。"""
    print("\n=== Test 4: Click Detection ===")

    if wav_path is None:
        wav_path = OUT / "continuity_test.wav"

    if not Path(wav_path).is_file():
        print(f"  file not found: {wav_path}")
        return {}

    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0

    # 找连续两个采样之间的异常跳变
    diffs = np.abs(np.diff(audio))
    threshold = 0.2  # 正常语音 diff < 0.1，>0.2 可能是 click
    clicks = np.where(diffs > threshold)[0]

    # 排除真正的语音瞬态（辅音爆破音 t/p/k 等）
    # 简单策略：连续 3 个以上异常点才算 click
    click_regions = []
    if len(clicks) > 0:
        start = clicks[0]
        for i in range(1, len(clicks)):
            if clicks[i] - clicks[i-1] > 3:
                if clicks[i-1] - start >= 3:
                    click_regions.append((int(start), int(clicks[i-1])))
                start = clicks[i]
        if clicks[-1] - start >= 3:
            click_regions.append((int(start), int(clicks[-1])))

    print(f"  audio: {len(audio)} samples, {len(audio)/sr:.2f}s")
    print(f"  high-diff points (>0.2): {len(clicks)}")
    print(f"  click regions (>=3 consecutive): {len(click_regions)}")

    # 检查每 512 采样边界处是否有 click（chunk 边界问题）
    block = 512
    boundary_clicks = []
    for pos in range(block, len(audio), block):
        if pos >= len(audio) - 1:
            break
        jump = abs(float(audio[pos]) - float(audio[pos-1]))
        if jump > 0.1:
            boundary_clicks.append({"pos": pos, "jump": float(jump)})

    print(f"  chunk-boundary clicks (>0.1): {len(boundary_clicks)}")
    if boundary_clicks:
        for bc in boundary_clicks[:5]:
            print(f"    pos={bc['pos']} jump={bc['jump']:.4f}")

    return {
        "high_diff_points": int(len(clicks)),
        "click_regions": int(len(click_regions)),
        "boundary_clicks": int(len(boundary_clicks)),
    }


# ═══════════════════════════════════════════════════════════
# 测试 5: 端到端 — 模拟 WS 传输全链路
# ═══════════════════════════════════════════════════════════

def test_e2e_pipeline():
    """端到端：从 TTS 生成 → int16 bytes → 模拟前端解码 → 保存 WAV。"""
    print("\n=== Test 5: End-to-End Pipeline ===")
    import asyncio
    from voice_infer.engine.tts.voxcpm2 import VoxCPM2TTS

    tts = VoxCPM2TTS(
        "/root/.cache/modelscope/models/openbmb--VoxCPM2/snapshots/master",
        atempo_rate=1.0,
    )
    tts.load_model()

    async def e2e():
        all_bytes = b""
        async for c in tts.synthesize("今天天气真好，我想出去走走。阳光明媚，微风轻拂，春天真的来了。",
                                       "default", "test", "t1"):
            if c.audio:
                all_bytes += c.audio
        return all_bytes

    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    raw = asyncio.run(e2e())

    # 模拟前端解码
    float_samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0

    # 检查是否有异常值
    clipped = (abs(float_samples) > 0.999).sum()
    zeros = (float_samples == 0).sum()

    path = OUT / "e2e_test.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(raw)  # 直接写原始 int16 bytes!

    print(f"  total bytes: {len(raw)}")
    print(f"  samples: {len(float_samples)}")
    print(f"  duration: {len(float_samples)/SR:.2f}s")
    print(f"  RMS: {math.sqrt(np.mean(float_samples**2)):.4f}")
    print(f"  peak: {abs(float_samples).max():.3f}")
    print(f"  clipped: {clipped}")
    print(f"  zero samples: {zeros}")
    print(f"  saved: {path}")

    return {
        "bytes": len(raw), "samples": int(len(float_samples)),
        "duration": round(len(float_samples)/SR, 2),
        "rms": round(float(math.sqrt(np.mean(float_samples**2))), 4),
        "peak": round(float(abs(float_samples).max()), 3),
        "clipped": int(clipped), "zeros": int(zeros),
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Audio Quality Test Suite")
    print("=" * 60)

    results = {}

    results["roundtrip"] = test_roundtrip_precision()
    results["continuity"] = test_chunk_continuity()
    results["int16"] = test_int16_bytes()
    results["clicks"] = test_click_detection()
    results["e2e"] = test_e2e_pipeline()

    # 总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    has_issues = False

    rt = results.get("roundtrip", {})
    if rt.get("rms_err", 0) > 0.0001:
        print("⚠  float32↔int16 precision loss > -80dB")
        has_issues = True

    ct = results.get("continuity", {})
    if ct.get("max_jump", 0) > 0.3:
        print(f"⚠  chunk boundary jump: {ct['max_jump']:.4f} — audible click!")
        has_issues = True
    else:
        print(f"✅ chunk boundaries smooth (max jump: {ct.get('max_jump', 0):.4f})")

    ck = results.get("clicks", {})
    if ck.get("boundary_clicks", 0) > 0:
        print(f"⚠  {ck['boundary_clicks']} boundary clicks detected")
        has_issues = True

    e2 = results.get("e2e", {})
    if e2.get("clipped", 0) > 10:
        print(f"⚠  {e2['clipped']} clipped samples — possible gain issue")
        has_issues = True
    if e2.get("zeros", 0) > 100:
        print(f"⚠  {e2.get('zeros', 0)} zero samples — possible gap")
        has_issues = True

    if not has_issues:
        print("✅ All tests PASS — audio pipeline is clean")

    # 保存结果
    (OUT / "quality_test.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved: {OUT / 'quality_test.json'}")
    print(f"Audio files: {OUT}/")


if __name__ == "__main__":
    main()
