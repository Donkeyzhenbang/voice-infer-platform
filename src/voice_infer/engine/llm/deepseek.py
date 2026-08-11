"""LLM 引擎：DeepSeek — 关 thinking + 过滤推理内容 + 复用 client。"""

from __future__ import annotations

import json, logging, os, re
from typing import AsyncIterator

import httpx

from voice_infer.common.schema import LLMResponse
from voice_infer.engine.interfaces import LLMEngine

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"[。！？!?\n]")
# 括号动作过滤（对齐 VoxEMW）：（笑）（拍大腿）等不让 TTS 朗读
_PAREN_ACTION = re.compile(r"[（(][^（）()]{1,20}[)）]")


class DeepSeekLLM(LLMEngine):
    def __init__(self, model="deepseek-v4-flash", base_url="https://api.deepseek.com/v1",
                 api_key="", api_key_env="DEEPSEEK_API_KEY", max_tokens=2048,
                 temperature=0.7, system_prompt=""):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        # 复用 client（review 建议）
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    async def generate(self, user_text, session_id, history=None, instructions=None):
        messages = []
        sys = instructions or self.system_prompt
        if sys: messages.append({"role": "system", "content": sys})
        if history: messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        # thinking 放顶层 JSON（不用 extra_body——那是 OpenAI SDK 概念）
        body = {
            "model": self.model, "messages": messages,
            "max_tokens": self.max_tokens, "temperature": self.temperature,
            "stream": True,
            "thinking": {"type": "disabled"},  # 顶层，DeepSeek 直接识别
        }

        async with self._client.stream(
            "POST", f"{self.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            import uuid; turn_id = uuid.uuid4().hex[:12]
            buf = ""

            async for line in resp.aiter_lines():
                if not line.startswith("data: "): continue
                d = line[6:]
                if d == "[DONE]": break
                try: data = json.loads(d)
                except: continue
                try: delta = data["choices"][0]["delta"]
                except: continue

                # 只取 content，忽略 reasoning_content（思考过程）
                content = delta.get("content", "")
                if content:
                    buf += content

                # 按标点分句
                while True:
                    m = _SENTENCE_END.search(buf)
                    if not m: break
                    sentence = buf[:m.end()].strip()
                    buf = buf[m.end():]
                    if sentence:
                        # 过滤括号动作
                        clean = _PAREN_ACTION.sub(" ", sentence).strip()
                        if clean:
                            yield LLMResponse(session_id=session_id, turn_id=turn_id,
                                              text=clean, is_final=False)

            # 收尾
            tail = _PAREN_ACTION.sub(" ", buf).strip()
            if tail:
                yield LLMResponse(session_id=session_id, turn_id=turn_id, text=tail, is_final=True)
