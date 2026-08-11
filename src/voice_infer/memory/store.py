"""Memory 模块：Mem0 长期记忆（可选，需 bge-m3）。"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryStore:
    """Mem0 封装：LLM 抽取事实 + bge-m3 embedding + Qdrant 存储。

    只在 memory.enabled=true 时启用，缺失依赖静默降级。
    """

    def __init__(self, config: dict, llm_config: dict, api_key: str):
        self.enabled = config.get("enabled", False)
        if not self.enabled:
            logger.info("Memory disabled")
            return

        store_dir = Path(config.get("store_dir", "data/memory"))
        store_dir.mkdir(parents=True, exist_ok=True)
        self.user_id = "default_user"
        self.top_k = config.get("top_k", 5)
        embedder_path = config.get("embedder_model_path", "")

        try:
            from mem0 import Memory

            self._m = Memory.from_config({
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": llm_config.get("model", "deepseek-v4-flash"),
                        "openai_base_url": llm_config.get("base_url", "https://api.deepseek.com/v1"),
                        "api_key": api_key,
                        "temperature": 0.0,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {"model": embedder_path},
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "voice_infer_memory",
                        "path": str(store_dir),
                        "embedding_model_dims": 1024,
                    },
                },
            })
            logger.info("Memory initialized: store=%s, top_k=%d", store_dir, self.top_k)
        except Exception as e:
            logger.warning("Memory init failed (non-fatal): %s", e)
            self.enabled = False

    def recall(self, agent_id: str = "default") -> list[str]:
        """召回该 persona 相关的记忆。"""
        if not self.enabled:
            return []
        try:
            results = self._m.search(
                "用户的重要事实与偏好", top_k=self.top_k,
                filters={"user_id": self.user_id, "agent_id": agent_id},
            )
            return [r["memory"] for r in (results.get("results") or []) if r.get("memory")]
        except Exception as e:
            logger.warning("Memory recall failed: %s", e)
            return []

    def remember(self, user_text: str, assistant_text: str, agent_id: str = "default") -> None:
        """异步记住一轮对话。"""
        if not self.enabled:
            return
        try:
            msgs = []
            if user_text:
                msgs.append({"role": "user", "content": user_text})
            if assistant_text:
                msgs.append({"role": "assistant", "content": assistant_text})
            if msgs:
                self._m.add(msgs, user_id=self.user_id, agent_id=agent_id,
                            prompt="只记录用户的事实/偏好/计划。不记助手内容。")
        except Exception as e:
            logger.warning("Memory save failed: %s", e)


def create_memory_store(pipeline_config, api_key: str) -> MemoryStore:
    """从 pipeline 配置创建 MemoryStore。"""
    mem_cfg = pipeline_config.memory if hasattr(pipeline_config, 'memory') else {}
    llm_cfg = pipeline_config.llm if hasattr(pipeline_config, 'llm') else {}
    llm_dict = {"model": getattr(llm_cfg, 'model', 'deepseek-v4-flash'),
                "base_url": getattr(llm_cfg, 'base_url', 'https://api.deepseek.com/v1')}
    return MemoryStore(
        config={"enabled": getattr(mem_cfg, 'enabled', False),
                "store_dir": getattr(mem_cfg, 'store_dir', 'data/memory'),
                "top_k": getattr(mem_cfg, 'top_k', 5),
                "embedder_model_path": getattr(mem_cfg, 'embedder_model_path', '')},
        llm_config=llm_dict,
        api_key=api_key,
    )
