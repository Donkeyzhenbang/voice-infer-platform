"""知识库 RAG：SQLite + BGE-M3 + 余弦检索。

轻量实现，不用 Qdrant（避免单进程锁冲突）。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

EMBED_DIM = 1024


class KnowledgeStore:
    """SQLite 知识库：文本切块 → BGE-M3 嵌入 → 余弦检索。"""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        if not self.enabled:
            logger.info("Knowledge disabled")
            return

        self.db_path = Path(config.get("db_path", "data/knowledge.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.top_k = config.get("top_k", 3)
        self.threshold = config.get("threshold", 0.35)
        self.embedder_path = config.get("embedder_model_path", "")
        self._model = None
        self._lock = threading.Lock()

        self._init_db()
        if self.embedder_path:
            self._load_model()
        logger.info("Knowledge initialized: db=%s, top_k=%d, threshold=%.2f",
                    self.db_path, self.top_k, self.threshold)

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_name TEXT NOT NULL,
                    chunk_idx INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc ON chunks(doc_name)")

    def _load_model(self):
        try:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.embedder_path)
            self._model = AutoModel.from_pretrained(self.embedder_path)
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            logger.info("Knowledge embedder loaded: %s", self.embedder_path)
        except Exception as e:
            logger.warning("Knowledge embedder load failed (non-fatal): %s", e)
            self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        """BGE-M3 嵌入，返回归一化向量。"""
        if self._model is None:
            raise RuntimeError("Embedder not loaded")
        with torch.no_grad():
            inputs = self._tokenizer(
                texts, padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            )
            if next(self._model.parameters()).is_cuda:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = self._model(**inputs)
            # BGE-M3: 取 CLS token + 归一化
            vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / (norms + 1e-8)
        return np.array([])

    def add_document(self, doc_name: str, text: str, chunk_size: int = 300, overlap: int = 50):
        """添加文档：切块 → 嵌入 → 写入 SQLite。"""
        if not self.enabled or self._model is None:
            return

        # 切块
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
            i += chunk_size - overlap
        if not chunks:
            return

        # 嵌入
        vecs = self.embed(chunks)

        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                # 先删旧文档
                conn.execute("DELETE FROM chunks WHERE doc_name = ?", (doc_name,))
                for idx, (chunk, vec) in enumerate(zip(chunks, vecs)):
                    conn.execute(
                        "INSERT INTO chunks (doc_name, chunk_idx, text, embedding) VALUES (?, ?, ?, ?)",
                        (doc_name, idx, chunk, vec.tobytes()),
                    )
        logger.info("Knowledge: added doc '%s' (%d chunks)", doc_name, len(chunks))

    def search(self, query: str) -> list[str]:
        """余弦检索，返回 top_k 个超过阈值的文本块。"""
        if not self.enabled or self._model is None:
            return []

        q_vec = self.embed([query])[0]

        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                rows = conn.execute("SELECT text, embedding FROM chunks").fetchall()

        if not rows:
            return []

        scores = []
        for text, emb_blob in rows:
            e = np.frombuffer(emb_blob, dtype=np.float32).reshape(1, -1)
            sim = float(np.dot(q_vec.reshape(1, -1), e.T)[0, 0])
            if sim >= self.threshold:
                scores.append((sim, text))

        scores.sort(reverse=True)
        return [text for _, text in scores[:self.top_k]]

    def build_rag_block(self, query: str) -> str:
        """构建注入 LLM 的 RAG 文本块。"""
        chunks = self.search(query)
        if not chunks:
            return ""
        lines = ["# 参考资料"]
        for i, c in enumerate(chunks, 1):
            lines.append(f"{i}. {c}")
        return "\n".join(lines)


def create_knowledge_store(config) -> KnowledgeStore:
    """从 pipeline 配置创建 KnowledgeStore。"""
    kc = config.knowledge if hasattr(config, 'knowledge') else {}
    return KnowledgeStore({
        "enabled": getattr(kc, 'enabled', False),
        "db_path": getattr(kc, 'db_path', 'data/knowledge.db'),
        "top_k": getattr(kc, 'top_k', 3),
        "threshold": getattr(kc, 'threshold', 0.35),
        "embedder_model_path": getattr(kc, 'embedder_model_path', ''),
    })
