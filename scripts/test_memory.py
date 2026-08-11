#!/usr/bin/env python
"""三层记忆体系功能测试"""
import sys, os
sys.path.insert(0, 'src')

from voice_infer.memory.store import MemoryStore
from voice_infer.knowledge.store import KnowledgeStore

# Test L2: MemoryStore
print("=== L2 Memory Test ===")
mem = MemoryStore(
    config={"enabled": True, "store_dir": "data/memory_test", "top_k": 5,
            "embedder_model_path": "/root/.cache/modelscope/models/BAAI--bge-m3/snapshots/master"},
    llm_config={"model": "deepseek-v4-flash", "base_url": "https://api.deepseek.com/v1"},
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
)
print(f"  enabled={mem.enabled}")

# Test L3: KnowledgeStore
print("\n=== L3 Knowledge Test ===")
kb = KnowledgeStore({
    "enabled": True,
    "db_path": "data/knowledge_test.db",
    "top_k": 3,
    "threshold": 0.35,
    "embedder_model_path": "/root/.cache/modelscope/models/BAAI--bge-m3/snapshots/master",
})
print(f"  enabled={kb.enabled}")

# Add test document
kb.add_document("test", "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。深度学习是机器学习的一个分支，通过多层神经网络来学习数据的表示。")
print(f"  doc added")

# Search
results = kb.search("什么是深度学习")
print(f"  search '什么是深度学习': {len(results)} results")
for r in results:
    print(f"    - {r[:80]}...")

print("\n=== All tests passed ===")
