"""YAML 配置加载，Pydantic 校验。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


# ── 配置模型 ──

class VADConfig(BaseModel):
    engine: str = "silero"
    min_silence_ms: int = 500
    min_speech_ms: int = 250
    speech_pad_ms: int = 400


class ASRConfig(BaseModel):
    engine: str = "sensevoice"
    model: str = "iic/SenseVoiceSmall"
    device: str = "cuda"
    language: str = "zh"


class LLMConfig(BaseModel):
    engine: str = "deepseek"
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com/v1"
    api_key_env: str = "DEEPSEEK_API_KEY"
    stream: bool = True
    max_tokens: int = 2048
    temperature: float = 0.7
    system_prompt: str = ""


class TTSVoiceConfig(BaseModel):
    ref_wav: str
    ref_text: str


class TTSConfig(BaseModel):
    engine: str = "voxcpm2"
    model: str = ""
    device: str = "cuda"
    sample_rate: int = 16000
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    optimize: bool = False
    atempo_rate: float = 1.0
    voices: dict[str, TTSVoiceConfig] = {}


class MemoryConfig(BaseModel):
    enabled: bool = False
    embedder_model_path: str = ""
    store_dir: str = "data/memory"
    top_k: int = 5


class KnowledgeConfig(BaseModel):
    enabled: bool = False
    embedder_model_path: str = ""
    db_path: str = "data/knowledge.db"
    top_k: int = 3
    threshold: float = 0.45


class PipelineConfig(BaseModel):
    vad: VADConfig = VADConfig()
    asr: ASRConfig = ASRConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    memory: MemoryConfig = MemoryConfig()
    knowledge: KnowledgeConfig = KnowledgeConfig()


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    websocket: dict = {"max_message_size": 16_777_216, "ping_interval": 30, "ping_timeout": 10}
    session: dict = {"max_history": 30, "idle_timeout": 300}


class Config(BaseModel):
    pipeline: PipelineConfig = PipelineConfig()
    server: ServerConfig = ServerConfig()


# ── 加载函数 ──

def _load_dotenv(path: Path) -> None:
    """加载 .env 文件（已存在的环境变量不覆盖）。"""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_config(
    pipeline_path: str | Path = "configs/pipeline.yaml",
    server_path: str | Path = "configs/server.yaml",
    env_path: str | Path = ".env.local",
    repo_root: str | Path | None = None,
) -> Config:
    """加载全部配置：.env → pipeline.yaml + server.yaml → Config 对象。

    repo_root: 项目根目录，用于解析相对路径。默认自动探测。
    """
    if repo_root is None:
        repo_root = _find_repo_root()
    else:
        repo_root = Path(repo_root)

    # .env
    _load_dotenv(repo_root / env_path)

    def _resolve(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else repo_root / path

    def _read_yaml(p: Path) -> dict[str, Any]:
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    pipeline_raw = _read_yaml(_resolve(pipeline_path))
    server_raw = _read_yaml(_resolve(server_path))

    # 为 llm 和 embedding 解析环境变量中的 API key
    llm_cfg = pipeline_raw.get("llm", {})
    key_env = llm_cfg.get("api_key_env", "DEEPSEEK_API_KEY")
    api_key = os.environ.get(key_env, "")
    if not api_key:
        raise RuntimeError(
            f"缺少 API Key: 环境变量 {key_env} 未设置。"
            f"请在 .env.local 中设置 {key_env}=your_key"
        )

    return Config(
        pipeline=PipelineConfig(**pipeline_raw),
        server=ServerConfig(**server_raw),
    )


def _find_repo_root() -> Path:
    """自动探测项目根目录（查找 pyproject.toml）。"""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").is_file():
            return current
        current = current.parent
    return Path.cwd()
