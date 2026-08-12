"""人设加载：解析 persona markdown，并绑定提示词、音色与形象。"""

from __future__ import annotations

from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def load_persona(path: str | Path) -> dict:
    """解析 personas/<id>.md：YAML frontmatter + Markdown 正文。

    返回人设元数据和正文。素材路径的解析由 ``load_persona_registry`` 负责。
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    meta: dict = {}
    body = raw

    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            import yaml

            meta = yaml.safe_load(raw[3:end]) or {}
            if not isinstance(meta, dict):
                meta = {}
            body = raw[end + 4:]

    name = str(meta.get("name") or path.stem)
    return {
        "id": str(meta.get("id") or path.stem),
        "name": name,
        "label": str(meta.get("label") or name),
        "text": body.strip(),
        "voice_id": str(meta.get("voice_id") or meta.get("id") or path.stem),
        "greeting": str(meta.get("greeting") or ""),
        "ref_wav": meta.get("ref_wav"),
        "ref_text": meta.get("ref_text"),
        "ref_image": meta.get("ref_image"),
    }


def _resolve_optional_file(repo_root: Path, value, field: str, persona_id: str):
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        logger.warning(
            "Persona '%s' %s not found: %s (fallback enabled)",
            persona_id, field, path,
        )
        return None
    return str(path)


def load_persona_registry(repo_root: str | Path, config) -> tuple[str, dict[str, dict]]:
    """加载配置中的 persona 注册表。

    配置存在时严格使用 ``personas.list``；老配置没有注册表时兼容扫描
    ``personas/*.md``。参考音频或图片缺失不会阻塞语音服务，只会回退到
    默认音色或 Canvas 形象。
    """
    repo_root = Path(repo_root)
    configured = dict(getattr(config, "list", {}) or {})
    default_id = str(getattr(config, "default", "default") or "default")

    if configured:
        entries = configured.items()
    else:
        entries = ((p.stem, str(p)) for p in sorted((repo_root / "personas").glob("*.md")))

    personas: dict[str, dict] = {}
    for persona_id, file_name in entries:
        path = Path(file_name)
        if not path.is_absolute():
            path = repo_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Persona file not found: {path}")

        persona = load_persona(path)
        persona["id"] = str(persona_id)
        persona["voice_id"] = str(persona.get("voice_id") or persona_id)
        persona["ref_wav"] = _resolve_optional_file(
            repo_root, persona.get("ref_wav"), "ref_wav", str(persona_id)
        )
        persona["ref_image"] = _resolve_optional_file(
            repo_root, persona.get("ref_image"), "ref_image", str(persona_id)
        )

        ref_text = persona.get("ref_text")
        if ref_text:
            text_path = Path(str(ref_text))
            if not text_path.is_absolute():
                text_path = repo_root / text_path
            if text_path.is_file():
                persona["ref_text"] = text_path.read_text("utf-8").strip()
            elif persona["ref_wav"]:
                # 有音频却没有逐字台词时，VoxCPM2 会使用 reference-only cloning。
                logger.warning("Persona '%s' ref_text not found: %s", persona_id, text_path)
                persona["ref_text"] = ""
            else:
                persona["ref_text"] = ""
        else:
            persona["ref_text"] = ""
        personas[str(persona_id)] = persona

    if not personas:
        personas["default"] = {
            "id": "default", "name": "语音助手", "label": "助手",
            "text": "你是语音助手。用中文口语回复，简短自然。",
            "voice_id": "default", "greeting": "你好，有什么可以帮你？",
            "ref_wav": None, "ref_text": "", "ref_image": None,
        }
    if default_id not in personas:
        raise ValueError(f"personas.default={default_id!r} is not present in personas.list")
    return default_id, personas
