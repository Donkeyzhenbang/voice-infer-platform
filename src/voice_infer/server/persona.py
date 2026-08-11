"""人设加载：解析 persona markdown 文件的 frontmatter + 正文。"""

from __future__ import annotations

from pathlib import Path


def load_persona(path: str | Path) -> dict:
    """解析 personas/<id>.md：YAML frontmatter + Markdown 正文。

    返回 {name, label, ref_wav, ref_text, ref_image, text}。
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
        "id": path.stem,
        "name": name,
        "label": str(meta.get("label") or name),
        "text": body.strip(),
        "ref_wav": meta.get("ref_wav"),
        "ref_text": meta.get("ref_text"),
        "ref_image": meta.get("ref_image"),
    }
