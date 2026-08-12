"""服务层 —— FastAPI 应用入口。"""


def create_app(*args, **kwargs):
    """延迟导入 Web 依赖，使 persona/session 纯逻辑可独立测试。"""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)

__all__ = ["create_app"]
