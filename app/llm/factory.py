"""ChatOpenAI 客户端工厂。

- 业务代码不要直接 ``ChatOpenAI(...)``，统一走 ``get_chat_model()``，便于替换/测试；
- 使用 OpenAI 兼容协议，可通过 OPENAI_BASE_URL 指向中转网关或自建服务。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def create_chat_model(settings: Settings | None = None, **overrides: Any) -> ChatOpenAI:
    """按配置创建 ChatOpenAI 实例。

    Args:
        settings: 配置对象，默认取全局配置。
        **overrides: 覆盖任意 ChatOpenAI 参数（如 temperature、model）。

    Raises:
        RuntimeError: 未配置 OPENAI_API_KEY。
    """
    settings = settings or get_settings()

    if not settings.model_configured:
        raise RuntimeError(
            "未配置 OPENAI_API_KEY：请复制 .env.example 为 .env 并填写模型凭据后重启服务"
        )

    kwargs: dict[str, Any] = {
        "model": settings.openai_model,
        "api_key": settings.openai_api_key,
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url

    kwargs.update(overrides)

    logger.info(
        "创建 ChatOpenAI：model=%s base_url=%s temperature=%s",
        kwargs["model"],
        kwargs.get("base_url", "<默认官方地址>"),
        kwargs["temperature"],
    )
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    """进程内共享的模型实例（客户端本身线程/协程安全）。"""
    return create_chat_model()


def reset_chat_model_cache() -> None:
    """清空模型缓存（测试用）。"""
    get_chat_model.cache_clear()
