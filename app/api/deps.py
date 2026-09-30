"""FastAPI 依赖注入。"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.agent_service import AgentService


def get_settings_dep() -> Settings:
    """以依赖形式提供配置对象。"""
    return get_settings()


@lru_cache(maxsize=1)
def get_agent_service_singleton() -> AgentService:
    """进程内共享的 AgentService（内部按需懒加载图，不会在启动时要求 API Key）。"""
    return AgentService()


def get_agent_service() -> AgentService:
    """FastAPI 依赖：``service: AgentService = Depends(get_agent_service)``。"""
    return get_agent_service_singleton()


def reset_agent_service_singleton() -> None:
    """清空服务单例（测试用）。"""
    get_agent_service_singleton.cache_clear()
