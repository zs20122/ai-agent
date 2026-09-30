"""健康检查与就绪探针。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.api.deps import get_settings_dep
from app.core.config import Settings

router = APIRouter()


@router.get("/health", summary="存活检查")
async def health(settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    """进程存活即返回 ok（不探测外部依赖）。"""
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
    }


@router.get("/ready", summary="就绪检查")
async def ready(settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    """检查模型凭据是否配置（只读配置，不发起网络请求）。

    未配置时返回 status=degraded，接口本身仍返回 200，
    便于容器编排区分“进程挂了”和“缺配置”。
    """
    configured = settings.model_configured
    return {
        "status": "ready" if configured else "degraded",
        "model": settings.openai_model,
        "model_configured": configured,
        "hint": "" if configured else "请在 .env 中填写 OPENAI_API_KEY 后重启服务",
    }
