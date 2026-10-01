"""FastAPI 应用入口。

本地启动::

    uvicorn app.main:app --reload

Windows 请优先使用 ``scripts\\run_dev.ps1``：它会自动挑选带依赖的解释器、
检查端口占用并做启动前自检（应用商店占位版的 ``python`` 会静默退出）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import get_logger, setup_logging
from app.memory.checkpointer import aclose_checkpointers

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动/停止钩子：初始化日志并做一次配置自检。"""
    settings: Settings = app.state.settings

    setup_logging(settings.log_level)
    logger.info("启动 %s v%s（env=%s）", settings.app_name, __version__, settings.app_env)

    if settings.model_configured:
        logger.info(
            "模型：%s（base_url=%s，最大推理步数=%s）",
            settings.openai_model,
            settings.openai_base_url or "<官方默认>",
            settings.max_agent_steps,
        )
    else:
        logger.warning(
            "未配置 OPENAI_API_KEY：/api/v1/chat 将返回 503。"
            "请复制 .env.example 为 .env 并填写后重启。"
        )

    yield

    # 关闭 SQLite 会话记忆连接：aiosqlite 的工作线程不是守护线程，不关会拖住进程退出
    await aclose_checkpointers()

    logger.info("应用已停止")


def create_app(settings: Settings | None = None) -> FastAPI:
    """应用工厂（测试可注入自定义配置）。"""
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", tags=["system"], summary="服务信息")
    async def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "api": "/api/v1",
        }

    return app


# uvicorn 引用入口：app.main:app
app = create_app()
