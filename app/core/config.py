"""应用配置。

设计说明：
- 只使用 python-dotenv + pydantic，避免额外引入 pydantic-settings；
- 先加载 .env（不覆盖已有环境变量），再逐项从 os.environ 读取并做类型转换；
- 通过 ``get_settings()`` 获取单例，测试里可用 ``reset_settings_cache()`` 重建。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# app/core/config.py -> app/core -> app -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# 幂等：文件不存在时静默跳过，不覆盖已存在的真实环境变量
load_dotenv(DEFAULT_ENV_FILE, override=False)


def _as_bool(raw: str | None, default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(raw: str | None, default: int) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _as_float(raw: str | None, default: float) -> float:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _as_list(raw: str | None, default: list[str]) -> list[str]:
    if raw is None or raw.strip() == "":
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings(BaseModel):
    """运行期配置（不可变语义：修改请改 .env 后重启）。"""

    # 应用
    app_name: str = "AI 研发助手 Agent"
    app_env: str = "dev"
    debug: bool = True
    log_level: str = "INFO"
    cors_origins: list[str] = ["*"]

    # 大模型
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # Agent
    max_agent_steps: int = 6

    # 工具
    workspace_dir: Path = PROJECT_ROOT / "data" / "workspace"
    enable_shell_tool: bool = False

    # 记忆
    checkpoint_backend: str = "memory"

    @property
    def model_configured(self) -> bool:
        """是否已配置可用的模型凭据（仅做配置检查，不发起网络请求）。"""
        return bool(self.openai_api_key and self.openai_api_key.strip())

    @property
    def recursion_limit(self) -> int:
        """LangGraph 递归上限：为“规划 + 多轮工具调用 + 收尾”留足余量。"""
        return max(self.max_agent_steps * 2 + 4, 10)


def _build_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "AI 研发助手 Agent"),
        app_env=os.getenv("APP_ENV", "dev"),
        debug=_as_bool(os.getenv("DEBUG"), True),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        cors_origins=_as_list(os.getenv("CORS_ORIGINS"), ["*"]),
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
        openai_base_url=(os.getenv("OPENAI_BASE_URL") or "").strip() or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        llm_temperature=_as_float(os.getenv("LLM_TEMPERATURE"), 0.2),
        llm_timeout_seconds=_as_float(os.getenv("LLM_TIMEOUT_SECONDS"), 60.0),
        llm_max_retries=_as_int(os.getenv("LLM_MAX_RETRIES"), 2),
        max_agent_steps=_as_int(os.getenv("MAX_AGENT_STEPS"), 6),
        workspace_dir=Path(
            os.getenv("WORKSPACE_DIR", str(PROJECT_ROOT / "data" / "workspace"))
        ).expanduser(),
        enable_shell_tool=_as_bool(os.getenv("ENABLE_SHELL_TOOL"), False),
        checkpoint_backend=os.getenv("CHECKPOINT_BACKEND", "memory").strip().lower() or "memory",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。"""
    return _build_settings()


def reset_settings_cache() -> None:
    """清空配置缓存（测试或 .env 变更后使用）。"""
    get_settings.cache_clear()
