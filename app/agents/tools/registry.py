"""工具注册表：Agent 只从这里拿工具，新增工具时改这一处即可。"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.agents.tools.file_ops import FILE_TOOLS
from app.agents.tools.knowledge import search_knowledge_base
from app.agents.tools.shell import build_shell_tool
from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_tools(settings: Settings | None = None) -> list[BaseTool]:
    """返回当前配置下可用的工具列表。"""
    settings = settings or get_settings()

    tools: list[BaseTool] = list(FILE_TOOLS)

    if settings.enable_rag_tool:
        tools.append(search_knowledge_base)

    if settings.enable_shell_tool:
        tools.append(build_shell_tool())
        logger.warning("已启用命令行工具（白名单模式），请确认运行环境可信")

    return tools
