"""会话记忆（checkpointer）工厂。

当前实现：进程内内存 checkpointer（``InMemorySaver``，按 thread_id 保存多轮历史）。
适合开发与单实例部署；进程重启即丢失，多副本之间不共享。

生产环境请替换为持久化实现（需额外安装对应包，本文件刻意不 import，避免多出硬依赖）：:

    # pip install langgraph-checkpoint-sqlite
    from langgraph.checkpoint.sqlite import SqliteSaver

    # pip install langgraph-checkpoint-postgres
    from langgraph.checkpoint.postgres import PostgresSaver

替换方式：修改 ``get_checkpointer()`` 的 return 即可，
``AgentService`` 只依赖“存在一个 checkpointer”这一事实，无需改动其他代码。
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 目前支持的后端；新增持久化实现时在此登记并在下方分支返回
SUPPORTED_BACKENDS = ("memory",)


@lru_cache(maxsize=1)
def get_checkpointer() -> InMemorySaver:
    """返回进程内共享的 checkpointer 单例。"""
    backend = get_settings().checkpoint_backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"不支持的 CHECKPOINT_BACKEND={backend!r}，当前仅支持: {list(SUPPORTED_BACKENDS)}"
        )

    logger.info("使用内存 checkpointer（会话历史仅保留在当前进程内）")
    return InMemorySaver()


def reset_checkpointer_cache() -> None:
    """清空 checkpointer 缓存（测试用）。"""
    get_checkpointer.cache_clear()
