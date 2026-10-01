"""会话记忆（checkpointer）工厂。

支持两种后端，由 ``CHECKPOINT_BACKEND`` 选择：

- ``memory``：进程内 ``InMemorySaver``，重启即丢失（默认，适合开发与测试）；
- ``sqlite``：``langgraph-checkpoint-sqlite`` 的 ``AsyncSqliteSaver``，会话历史落盘到
  ``SQLITE_DB_PATH``（目录不存在会自动创建），进程重启后同一 ``thread_id`` 仍可续聊。

为什么用异步实现：``AgentService`` 全程以 ``ainvoke`` / ``astream`` 调用图，而 LangGraph 的
同步 ``SqliteSaver`` 对异步方法会直接抛 ``NotImplementedError``（提示改用 ``AsyncSqliteSaver``）；
``AsyncSqliteSaver`` 内部用 ``asyncio.Lock`` 串行化读写，并发请求也是安全的。

两个必须知道的约束（都由本模块兜住）：

1. ``AsyncSqliteSaver`` 构造时会绑定当前事件循环（``asyncio.get_running_loop()``），因此
   ``get_checkpointer()`` 需要在事件循环内调用；没有运行中的循环时（例如同步脚本里提前构图）
   会记一条警告并回退到内存实现，保证服务照常启动；
2. ``aiosqlite`` 的连接工作线程不是守护线程，不关闭会拖住进程退出，所以应用停止与测试收尾
   都要调用 ``aclose_checkpointers()``（``app.main`` 的 lifespan 已经接好）。
"""

from __future__ import annotations

import asyncio
import importlib
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 支持的后端；新增实现时在此登记，并补上对应的构建分支
SUPPORTED_BACKENDS = ("memory", "sqlite")

# InMemorySaver 或 AsyncSqliteSaver：后者来自可选依赖，用 Any 避免静态检查强耦合
Checkpointer = Any


@dataclass
class _SqliteEntry:
    """一份绑定到某个事件循环的 SQLite checkpointer。"""

    loop: Any
    saver: Any
    connection: Any


# 按数据库路径缓存（生产环境只有一个事件循环；测试重置缓存后会重建）
_sqlite_entries: dict[str, _SqliteEntry] = {}
# 待关闭的连接：关闭是协程（需要事件循环），先寄存，由 aclose_checkpointers() 统一处理
_pending_close: list[Any] = []


def _load_sqlite_support() -> tuple[Any, Any] | None:
    """动态导入可选依赖 ``aiosqlite`` 与 ``langgraph-checkpoint-sqlite``。

    用 ``importlib`` 而不是顶层 import：未安装时本模块仍可正常导入（回退到内存实现），
    与 RAG 工具缺失可选依赖时的处理方式保持一致。

    Returns:
        ``(aiosqlite, AsyncSqliteSaver)``；依赖缺失时返回 None。
    """
    try:
        aiosqlite: Any = importlib.import_module("aiosqlite")
        aio_module: Any = importlib.import_module("langgraph.checkpoint.sqlite.aio")
    except ImportError:
        return None

    saver_cls: Any = aio_module.AsyncSqliteSaver
    return aiosqlite, saver_cls


def _build_sqlite_checkpointer(db_path: Path) -> Checkpointer:
    """创建（或复用）SQLite checkpointer；条件不满足时回退到内存实现。"""
    support = _load_sqlite_support()
    if support is None:
        logger.warning(
            "CHECKPOINT_BACKEND=sqlite 需要可选依赖 langgraph-checkpoint-sqlite，"
            "请执行 python -m pip install -r requirements.txt 后重启；"
            "本次回退到内存 checkpointer（会话不会落盘）"
        )
        return InMemorySaver()

    aiosqlite, saver_cls = support
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning(
            "SQLite checkpointer 必须在事件循环内创建（AsyncSqliteSaver 会绑定当前循环），"
            "当前没有运行中的事件循环，本次回退到内存 checkpointer（会话不会落盘）"
        )
        return InMemorySaver()

    key = str(db_path)
    entry = _sqlite_entries.get(key)
    if entry is not None and entry.loop is not running_loop:
        # 事件循环变了（例如多次 asyncio.run）：旧实例的读写锁绑定在已结束的循环上，必须重建
        logger.warning("事件循环已变更，重建 SQLite checkpointer：%s", db_path)
        _pending_close.append(entry.connection)
        entry = None

    if entry is None:
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)  # sqlite 不会自动创建目录
            connection = aiosqlite.connect(str(db_path))
        except (OSError, sqlite3.Error) as exc:
            logger.warning(
                "SQLite 会话库不可用（%s），本次回退到内存 checkpointer：%s", db_path, exc
            )
            return InMemorySaver()

        entry = _SqliteEntry(loop=running_loop, saver=saver_cls(connection), connection=connection)
        _sqlite_entries[key] = entry
        logger.info("会话记忆已落盘：%s（表结构在首次读写时自动创建）", db_path)

    return entry.saver


@lru_cache(maxsize=1)
def _memory_checkpointer() -> InMemorySaver:
    """进程内共享的内存 checkpointer 单例。"""
    logger.info("使用内存 checkpointer：会话历史仅保留在当前进程内，重启即丢失")
    return InMemorySaver()


def get_checkpointer() -> Checkpointer:
    """返回按 ``CHECKPOINT_BACKEND`` 选出的进程内共享 checkpointer。

    ``sqlite`` 后端需要在事件循环内调用（见模块文档）；后端取值非法时抛 ``ValueError``。
    """
    settings = get_settings()
    backend = settings.checkpoint_backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"不支持的 CHECKPOINT_BACKEND={backend!r}，当前仅支持: {list(SUPPORTED_BACKENDS)}"
        )

    if backend == "sqlite":
        return _build_sqlite_checkpointer(settings.sqlite_db_path)
    return _memory_checkpointer()


def reset_checkpointer_cache() -> None:
    """清空 checkpointer 缓存（测试用）。

    连接不能在这里直接关闭（``aiosqlite`` 的关闭是协程），先移交待关闭队列，
    由 ``aclose_checkpointers()`` 在事件循环内真正关闭。
    """
    _pending_close.extend(entry.connection for entry in _sqlite_entries.values())
    _sqlite_entries.clear()
    _memory_checkpointer.cache_clear()


async def aclose_checkpointers() -> None:
    """关闭所有 SQLite 连接（应用停止 / 测试收尾时调用）。

    ``aiosqlite`` 的连接工作线程不是守护线程，不关闭会拖住进程退出；
    从未真正读写过的连接不会启动线程，``close()`` 会立即返回。
    """
    connections = [entry.connection for entry in _sqlite_entries.values()]
    _sqlite_entries.clear()
    connections.extend(_pending_close)
    _pending_close.clear()

    for connection in connections:
        try:
            await connection.close()
        except Exception as exc:  # noqa: BLE001 - 关闭失败只告警，不影响进程退出
            logger.warning("关闭 SQLite 会话记忆连接失败：%s", exc)
