"""会话记忆测试：后端选择、SQLite 落盘与各条降级路径，全程离线。

用例覆盖三件事：

1. sqlite 后端把会话历史写进 ``SQLITE_DB_PATH``，重建 checkpointer（等价进程重启）后仍能读回；
2. 库文件所在目录不存在时自动创建；
3. 依赖缺失 / 无事件循环 / 后端名非法时的行为（回退或明确报错）。
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import app.memory.checkpointer as checkpointer_module
from app.agents.graph import build_graph
from app.agents.state import create_initial_state
from app.core.config import PROJECT_ROOT, Settings, get_settings, reset_settings_cache
from app.memory.checkpointer import (
    aclose_checkpointers,
    get_checkpointer,
    reset_checkpointer_cache,
)
from app.schemas.chat import ChatRequest
from app.services.agent_service import AgentService
from tests.fakes import FakeChatModel

SYSTEM_PROMPT = "你是测试助手。"
# 图结构为 planner -> agent -> finalize，空工具场景每轮消耗 3 条预设回答
SCRIPT = ["1. 思考\n2. 回答", "推理结论", "最终回答"]


def _render(messages: list[Any]) -> str:
    """把消息列表拼成纯文本，用于断言模型实际看到了什么。"""
    return " ".join(str(getattr(item, "content", item)) for item in messages)


def _use_sqlite_backend(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> Path:
    """把配置切到 sqlite 后端，并把库文件指到临时目录。"""
    monkeypatch.setenv("CHECKPOINT_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    reset_settings_cache()
    return db_path


async def _run_once(question: str, thread_id: str) -> list[Any]:
    """在默认 checkpointer 上跑一轮对话，返回模型最后一次调用看到的消息。"""
    model = FakeChatModel(list(SCRIPT))
    graph = build_graph(model=model, tools=[])
    await graph.ainvoke(
        create_initial_state(question, SYSTEM_PROMPT),
        {"configurable": {"thread_id": thread_id}},
    )
    return model.calls[-1]


def test_sqlite_backend_persists_history_across_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """第一轮写入后重建 checkpointer（等价进程重启），第二轮仍能读到上一轮历史。"""
    db_path = _use_sqlite_backend(monkeypatch, tmp_path / "memory" / "checkpoints.db")

    async def scenario() -> str:
        await _run_once("第一轮问题：我叫小王", "s-persist")

        # 等价于进程重启：丢掉单例与连接，再从同一个库文件重新构建
        reset_checkpointer_cache()
        await aclose_checkpointers()

        return _render(await _run_once("第二轮问题：我叫什么？", "s-persist"))

    final_prompt = asyncio.run(scenario())
    asyncio.run(aclose_checkpointers())

    assert "第一轮问题：我叫小王" in final_prompt, "重启后应能读到上一轮的会话历史"
    assert "第二轮问题：我叫什么？" in final_prompt
    assert db_path.is_file(), "会话历史应落盘到 SQLITE_DB_PATH"

    # 绕过 LangGraph 直接读库，确认数据确实写进了 SQLite 文件
    with sqlite3.connect(db_path) as raw:
        rows = raw.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
    assert [row[0] for row in rows] == ["s-persist"]


def test_sqlite_backend_creates_missing_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SQLITE_DB_PATH 的上级目录不存在时应自动创建。"""
    db_path = _use_sqlite_backend(monkeypatch, tmp_path / "nested" / "deep" / "checkpoints.db")
    assert not db_path.parent.exists()

    asyncio.run(_run_once("你好", "s-mkdir"))
    asyncio.run(aclose_checkpointers())

    assert db_path.parent.is_dir()
    assert db_path.is_file()


def test_agent_service_keeps_session_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一个 session_id：服务重建（等价重启）后仍能续上上一轮对话。"""
    db_path = _use_sqlite_backend(monkeypatch, tmp_path / "checkpoints.db")

    async def scenario() -> str:
        first = AgentService(graph=build_graph(model=FakeChatModel(list(SCRIPT)), tools=[]))
        first_response = await first.chat(ChatRequest(message="我叫小王", session_id="s-agent"))
        assert first_response.answer == "最终回答"

        # 等价于进程重启
        reset_checkpointer_cache()
        await aclose_checkpointers()

        model = FakeChatModel(list(SCRIPT))
        second = AgentService(graph=build_graph(model=model, tools=[]))
        second_response = await second.chat(ChatRequest(message="我叫什么？", session_id="s-agent"))
        assert second_response.answer == "最终回答"
        return _render(model.calls[-1])

    final_prompt = asyncio.run(scenario())
    asyncio.run(aclose_checkpointers())

    assert db_path.is_file()
    assert "我叫小王" in final_prompt, "重启后同一 session_id 应恢复上一轮上下文"


def test_unknown_backend_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """非法后端名要明确报错，而不是静默回退。"""
    monkeypatch.setenv("CHECKPOINT_BACKEND", "redis")
    reset_settings_cache()

    with pytest.raises(ValueError, match="CHECKPOINT_BACKEND"):
        get_checkpointer()


def test_missing_sqlite_package_falls_back_to_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未安装可选依赖时回退到内存实现（服务不中断，只是不落盘）。"""
    _use_sqlite_backend(monkeypatch, tmp_path / "checkpoints.db")
    real_import = checkpointer_module.importlib.import_module

    def fake_import(name: str, package: str | None = None) -> Any:
        if name.startswith(("aiosqlite", "langgraph.checkpoint.sqlite")):
            raise ImportError(name)
        return real_import(name, package)

    monkeypatch.setattr(checkpointer_module.importlib, "import_module", fake_import)

    assert isinstance(get_checkpointer(), InMemorySaver)


def test_checkpointer_outside_event_loop_falls_back_to_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AsyncSqliteSaver 需要事件循环；同步上下文里应回退而不是抛异常。"""
    _use_sqlite_backend(monkeypatch, tmp_path / "checkpoints.db")

    assert isinstance(get_checkpointer(), InMemorySaver)


def test_settings_resolve_sqlite_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SQLITE_DB_PATH：相对路径按项目根目录解析，绝对路径原样使用；默认值保持不变。"""
    assert Settings().checkpoint_backend == "memory"
    assert Settings().sqlite_db_path == PROJECT_ROOT / "data" / "checkpoints.db"

    monkeypatch.setenv("SQLITE_DB_PATH", "data/checkpoints.db")
    reset_settings_cache()
    assert get_settings().sqlite_db_path == PROJECT_ROOT / "data" / "checkpoints.db"

    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "other.db"))
    reset_settings_cache()
    assert get_settings().sqlite_db_path == tmp_path / "other.db"
