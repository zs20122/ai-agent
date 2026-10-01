"""会话历史接口测试：角色归一化、limit 截断、SQLite 落盘后经 HTTP 读回，全程离线。

写入侧用假模型在**独立事件循环**里先造出历史（等价于“历史由上一个进程写进 SQLite”），
再通过 TestClient 调 ``GET /api/v1/chat/history/{session_id}`` 读回来，
因此这里同时覆盖了「历史真的落盘」和「接口真的从 checkpointer 读」两件事。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from app.agents.graph import build_graph
from app.agents.state import create_initial_state
from app.agents.tools.registry import get_tools
from app.core.config import get_settings, reset_settings_cache
from app.memory.checkpointer import aclose_checkpointers
from app.schemas.chat import MAX_HISTORY_LIMIT
from tests.fakes import FakeChatModel

SYSTEM_PROMPT = "你是测试助手。"
# 图结构为 planner -> agent -> finalize，空工具场景每轮消耗 3 条预设回答
SCRIPT = ["1. 思考\n2. 回答", "推理结论", "最终回答"]


def _tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _use_backend(
    monkeypatch: pytest.MonkeyPatch, backend: str, db_path: Path | None = None
) -> None:
    """切换会话记忆后端（sqlite 时顺带把库文件指到临时目录）。"""
    monkeypatch.setenv("CHECKPOINT_BACKEND", backend)
    if db_path is not None:
        monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    reset_settings_cache()


def _seed(
    session_id: str,
    questions: list[str],
    script: list[Any] | None = None,
    tools: list[Any] | None = None,
) -> None:
    """把若干轮对话写进默认 checkpointer，并关闭写入侧连接。

    不能在这里调用 ``reset_checkpointer_cache()``：memory 后端的缓存就是数据本身，
    清掉之后接口就读不到刚写入的历史了（sqlite 后端的连接由 ``aclose_checkpointers()`` 释放）。
    """
    responses = script if script is not None else SCRIPT * len(questions)

    async def scenario() -> None:
        graph = build_graph(model=FakeChatModel(list(responses)), tools=tools if tools else [])
        for question in questions:
            await graph.ainvoke(
                create_initial_state(question, SYSTEM_PROMPT),
                {"configurable": {"thread_id": session_id}},
            )

    asyncio.run(scenario())
    asyncio.run(aclose_checkpointers())


def _history(client: Any, session_id: str, query: str = "") -> dict[str, Any]:
    response = client.get(f"/api/v1/chat/history/{session_id}{query}")
    assert response.status_code == 200, response.text
    return response.json()


def test_history_returns_empty_list_for_unknown_session(client: Any) -> None:
    """会话不存在时返回 200 + 空列表（前端可直接渲染“暂无历史”）。"""
    body = _history(client, "no-such-session")

    assert body["session_id"] == "no-such-session"
    assert body["messages"] == []
    assert body["total_messages"] == 0
    assert body["message_count"] == 0
    assert body["filtered_messages"] == 0
    assert body["include_internal"] is False
    assert body["truncated"] is False


def test_history_rejects_invalid_limit(client: Any) -> None:
    """limit 超出 1..MAX_HISTORY_LIMIT 范围时由 FastAPI 直接返回 422。"""
    assert client.get("/api/v1/chat/history/s1?limit=0").status_code == 422
    assert client.get(f"/api/v1/chat/history/s1?limit={MAX_HISTORY_LIMIT + 1}").status_code == 422


def test_history_reads_persisted_sqlite_history(
    client: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SQLite 后端：历史落盘后接口能从 checkpointer 读回，且没配 API Key 也能读。"""
    db_path = tmp_path / "history" / "checkpoints.db"
    _use_backend(monkeypatch, "sqlite", db_path)
    _seed("s-sqlite", ["我叫小王"])

    assert db_path.is_file(), "会话历史应落盘到 SQLITE_DB_PATH"

    body = _history(client, "s-sqlite")

    assert body["backend"] == "sqlite"
    # 默认视图只保留「提问 + 最终回答」：系统提示词与 agent 草稿（推理结论）被过滤
    assert body["total_messages"] == 2
    assert body["message_count"] == 2
    assert body["filtered_messages"] == 2
    assert body["include_internal"] is False
    assert [(item["role"], item["content"]) for item in body["messages"]] == [
        ("user", "我叫小王"),
        ("assistant", "最终回答"),
    ]

    # 对照：同一个进程里对话接口因为没有 API Key 会返回 503，但历史照样读得到
    assert client.post("/api/v1/chat", json={"message": "你好"}).status_code == 503


def test_history_reports_memory_backend(
    client: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """memory 后端同样可读，backend 字段如实反映“历史只在进程内”。"""
    _use_backend(monkeypatch, "memory", tmp_path / "unused.db")
    _seed("s-memory", ["我是谁"])

    body = _history(client, "s-memory")

    assert body["backend"] == "memory"
    assert body["messages"][0] == {
        "role": "user",
        "content": "我是谁",
        "name": "",
        "tool_calls": [],
    }
    assert body["messages"][1]["role"] == "assistant"


def test_history_limit_keeps_recent_messages(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """limit 只保留最近的消息，并用 truncated / total_messages 说明被截断。"""
    _use_backend(monkeypatch, "memory")
    _seed("s-limit", ["第一轮", "第二轮"])

    full = _history(client, "s-limit")
    assert full["total_messages"] == 4  # 每轮 2 条：用户提问 + 最终回答
    assert full["message_count"] == 4
    assert full["filtered_messages"] == 4  # 每轮另有 2 条内部消息（系统提示词 / agent 草稿）
    assert full["truncated"] is False

    tail = _history(client, "s-limit", "?limit=2")

    assert tail["total_messages"] == 4
    assert tail["message_count"] == 2
    assert tail["truncated"] is True
    # 仍是时间正序：最后两条是第二轮的提问与最终回答
    assert [(item["role"], item["content"]) for item in tail["messages"]] == [
        ("user", "第二轮"),
        ("assistant", "最终回答"),
    ]


def test_history_filters_out_tool_steps(
    client: Any, monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """带工具调用的轮次：AIMessage 中间步骤与 ToolMessage 都不进默认视图。"""
    (workspace / "demo.py").write_text("print('hi')\n", encoding="utf-8")
    _use_backend(monkeypatch, "memory")
    _seed(
        "s-tool",
        ["项目里有哪些文件？"],
        script=[
            "1. 查看文件\n2. 给出结论",
            _tool_call("list_project_files", {"subdir": ""}),
            "我看过文件了。",
            "最终回答：项目里包含 demo.py。",
        ],
        tools=get_tools(get_settings()),
    )

    body = _history(client, "s-tool")

    assert [item["role"] for item in body["messages"]] == ["user", "assistant"]
    assert body["messages"][1]["content"] == "最终回答：项目里包含 demo.py。"
    assert body["messages"][1]["tool_calls"] == [], "调工具的那条中间消息已被过滤"
    assert "我看过文件了。" not in [item["content"] for item in body["messages"]]
    assert body["filtered_messages"] == 4  # system + 调工具步骤 + tool 结果 + agent 草稿


def test_history_can_include_internal_steps_for_debugging(
    client: Any, monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """``?include_internal=true`` 时保留完整过程（system / tool / 中间步骤），供排查用。"""
    (workspace / "demo.py").write_text("print('hi')\n", encoding="utf-8")
    _use_backend(monkeypatch, "memory")
    _seed(
        "s-internal",
        ["项目里有哪些文件？"],
        script=[
            "1. 查看文件\n2. 给出结论",
            _tool_call("list_project_files", {"subdir": ""}),
            "我看过文件了。",
            "最终回答：项目里包含 demo.py。",
        ],
        tools=get_tools(get_settings()),
    )

    body = _history(client, "s-internal", "?include_internal=true")

    assert body["include_internal"] is True
    assert body["filtered_messages"] == 0, "完整视图下没有“被过滤”的条数"
    assert [item["role"] for item in body["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "assistant",
    ]
    assert body["messages"][2]["tool_calls"] == ["list_project_files"]
    assert body["messages"][3]["name"] == "list_project_files"
    assert "demo.py" in body["messages"][3]["content"]


def test_to_history_messages_keeps_only_question_and_final_answer() -> None:
    """消息映射：只留 user 与最终回答；工具步骤 / 空消息 / 未知类型 / 草稿都被丢弃。"""

    class _Message:
        def __init__(
            self,
            type_: str,
            content: Any = "",
            name: str = "",
            tool_calls: Any = None,
        ) -> None:
            self.type = type_
            self.content = content
            self.name = name
            self.tool_calls = tool_calls

    # 私有工具函数：只有绕过模型才能造出内容块 / 未知类型这类形态
    from app.services.agent_service import _to_history_messages

    raw = [
        _Message("system", "系统提示词"),
        _Message("human", "你好"),
        _Message("ai", [{"type": "text", "text": "第一段"}, {"type": "text", "text": "第二段"}]),
        _Message("ai", "", tool_calls=[{"name": "read_project_file", "args": {}}]),
        _Message("ai", "   "),  # 空消息 -> 丢弃
        _Message("tool", "文件内容", name="read_project_file"),  # 工具结果 -> 丢弃
        _Message("ai", "最终回答"),
        _Message("custom-channel", "无关内容"),  # 未知类型 -> 丢弃
    ]

    assert [(item.role, item.content) for item in _to_history_messages(raw)] == [
        ("user", "你好"),
        ("assistant", "最终回答"),
    ]

    full = _to_history_messages(raw, include_internal=True)

    assert [(item.role, item.content) for item in full] == [
        ("system", "系统提示词"),
        ("user", "你好"),
        ("assistant", "第一段第二段"),
        ("assistant", ""),
        ("tool", "文件内容"),
        ("assistant", "最终回答"),
    ]
    assert full[3].tool_calls == ["read_project_file"]


def test_to_history_messages_drops_duplicate_and_draft_answers() -> None:
    """同一轮里 AI 多次输出（草稿 / 重复落盘）时只保留最后一条最终回答。"""

    class _Message:
        def __init__(self, type_: str, content: str = "") -> None:
            self.type = type_
            self.content = content
            self.name = ""
            self.tool_calls = None

    from app.services.agent_service import _to_history_messages

    raw = [
        _Message("human", "问题一"),
        _Message("ai", "草稿：我先看看"),
        _Message("ai", "最终回答一"),
        _Message("ai", "最终回答一"),  # finalize 重复落盘的同一条回答
        _Message("human", "问题二"),
        _Message("ai", "最终回答二"),
    ]

    assert [(item.role, item.content) for item in _to_history_messages(raw)] == [
        ("user", "问题一"),
        ("assistant", "最终回答一"),
        ("user", "问题二"),
        ("assistant", "最终回答二"),
    ]
