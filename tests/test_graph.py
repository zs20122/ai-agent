"""图装配与流程测试：使用假模型，完全不访问网络。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import build_graph
from app.agents.nodes import route_after_agent
from app.agents.state import create_initial_state
from app.agents.tools.registry import get_tools
from app.core.config import get_settings
from tests.fakes import FakeChatModel


def _tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _render(messages: list[Any]) -> str:
    """把消息列表拼成纯文本，用于断言模型实际看到了什么。"""
    return " ".join(str(getattr(item, "content", item)) for item in messages)


def _assert_no_dangling_tool_calls(messages: list[Any]) -> None:
    """任何带 tool_calls 的 AI 消息都必须有配对的 ToolMessage。

    这是 OpenAI 协议的硬要求：同一 thread 续聊时历史会被整体重放，悬空的 tool_calls
    会直接让请求返回 400。无论 Agent 是正常结束还是到达步数上限后强制收尾，
    这个不变量都必须成立。
    """
    pending: set[str] = set()
    for item in messages:
        calls = getattr(item, "tool_calls", None) or []
        if calls:
            pending.update(str(call.get("id")) for call in calls)
        elif isinstance(item, ToolMessage):
            pending.discard(str(item.tool_call_id))
    assert not pending, f"存在未配对的工具调用: {sorted(pending)}"


def test_graph_runs_tool_then_finalizes(workspace: Path) -> None:
    (workspace / "demo.py").write_text("print('hi')\n", encoding="utf-8")

    model = FakeChatModel(
        [
            "1. 查看文件\n2. 给出结论",
            _tool_call("list_project_files", {"subdir": ""}),
            "我看过文件了。",
            "最终回答：项目里包含 demo.py。",
        ]
    )
    graph = build_graph(
        model=model,
        tools=get_tools(get_settings()),
        checkpointer=InMemorySaver(),
    )

    result = asyncio.run(
        graph.ainvoke(
            create_initial_state("项目里有哪些文件？", "你是一名资深 AI 研发助手"),
            {"configurable": {"thread_id": "t-tool"}},
        )
    )

    assert result["answer"] == "最终回答：项目里包含 demo.py。"
    assert result["steps"] == 2
    assert result["plan"].startswith("1. 查看文件")

    tool_messages = [item for item in result["messages"] if isinstance(item, ToolMessage)]
    assert len(tool_messages) == 1
    assert "demo.py" in tool_messages[0].content
    assert model.bound_tools, "推理节点应当把工具绑定到模型上"
    _assert_no_dangling_tool_calls(result["messages"])


def test_graph_stops_at_max_steps(workspace: Path, monkeypatch: Any) -> None:
    """达到最大步数时不再执行工具，直接收尾（且不产生悬空的 tool_calls）。"""
    from app.core.config import reset_settings_cache

    monkeypatch.setenv("MAX_AGENT_STEPS", "1")
    reset_settings_cache()

    model = FakeChatModel(
        [
            "计划",
            _tool_call("list_project_files", {"subdir": ""}, "call_2"),
            "收尾答案",
        ]
    )
    graph = build_graph(
        model=model,
        tools=get_tools(get_settings()),
        checkpointer=InMemorySaver(),
    )

    result = asyncio.run(
        graph.ainvoke(
            create_initial_state("一直查文件", "system"),
            {"configurable": {"thread_id": "t-max"}},
        )
    )

    # 只调用模型 3 次：规划 -> 推理 -> 收尾，说明工具节点被完全跳过
    assert len(model.calls) == 3
    assert result["answer"] == "收尾答案"
    assert result["steps"] == 1
    assert not [item for item in result["messages"] if isinstance(item, ToolMessage)]

    # 悬空的工具调用被降级成普通文本消息，并写回状态
    assert any(
        isinstance(item, AIMessage) and "未执行的工具调用" in str(item.content)
        for item in result["messages"]
    )
    _assert_no_dangling_tool_calls(result["messages"])

    # 收尾模型看到的是降级后的文本，而不是非法的 tool_calls 序列
    finalize_prompt = model.calls[-1]
    assert "未执行的工具调用" in _render(finalize_prompt)
    assert not any(getattr(item, "tool_calls", None) for item in finalize_prompt)


def test_thread_can_continue_after_max_steps(workspace: Path, monkeypatch: Any) -> None:
    """到达步数上限的 thread 仍可续聊：第二轮推理重放的历史必须合法。"""
    from app.core.config import reset_settings_cache

    monkeypatch.setenv("MAX_AGENT_STEPS", "1")
    reset_settings_cache()

    model = FakeChatModel(
        [
            "计划1",
            _tool_call("list_project_files", {"subdir": ""}, "call_1"),
            "收尾1",
            "计划2",
            "第二问的回答",
            "收尾2",
        ]
    )
    graph = build_graph(
        model=model,
        tools=get_tools(get_settings()),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-max-continue"}}

    first = asyncio.run(graph.ainvoke(create_initial_state("第一问", "system"), config))
    second = asyncio.run(graph.ainvoke(create_initial_state("第二问", "system"), config))

    assert first["answer"] == "收尾1"
    assert second["answer"] == "收尾2"
    assert second["steps"] == 1  # 新一轮重新计数

    # 第 5 次模型调用（索引 4）是第二轮的推理节点：它看到的历史里不能有悬空 tool_calls
    replay = model.calls[4]
    assert not any(getattr(item, "tool_calls", None) for item in replay)
    assert "未执行的工具调用" in _render(replay)
    assert "第一问" in _render(replay)
    assert "第二问" in _render(replay)

    _assert_no_dangling_tool_calls(second["messages"])


def test_graph_keeps_multi_turn_history(workspace: Path) -> None:
    """相同 thread_id 的第二轮可看到第一轮的历史消息。"""
    model = FakeChatModel(["计划A", "问题A的回答", "计划B", "问题B的回答"])
    checkpointer = InMemorySaver()
    graph = build_graph(model=model, tools=[], checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "t-memory"}}

    asyncio.run(graph.ainvoke(create_initial_state("问题A", "system"), config))
    asyncio.run(graph.ainvoke(create_initial_state("问题B", "system"), config))

    # 第二轮第一次模型调用（规划）只有 2 条消息；最后一次（收尾）能看到历史
    rendered = _render(model.calls[-1])
    assert "问题A" in rendered
    assert "问题B" in rendered


def test_route_after_agent() -> None:
    plain = AIMessage(content="直接回答")
    with_call = _tool_call("read_project_file", {"path": "a.py"}, "call_x")

    assert route_after_agent({"messages": [], "steps": 0}) == "finalize"
    assert route_after_agent({"messages": [plain], "steps": 1}) == "finalize"
    assert route_after_agent({"messages": [with_call], "steps": 1}, max_steps=3) == "tools"
    assert route_after_agent({"messages": [with_call], "steps": 3}, max_steps=3) == "finalize"
