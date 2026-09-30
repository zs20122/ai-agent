"""服务层测试：图结果映射、兜底逻辑与流式事件。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage

from app.core.config import get_settings
from app.schemas.chat import ChatRequest
from app.services.agent_service import FINALIZE_NODE, AgentService
from tests.fakes import FakeGraph


async def _collect(source: AsyncIterator[Any]) -> list[Any]:
    return [chunk async for chunk in source]


def test_chat_maps_graph_result() -> None:
    graph = FakeGraph({"answer": "答案是 A", "plan": "先做 X", "steps": 2, "messages": []})
    service = AgentService(graph=graph)

    response = asyncio.run(service.chat(ChatRequest(message="问题", session_id="s1")))

    assert response.answer == "答案是 A"
    assert response.plan == "先做 X"
    assert response.steps == 2
    assert response.model == get_settings().openai_model

    assert graph.last_config is not None
    assert graph.last_config["configurable"]["thread_id"] == "s1"
    assert graph.last_config["recursion_limit"] >= 10


def test_chat_falls_back_to_last_message() -> None:
    graph = FakeGraph({"answer": "", "messages": [AIMessage(content="兜底回答")]})

    response = asyncio.run(AgentService(graph=graph).chat(ChatRequest(message="hi")))

    assert response.answer == "兜底回答"


def test_chat_uses_default_session_id() -> None:
    graph = FakeGraph({"answer": "ok", "messages": []})

    response = asyncio.run(AgentService(graph=graph).chat(ChatRequest(message="hi")))

    assert response.session_id == "default"


def test_stream_emits_start_token_progress_and_end() -> None:
    service = AgentService(graph=FakeGraph())

    chunks = asyncio.run(_collect(service.stream(ChatRequest(message="hi", session_id="s2"))))

    assert [chunk.event for chunk in chunks] == ["start", "token", "progress", "end"]
    assert chunks[0].session_id == "s2"
    # 只有 finalize 节点的增量计为最终回答，其他节点归入 progress
    assert chunks[1].delta == "你好"
    assert chunks[1].node == FINALIZE_NODE
    assert chunks[2].node == "planner"
    assert chunks[-1].answer == "你好"
