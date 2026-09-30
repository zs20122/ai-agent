"""测试替身：假模型与假图，保证单元测试完全离线。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk


class FakeChatModel:
    """按预设脚本依次返回的假聊天模型。

    用法::

        model = FakeChatModel(["计划文本", AIMessage(content="", tool_calls=[...]), "最终答复"])
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Any]] = []
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> FakeChatModel:
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages: list[Any], config: Any = None, **kwargs: Any) -> AIMessage:
        self.calls.append(list(messages))
        item = self._responses.pop(0) if self._responses else "（预设脚本已用尽）"
        if isinstance(item, AIMessage):
            return item
        return AIMessage(content=str(item))


class FakeGraph:
    """假图：只验证 AgentService 对图结果的映射与流式事件，不涉及 LangGraph。"""

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or {}
        self.last_config: dict[str, Any] | None = None

    async def ainvoke(self, state: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.last_config = config
        return self.result

    async def astream(
        self,
        state: Any,
        config: dict[str, Any] | None = None,
        stream_mode: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        self.last_config = config
        yield AIMessageChunk(content="你好"), {"langgraph_node": "finalize"}
        yield AIMessageChunk(content="，世界"), {"langgraph_node": "planner"}
