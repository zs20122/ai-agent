"""LangGraph 图组装。

业务只需要 ``get_graph()``；测试可用 ``build_graph(model=假模型, tools=[], checkpointer=...)``。

图结构::

    START -> planner -> agent -+-> tools -> agent   (循环)
                               +-> finalize -> END
"""

from __future__ import annotations

import functools
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    make_agent_node,
    make_finalize_node,
    make_planner_node,
    make_tools_node,
    route_after_agent,
)
from app.agents.state import AgentState
from app.agents.tools.registry import get_tools
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.llm.factory import get_chat_model
from app.memory.checkpointer import get_checkpointer

logger = get_logger(__name__)

PLANNER_NODE = "planner"
AGENT_NODE = "agent"
TOOLS_NODE = "tools"
FINALIZE_NODE = "finalize"


def build_graph(
    model: Any | None = None,
    tools: list[Any] | None = None,
    checkpointer: Any | None = None,
    settings: Settings | None = None,
) -> Any:
    """按依赖注入的方式组装并编译图。

    Args:
        model: 聊天模型；为空则使用 ``get_chat_model()``（需要已配置 API Key）。
        tools: 工具列表；为 None 时按配置从注册表读取。
        checkpointer: 会话记忆；为空则使用默认 checkpointer。
        settings: 配置对象，默认取全局配置。

    Returns:
        已编译的 LangGraph（CompiledStateGraph），支持 ainvoke / astream。
    """
    settings = settings or get_settings()

    if model is None:
        model = get_chat_model()
    if tools is None:
        tools = get_tools(settings)
    if checkpointer is None:
        checkpointer = get_checkpointer()

    builder = StateGraph(AgentState)

    builder.add_node(PLANNER_NODE, make_planner_node(model))
    builder.add_node(AGENT_NODE, make_agent_node(model, tools))
    builder.add_node(TOOLS_NODE, make_tools_node(tools))
    builder.add_node(FINALIZE_NODE, make_finalize_node(model))

    builder.add_edge(START, PLANNER_NODE)
    builder.add_edge(PLANNER_NODE, AGENT_NODE)
    builder.add_conditional_edges(
        AGENT_NODE,
        functools.partial(route_after_agent, max_steps=settings.max_agent_steps),
        {TOOLS_NODE: TOOLS_NODE, FINALIZE_NODE: FINALIZE_NODE},
    )
    builder.add_edge(TOOLS_NODE, AGENT_NODE)
    builder.add_edge(FINALIZE_NODE, END)

    graph = builder.compile(checkpointer=checkpointer)
    logger.info(
        "Agent 图已编译：tools=%s max_agent_steps=%s",
        [getattr(item, "name", "?") for item in tools],
        settings.max_agent_steps,
    )
    return graph


@lru_cache(maxsize=1)
def get_graph() -> Any:
    """进程内共享的已编译图单例（延迟到首次调用，避免启动时就需要 API Key）。"""
    return build_graph()


def reset_graph_cache() -> None:
    """清空图缓存（测试或配置变更后使用）。"""
    get_graph.cache_clear()
