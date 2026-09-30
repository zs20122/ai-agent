"""LangGraph 图状态定义。

- ``messages`` 使用 ``add_messages`` reducer：节点只需返回“新增的消息”，由框架负责追加；
- 其余字段为普通通道，后写入的值覆盖先前的值。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Agent 在图节点间传递的共享状态。"""

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    plan: str
    steps: int
    answer: str
    error: str | None


def create_initial_state(question: str, system_prompt: str | None = None) -> AgentState:
    """构造一次对话的初始状态。

    Args:
        question: 用户问题。
        system_prompt: 系统提示词，为空时不注入 SystemMessage。
    """
    messages: list[AnyMessage] = []
    if system_prompt and system_prompt.strip():
        messages.append(SystemMessage(content=system_prompt.strip()))
    messages.append(HumanMessage(content=question))

    return AgentState(
        messages=messages,
        question=question,
        plan="",
        steps=0,
        answer="",
        error=None,
    )
