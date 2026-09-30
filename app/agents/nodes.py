"""LangGraph 节点工厂。

用工厂函数（而非裸函数 + 全局模型）的原因：模型与工具通过闭包注入，
既避免模块级副作用，也让单元测试可以塞入假模型，无需真实网络调用。

图结构::

    START -> planner -> agent -+-> tools -> agent   (循环，最多 max_agent_steps 轮)
                               +-> finalize -> END

节点约定：入参是完整 AgentState，出参是“本次要写回状态的增量字典”。
``messages`` 通道带 ``add_messages`` reducer，因此节点只返回新增消息。

最大步数保护（收尾时）：
到达 ``max_agent_steps`` 时模型可能刚返回一个带 ``tool_calls`` 的消息，但工具还没执行，
此时直接进入收尾会形成“assistant 带 tool_calls 却没有配对 tool 消息”的非法序列。
收尾节点因此做两件事：给模型看的上下文里把该消息降级为普通文本；同时用
``RemoveMessage`` 把它从图状态中删掉——否则同一 thread 续聊时会把非法序列重放给模型
（OpenAI 会直接返回 400）。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Sequence
from typing import Any, Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from app.agents.state import AgentState
from app.core.logging import get_logger

logger = get_logger(__name__)


class NodeCallable(Protocol):
    """LangGraph 节点签名：入参是完整图状态，返回“本次要写回状态的增量字典”。

    这里用 ``Protocol`` 而不是 ``Callable[[AgentState], ...]``：LangGraph 的
    ``StateGraph.add_node`` 允许按关键字传入状态（形参名 ``state``），而 ``Callable``
    声明的是位置参数，类型检查器会因此报“缺少关键字参数 state”。
    """

    def __call__(self, state: AgentState) -> Awaitable[dict[str, Any]]: ...


DEFAULT_SYSTEM_PROMPT = """你是一名资深 AI 研发助手，负责代码分析、架构设计与问题排查。
工作准则：
1. 使用中文回答，结构清晰，必要时给出可直接运行的示例代码；
2. 需要了解项目内容时先调用工具查看，不要凭空猜测文件内容；
3. 不编造 API、文件路径或运行结果；不确定的地方明确说明假设与风险。"""

PLANNER_PROMPT = """你是任务规划器。请针对用户问题给出不超过 5 条、每条一行的解决步骤。
只输出步骤列表，不要输出与规划无关的内容。"""

FINALIZE_PROMPT = """你是资深 AI 研发助手。请基于以上对话过程与工具返回结果，给出最终答复。
要求：中文回答；结论先行；引用了工具结果时说明依据；信息不足时明确指出还缺什么。"""


# --------------------------------------------------------------------------- #
# 内部工具函数
# --------------------------------------------------------------------------- #
def message_text(message: Any) -> str:
    """把消息对象/内容块安全地转成纯文本。"""
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
        return "".join(parts)
    return str(content)


def extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    """统一取出消息上的 tool_calls（兼容 dict / 对象两种形态）。"""
    raw_calls = getattr(message, "tool_calls", None) or []
    calls: list[dict[str, Any]] = []

    for call in raw_calls:
        if isinstance(call, dict):
            calls.append(dict(call))
        else:
            calls.append(
                {
                    "name": getattr(call, "name", None),
                    "args": getattr(call, "args", {}) or {},
                    "id": getattr(call, "id", None),
                }
            )
    return calls


def _stringify_output(output: Any) -> str:
    """把工具返回值转成字符串（工具应返回字符串，这里兜底）。"""
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    try:
        return json.dumps(output, ensure_ascii=False, default=str)
    except TypeError:
        return str(output)


def _latest_question(state: AgentState) -> str:
    question = state.get("question")
    if question:
        return str(question)
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return message_text(message)
    return ""


# --------------------------------------------------------------------------- #
# 节点工厂
# --------------------------------------------------------------------------- #
def make_planner_node(model: Any) -> NodeCallable:
    """规划节点：先产出执行计划（失败则降级为“无计划”，不阻塞主流程）。"""

    async def planner_node(state: AgentState) -> dict[str, Any]:
        question = _latest_question(state)
        try:
            response = await model.ainvoke(
                [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=question)]
            )
            plan = message_text(response)
            logger.info("规划完成：%s", plan.splitlines()[0] if plan else "<空>")
        except Exception as exc:  # noqa: BLE001 - 规划失败不应中断整轮对话
            logger.warning("规划节点失败，继续执行：%s", exc)
            plan = ""

        return {"plan": plan, "steps": 0}

    return planner_node


def make_agent_node(model: Any, tools: Sequence[Any] | None = None) -> NodeCallable:
    """推理节点：带工具调用能力的模型推理。"""
    tool_list = list(tools or [])
    bound_model: Any = model
    if tool_list and hasattr(model, "bind_tools"):
        bound_model = model.bind_tools(tool_list)

    async def agent_node(state: AgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        steps = int(state.get("steps") or 0) + 1

        try:
            response = await bound_model.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 - 把错误变成消息，让流程可收尾
            logger.exception("推理节点失败")
            return {
                "messages": [AIMessage(content=f"（模型调用失败：{exc}）")],
                "steps": steps,
                "error": str(exc),
            }

        if not isinstance(response, BaseMessage):
            response = AIMessage(content=_stringify_output(response))

        calls = extract_tool_calls(response)
        if calls:
            logger.info("第 %s 步请求工具：%s", steps, [call.get("name") for call in calls])

        return {"messages": [response], "steps": steps, "error": None}

    return agent_node


def make_tools_node(tools: Sequence[Any]) -> NodeCallable:
    """工具执行节点：按 AI 消息里的 tool_calls 逐个执行并回填 ToolMessage。"""
    tool_map = {str(getattr(item, "name", "") or ""): item for item in tools}

    async def tools_node(state: AgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        last_message = messages[-1] if messages else None
        calls = extract_tool_calls(last_message)
        if not calls:
            return {}

        results: list[BaseMessage] = []
        for index, call in enumerate(calls):
            name = str(call.get("name") or "")
            args = call.get("args") or {}
            call_id = str(call.get("id") or f"call_{index}_{name or 'unknown'}")
            tool = tool_map.get(name)

            if tool is None:
                content = (
                    f"错误: 未注册的工具 {name!r}，"
                    f"可用工具: {sorted(key for key in tool_map if key)}"
                )
            else:
                try:
                    content = _stringify_output(await tool.ainvoke(args))
                except Exception as exc:  # noqa: BLE001 - 工具失败也要回填，保证消息配对完整
                    logger.warning("工具 %s 执行失败：%s", name, exc)
                    content = f"错误: 工具 {name} 执行失败: {exc}"

            kwargs: dict[str, Any] = {"content": content, "tool_call_id": call_id}
            if name:
                kwargs["name"] = name
            results.append(ToolMessage(**kwargs))

        return {"messages": results}

    return tools_node


def make_finalize_node(model: Any) -> NodeCallable:
    """收尾节点：基于完整上下文生成最终回答。"""

    async def finalize_node(state: AgentState) -> dict[str, Any]:
        history, stale_ids, replacements = _sanitize_for_finalize(list(state.get("messages") or []))
        # 先删除悬空消息、再追加降级后的文本，写回状态的序列才是合法的
        cleanup: list[BaseMessage] = [RemoveMessage(id=item) for item in stale_ids]
        if cleanup:
            logger.info("收尾前清理 %s 条未配对的工具调用消息", len(cleanup))

        prompt = [SystemMessage(content=FINALIZE_PROMPT), *history]

        try:
            response = await model.ainvoke(prompt)
            answer = message_text(response).strip() or "（模型未返回内容）"
        except Exception as exc:  # noqa: BLE001 - 收尾失败也要给出可读结果
            logger.exception("收尾节点失败")
            return {
                "messages": [*cleanup, *replacements],
                "answer": f"（生成最终回答失败：{exc}）",
                "error": str(exc),
            }

        return {
            "messages": [*cleanup, *replacements, response],
            "answer": answer,
            "error": None,
        }

    return finalize_node


def _sanitize_for_finalize(messages: list[Any]) -> tuple[list[Any], list[str], list[Any]]:
    """处理末尾“有工具调用但没有工具结果”的 AI 消息，保证消息序列合法。

    OpenAI 协议要求带 ``tool_calls`` 的 assistant 消息后必须紧跟对应数量的 tool 消息，
    强制收尾（到达步数上限）时会命中该情况。这里一并给出三份信息：

    1. 给收尾模型看的消息列表：悬空消息被替换成普通文本，模型仍知道“哪些工具没跑”；
    2. 需要从图状态中删除的悬空消息 id（供 ``RemoveMessage`` 使用）；
    3. 需要写回状态、用来替代悬空消息的文本消息。

    图状态里的消息由 ``add_messages`` 统一分配 id，正常都能取到；只有直接手搓 state
    调用本函数（例如某些单测）时才可能取不到 id，此时仅做提示词降级、不删除，
    避免对不存在的 id 发 ``RemoveMessage``（会抛 ValueError）。

    Returns:
        (提示词消息列表, 需要删除的消息 id 列表, 需要写回的替代消息列表)
    """
    cleaned = list(messages)
    stale_ids: list[str] = []
    replacements: list[Any] = []

    while cleaned and isinstance(cleaned[-1], AIMessage) and extract_tool_calls(cleaned[-1]):
        pending = cleaned.pop()
        names = ", ".join(str(call.get("name") or "?") for call in extract_tool_calls(pending))
        replacement = AIMessage(content=f"（已达到最大推理步数限制，未执行的工具调用：{names}）")

        pending_id = getattr(pending, "id", None)
        if pending_id:
            stale_ids.append(str(pending_id))
            replacements.append(replacement)
        else:
            logger.warning("悬空工具调用消息缺少 id，无法从状态删除，仅在提示词中降级")

        cleaned.append(replacement)

    return cleaned, stale_ids, replacements


def route_after_agent(state: AgentState, max_steps: int = 6) -> str:
    """条件边：决定继续执行工具还是进入收尾。

    Returns:
        "tools" 或 "finalize"。
    """
    messages = list(state.get("messages") or [])
    last_message = messages[-1] if messages else None
    calls = extract_tool_calls(last_message)

    if not calls:
        return "finalize"

    if int(state.get("steps") or 0) >= max(1, int(max_steps)):
        logger.warning("已达最大推理步数 %s，停止调用工具并进入收尾", max_steps)
        return "finalize"

    return "tools"
