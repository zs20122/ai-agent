"""业务编排层：把 LangGraph 图包装成对上层（API / CLI）可用的对话能力。

分层要求：本层可以依赖 agents / llm / schemas / core，但 agents 层不得反向依赖本层。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import get_graph
from app.agents.nodes import DEFAULT_SYSTEM_PROMPT, message_text
from app.agents.state import AgentState, create_initial_state
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.memory.checkpointer import get_checkpointer
from app.schemas.chat import (
    DEFAULT_HISTORY_LIMIT,
    ChatHistoryMessage,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    MessageRole,
    StreamChunk,
)

logger = get_logger(__name__)

FINALIZE_NODE = "finalize"

# LangChain 的消息类型 -> 前端习惯的角色名（未登记的类型会被跳过）
_ROLE_BY_TYPE: dict[str, MessageRole] = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
}

# 会话历史接口默认只返回这两类角色：用户的提问 + AI 的最终回答。
# SystemMessage（系统提示词）、ToolMessage（工具返回）以及带 tool_calls 的中间步骤
# 都属于 Agent 的内部过程，聊天界面不需要，默认过滤掉（调试可加 ?include_internal=true）。
_CONVERSATION_ROLES: frozenset[str] = frozenset({"user", "assistant"})


class AgentService:
    """Agent 调用入口。

    Args:
        graph: 已编译的图；为空时延迟到首次使用才通过 ``get_graph()`` 构建
            （这样服务可以在没有 API Key 的情况下正常启动）。
        settings: 配置对象，默认取全局配置。
    """

    def __init__(self, graph: Any | None = None, settings: Settings | None = None) -> None:
        self._graph = graph
        self._settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def graph(self) -> Any:
        if self._graph is None:
            self._graph = get_graph()
        return self._graph

    # ---------------------------------------------------------------- 内部
    @staticmethod
    def _thread_config(session_id: str) -> dict[str, Any]:
        """thread_id = session_id：读写会话历史时定位同一份 checkpoint。"""
        return {"configurable": {"thread_id": session_id}}

    def _run_config(self, session_id: str) -> dict[str, Any]:
        """执行一轮对话时的配置（thread_id + 兜底递归上限）。"""
        return {
            **self._thread_config(session_id),
            "recursion_limit": self._settings.recursion_limit,
        }

    def _initial_state(self, request: ChatRequest) -> AgentState:
        return create_initial_state(request.message, request.system_prompt or DEFAULT_SYSTEM_PROMPT)

    @staticmethod
    def _last_text(result: dict[str, Any]) -> str:
        for message in reversed(list(result.get("messages") or [])):
            text = message_text(message).strip()
            if text:
                return text
        return ""

    # ---------------------------------------------------------------- 对外
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """执行一轮对话，返回完整回答。"""
        logger.info("会话 %s 收到请求：%s", request.session_id, request.message[:80])

        result = await self.graph.ainvoke(
            self._initial_state(request), self._run_config(request.session_id)
        )
        answer = str(result.get("answer") or "").strip() or self._last_text(result)

        return ChatResponse(
            session_id=request.session_id,
            answer=answer or "（本次未生成有效回答，请检查日志）",
            plan=str(result.get("plan") or ""),
            steps=int(result.get("steps") or 0),
            model=self._settings.openai_model,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """流式执行一轮对话。

        事件含义：
        - start：开始；
        - progress：planner / agent 节点的输出（供前端展示“思考过程”）；
        - token：finalize 节点的逐字增量（最终回答）；
        - end：结束，answer 字段为完整回答。
        """
        session_id = request.session_id
        yield StreamChunk(event="start", session_id=session_id, node="planner")

        answer_parts: list[str] = []
        async for item in self.graph.astream(
            self._initial_state(request),
            self._run_config(session_id),
            stream_mode="messages",
        ):
            chunk, metadata = _split_stream_item(item)
            node = str(metadata.get("langgraph_node") or "")
            text = message_text(chunk)
            if not text:
                continue

            if node == FINALIZE_NODE:
                answer_parts.append(text)
                yield StreamChunk(event="token", session_id=session_id, node=node, delta=text)
            else:
                yield StreamChunk(event="progress", session_id=session_id, node=node, delta=text)

        # 兜底：若流式过程中没有拿到 finalize 的增量（例如元数据缺失），直接读最终状态
        if not answer_parts and hasattr(self.graph, "aget_state"):
            snapshot = await self.graph.aget_state(self._run_config(session_id))
            values = getattr(snapshot, "values", None) or {}
            answer_parts.append(str(values.get("answer") or ""))

        yield StreamChunk(
            event="end",
            session_id=session_id,
            node=FINALIZE_NODE,
            answer="".join(answer_parts),
        )

    async def history(
        self,
        session_id: str,
        limit: int = DEFAULT_HISTORY_LIMIT,
        *,
        include_internal: bool = False,
    ) -> ChatHistoryResponse:
        """读取某个会话的历史消息（直接读 checkpointer，不经过图与模型）。

        会话不存在（还没有任何一轮对话）时返回空列表而不是报错，便于前端直接渲染“暂无历史”；
        因为只依赖会话存储，**没配置 API Key 也能查看历史**。

        默认只返回**用户的提问与 AI 的最终回答**：SystemMessage、ToolMessage 以及
        「先调工具」的中间步骤都不返回；同一轮里 agent 的草稿也会被 finalize 的最终答复覆盖。
        这样聊天界面拿到什么就画什么，不会再出现「原始 7 条、气泡 2 个」的错位感。
        ``include_internal=True`` 时可拿到完整过程（用于排查）。

        Args:
            session_id: 会话 ID，对应 checkpointer 的 thread_id。
            limit: 最多返回最近多少条消息（按时间正序返回）。
            include_internal: 是否连同 system / tool / 中间步骤一起返回（默认否）。

        Raises:
            ValueError: CHECKPOINT_BACKEND 配置了不支持的后端。
        """
        checkpointer = get_checkpointer()
        snapshot = await checkpointer.aget_tuple(self._thread_config(session_id))
        # checkpoint 的 channel_values 与 AgentState 同构
        # （messages / question / plan / answer / steps）
        values = (snapshot.checkpoint.get("channel_values") or {}) if snapshot is not None else {}
        all_items = _all_history_messages(values.get("messages") or [])
        conversation = _conversation_only(all_items)
        messages = all_items if include_internal else conversation

        total = len(messages)
        truncated = total > limit
        if truncated:
            messages = messages[-limit:]  # 只保留最近的若干条

        return ChatHistoryResponse(
            session_id=session_id,
            backend=_backend_name(checkpointer),
            total_messages=total,
            message_count=len(messages),
            filtered_messages=0 if include_internal else max(len(all_items) - len(conversation), 0),
            include_internal=include_internal,
            truncated=truncated,
            messages=messages,
        )


def _backend_name(checkpointer: Any) -> str:
    """如实反映历史来自哪种存储（sqlite 依赖缺失时会降级成 memory）。"""
    return "memory" if isinstance(checkpointer, InMemorySaver) else "sqlite"


def _all_history_messages(raw_messages: Any) -> list[ChatHistoryMessage]:
    """把图状态里的消息逐条转成历史条目（system / tool / 中间步骤也保留，供调试视图使用）。

    - 角色归一化：human -> user、ai -> assistant，system / tool 原样保留；
    - 正文统一走 ``message_text()``（内容块、多模态块都能转成纯文本）；
    - 既没有正文也没有工具调用的空消息会被跳过，避免前端出现空气泡。
    """
    result: list[ChatHistoryMessage] = []
    for message in raw_messages:
        role = _ROLE_BY_TYPE.get(str(getattr(message, "type", "")).lower())
        if role is None:
            logger.debug("跳过无法识别的历史消息：%s", type(message).__name__)
            continue

        tool_calls = _tool_call_names(message)
        content = message_text(message).strip()
        if not content and not tool_calls:
            continue

        result.append(
            ChatHistoryMessage(
                role=role,
                content=content,
                name=str(getattr(message, "name", "") or ""),
                tool_calls=tool_calls,
            )
        )
    return result


def _conversation_only(items: list[ChatHistoryMessage]) -> list[ChatHistoryMessage]:
    """从完整历史里挑出「用户提问 + AI 最终回答」，把 Agent 的内部过程全部丢掉。

    过滤规则（这就是 ``GET /api/v1/chat/history`` 的默认返回）：

    - 只保留 ``user`` / ``assistant``：system（系统提示词）与 tool（工具返回）直接丢弃；
    - 带 ``tool_calls`` 的 assistant 消息是「先调工具」的中间步骤，不是最终回答，丢弃；
    - 正文为空的 assistant 消息丢弃；
    - 同一轮里出现多条 assistant 文字（agent 草稿 + finalize 最终答复）时，
      只保留**最后一条**，前面的草稿被覆盖。
    """
    kept: list[ChatHistoryMessage] = []

    for item in items:
        if item.role not in _CONVERSATION_ROLES:
            continue
        if item.tool_calls or not item.content:
            continue
        if item.role == "assistant" and kept and kept[-1].role == "assistant":
            kept.pop()  # 同一轮的草稿被最终回答覆盖
        kept.append(item)

    return kept


def _to_history_messages(
    raw_messages: Any, *, include_internal: bool = False
) -> list[ChatHistoryMessage]:
    """把图状态里的消息转成前端友好的历史条目。

    ``include_internal=False``（默认）时等价于 ``_conversation_only(_all_history_messages(...))``，
    即只返回 HumanMessage 与 AIMessage 的最终回答；置 True 则保留完整过程。
    """
    items = _all_history_messages(raw_messages)
    return items if include_internal else _conversation_only(items)


def _tool_call_names(message: Any) -> list[str]:
    """取出该条 AI 消息请求调用的工具名（兼容老式的 ``additional_kwargs.tool_calls``）。"""
    calls = getattr(message, "tool_calls", None)
    if not calls:
        calls = (getattr(message, "additional_kwargs", None) or {}).get("tool_calls")

    names: list[str] = []
    for call in calls or []:
        if isinstance(call, dict) and call.get("name"):
            names.append(str(call["name"]))
    return names


def _split_stream_item(item: Any) -> tuple[Any, dict[str, Any]]:
    """兼容 stream_mode="messages" 的产物形态：(chunk, metadata) 或裸 chunk。"""
    if isinstance(item, tuple) and len(item) == 2:
        chunk, metadata = item
        return chunk, metadata if isinstance(metadata, dict) else {}
    return item, {}
