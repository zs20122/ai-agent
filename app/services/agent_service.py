"""业务编排层：把 LangGraph 图包装成对上层（API / CLI）可用的对话能力。

分层要求：本层可以依赖 agents / llm / schemas / core，但 agents 层不得反向依赖本层。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.agents.graph import get_graph
from app.agents.nodes import DEFAULT_SYSTEM_PROMPT, message_text
from app.agents.state import AgentState, create_initial_state
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.chat import ChatRequest, ChatResponse, StreamChunk

logger = get_logger(__name__)

FINALIZE_NODE = "finalize"


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
    def _run_config(self, session_id: str) -> dict[str, Any]:
        """thread_id = session_id，从而实现同一会话的多轮记忆。"""
        return {
            "configurable": {"thread_id": session_id},
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


def _split_stream_item(item: Any) -> tuple[Any, dict[str, Any]]:
    """兼容 stream_mode="messages" 的产物形态：(chunk, metadata) 或裸 chunk。"""
    if isinstance(item, tuple) and len(item) == 2:
        chunk, metadata = item
        return chunk, metadata if isinstance(metadata, dict) else {}
    return item, {}
