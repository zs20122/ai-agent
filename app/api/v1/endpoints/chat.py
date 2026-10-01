"""对话接口：一次性返回、SSE 流式返回与会话历史读取。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import StreamingResponse

from app.api.deps import get_agent_service
from app.core.logging import get_logger
from app.schemas.chat import (
    DEFAULT_HISTORY_LIMIT,
    MAX_HISTORY_LIMIT,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    StreamChunk,
)
from app.services.agent_service import AgentService

logger = get_logger(__name__)

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，保证逐字下发
}


@router.post("/chat", response_model=ChatResponse, summary="对话（一次性返回）")
async def chat(
    request: ChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> ChatResponse:
    """执行一轮 Agent 对话并返回完整结果。"""
    try:
        return await service.chat(request)
    except RuntimeError as exc:
        # 典型场景：未配置 OPENAI_API_KEY
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - 统一转成 HTTP 语义错误
        logger.exception("对话执行失败")
        raise HTTPException(status_code=502, detail=f"Agent 执行失败: {exc}") from exc


@router.post("/chat/stream", summary="对话（SSE 流式返回）")
async def chat_stream(
    request: ChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> StreamingResponse:
    """以 Server-Sent Events 逐段返回执行过程与最终回答。

    事件格式::

        event: token
        data: {"event":"token","session_id":"default","node":"finalize","delta":"你","answer":null}
    """

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for chunk in service.stream(request):
                yield _format_sse(chunk)
        except RuntimeError as exc:
            yield _format_sse(
                StreamChunk(event="error", session_id=request.session_id, delta=str(exc))
            )
        except Exception as exc:  # noqa: BLE001 - 流已经开始，只能以事件形式报错
            logger.exception("流式对话执行失败")
            yield _format_sse(
                StreamChunk(
                    event="error",
                    session_id=request.session_id,
                    delta=f"Agent 执行失败: {exc}",
                )
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.get(
    "/chat/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="读取会话历史（来自 checkpointer）",
)
async def chat_history(
    session_id: str = Path(
        ...,
        min_length=1,
        max_length=64,
        description="会话 ID，即 checkpointer 的 thread_id",
    ),
    limit: int = Query(
        DEFAULT_HISTORY_LIMIT,
        ge=1,
        le=MAX_HISTORY_LIMIT,
        description="最多返回最近多少条消息",
    ),
    include_internal: bool = Query(
        False,
        description="调试用：连同 system / tool / 中间步骤一起返回（默认只返回提问与最终回答）",
    ),
    service: AgentService = Depends(get_agent_service),
) -> ChatHistoryResponse:
    """从 checkpointer 读取某个会话的历史消息。

    约定：

    - **默认只返回 HumanMessage 与 AIMessage**（用户的提问 + AI 的最终回答）：
      ``SystemMessage``（系统提示词）、``ToolMessage``（工具执行结果）以及带 ``tool_calls``
      的中间步骤会被过滤，``filtered_messages`` 说明过滤掉了几条；
    - 同一轮里 AI 有多次输出（agent 草稿 + finalize 最终答复）时只保留最终答复；
    - 会话不存在（或还没有任何一轮对话）时返回 **200 + 空列表**，前端可直接渲染“暂无历史”；
    - 消息按时间正序（由旧到新）返回，``limit`` 只保留**最近**的若干条，
      ``truncated`` 标记更早的历史是否被截断；
    - 角色已归一化成前端习惯的写法：``user`` / ``assistant``（``include_internal=true``
      时才会出现 ``system`` / ``tool``）；
    - 读取只依赖会话存储：``CHECKPOINT_BACKEND=sqlite`` 时即 SQLite 库里的落盘历史，
      因此**未配置 API Key 也能查看历史**（对话接口此时返回 503）。
    """
    try:
        return await service.history(session_id, limit=limit, include_internal=include_internal)
    except ValueError as exc:
        # 典型场景：CHECKPOINT_BACKEND 配置了不支持的后端
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - 统一转成 HTTP 语义错误
        logger.exception("读取会话历史失败")
        raise HTTPException(status_code=502, detail=f"读取会话历史失败: {exc}") from exc


def _format_sse(chunk: StreamChunk) -> str:
    """SSE 帧：event 行 + data 行（JSON）+ 空行结尾。"""
    return f"event: {chunk.event}\ndata: {chunk.model_dump_json()}\n\n"
