"""对话接口：一次性返回 + SSE 流式返回。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import get_agent_service
from app.core.logging import get_logger
from app.schemas.chat import ChatRequest, ChatResponse, StreamChunk
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


def _format_sse(chunk: StreamChunk) -> str:
    """SSE 帧：event 行 + data 行（JSON）+ 空行结尾。"""
    return f"event: {chunk.event}\ndata: {chunk.model_dump_json()}\n\n"
