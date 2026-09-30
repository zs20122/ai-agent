"""对话相关的请求/响应模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_SESSION_ID = "default"


class ChatRequest(BaseModel):
    """一次对话请求。"""

    message: str = Field(..., min_length=1, description="用户输入的内容")
    session_id: str = Field(
        default=DEFAULT_SESSION_ID,
        min_length=1,
        max_length=64,
        description="会话 ID，相同 ID 共享多轮记忆（checkpointer 的 thread_id）",
    )
    system_prompt: str | None = Field(
        default=None,
        description="覆盖默认的系统提示词；留空使用内置默认值",
    )


class ChatResponse(BaseModel):
    """非流式对话响应。"""

    session_id: str
    answer: str = Field(..., description="最终回答")
    plan: str = Field(default="", description="Agent 生成的执行计划（调试用）")
    steps: int = Field(default=0, description="实际执行的重试/工具调用步数")
    model: str = Field(default="", description="本次使用的模型名")


class StreamChunk(BaseModel):
    """SSE 流式片段。

    event 取值：
    - start    会话开始
    - progress 中间节点（规划 / 工具调用）的输出，node 字段标明来源
    - token    最终回答的逐字增量
    - end      结束，answer 字段携带完整回答
    - error    执行异常，delta 字段携带错误信息
    """

    event: Literal["start", "progress", "token", "end", "error"] = "token"
    session_id: str = DEFAULT_SESSION_ID
    node: str = ""
    delta: str = ""
    answer: str | None = None
