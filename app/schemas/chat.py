"""对话相关的请求/响应模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_SESSION_ID = "default"

# 会话历史接口：默认只返回最近 100 条，单次最多 500 条（避免一次拉出整段超长会话）
DEFAULT_HISTORY_LIMIT = 100
MAX_HISTORY_LIMIT = 500

# 前端习惯的角色名（LangChain 的 human / ai / system / tool 已在此归一化）
MessageRole = Literal["system", "user", "assistant", "tool"]


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


class ChatHistoryMessage(BaseModel):
    """会话历史里的一条消息（角色已归一化成前端习惯的写法）。"""

    role: MessageRole = Field(..., description="system / user / assistant / tool")
    content: str = Field(default="", description="消息正文（内容块已转成纯文本）")
    name: str = Field(default="", description="工具消息的工具名，其它角色为空")
    tool_calls: list[str] = Field(
        default_factory=list,
        description="该条 assistant 消息请求调用的工具名（仅 include_internal=true 时可能非空）",
    )


class ChatHistoryResponse(BaseModel):
    """会话历史响应（GET /api/v1/chat/history/{session_id}）。

    默认只返回 HumanMessage 与 AIMessage（用户的提问 + AI 的最终回答）：
    系统提示词、工具返回、中间工具调用步骤都在后端就被过滤，不进入 ``messages``。
    """

    session_id: str = Field(..., description="会话 ID，即 checkpointer 的 thread_id")
    backend: str = Field(
        default="memory",
        description="实际提供历史的后端：sqlite（落盘）/ memory（进程内，依赖缺失时降级）",
    )
    total_messages: int = Field(
        default=0, description="当前视图下的消息总条数（默认视图 = 提问 + 最终回答）"
    )
    message_count: int = Field(default=0, description="本次实际返回的消息条数")
    filtered_messages: int = Field(
        default=0,
        description="被过滤掉的内部消息条数（系统提示词 / 工具返回 / 中间步骤 / 草稿）",
    )
    include_internal: bool = Field(
        default=False, description="本次返回的是否为完整过程视图（含 system / tool / 中间步骤）"
    )
    truncated: bool = Field(default=False, description="是否因为 limit 截断了更早的历史")
    messages: list[ChatHistoryMessage] = Field(
        default_factory=list,
        description="按时间正序（由旧到新）排列的消息；会话不存在时为空列表",
    )
