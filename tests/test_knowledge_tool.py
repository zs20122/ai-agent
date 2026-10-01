"""知识库检索工具（RAG）测试：完全离线，用假 Embedding 替代 sentence-transformers。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import build_graph
from app.agents.nodes import DEFAULT_SYSTEM_PROMPT, PLANNER_PROMPT
from app.agents.state import create_initial_state
from app.agents.tools import knowledge
from app.agents.tools.knowledge import search_knowledge_base
from app.agents.tools.registry import get_tools
from app.core.config import get_settings, reset_settings_cache
from tests.fakes import FakeChatModel


class FakeEmbeddings(Embeddings):
    """确定性假 Embedding：按关键词出现次数构造向量并做 L2 归一化。

    必须继承 ``langchain_core.embeddings.Embeddings``：FAISS 只在
    ``isinstance(embedding_function, Embeddings)`` 成立时才按“嵌入模型”调用，
    否则会把它当成普通函数直接调用。

    归一化后 L2 距离与余弦相似度单调对应，因此“哪段最相关”是可预期的，
    从而能在不下载、不加载真实模型的前提下验证检索排序。
    """

    def __init__(self, keywords: list[str]) -> None:
        self._keywords = list(keywords)

    def _vector(self, text: str) -> list[float]:
        raw = [float(text.count(word)) for word in self._keywords]
        norm = math.sqrt(sum(value * value for value in raw))
        if norm == 0:  # 全零向量：保持零向量，避免除零
            return raw
        return [value / norm for value in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def knowledge_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """把知识库目录指向临时目录，并把真实 Embedding 换成假实现。"""
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("KNOWLEDGE_DIRS", str(docs))
    monkeypatch.setenv("KNOWLEDGE_CHUNK_SIZE", "200")
    monkeypatch.setenv("KNOWLEDGE_CHUNK_OVERLAP", "0")
    reset_settings_cache()
    knowledge.reset_knowledge_cache()

    fake = FakeEmbeddings(["启动", "streamlit", "安全", "依赖"])
    monkeypatch.setattr(knowledge, "_load_embeddings", lambda model_name: fake)

    yield docs

    knowledge.reset_knowledge_cache()


def test_search_returns_most_relevant_chunk(knowledge_dir: Path) -> None:
    (knowledge_dir / "startup.md").write_text(
        "# 启动方式\n先启动后端，再启动前端：uvicorn 与 streamlit 各占一个终端。\n",
        encoding="utf-8",
    )
    (knowledge_dir / "security.md").write_text(
        "# 安全说明\n命令行工具默认关闭，仅允许白名单命令。\n",
        encoding="utf-8",
    )

    output = search_knowledge_base.invoke({"query": "这个项目怎么启动？"})

    assert output.startswith("检索到")
    top = output.split("[片段 1]")[1].split("[片段 2]")[0]
    assert "startup.md" in top, f"top1 应当是启动文档，实际输出：\n{output}"
    assert "uvicorn" in top


def test_top_k_limits_returned_chunks(knowledge_dir: Path) -> None:
    (knowledge_dir / "a.md").write_text("# A\n启动方式见 README。\n", encoding="utf-8")
    (knowledge_dir / "b.md").write_text("# B\n依赖安装用 pip。\n", encoding="utf-8")

    output = search_knowledge_base.invoke({"query": "启动和依赖", "top_k": 1})

    assert "[片段 1]" in output
    assert "[片段 2]" not in output


def test_empty_knowledge_base_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    monkeypatch.setenv("KNOWLEDGE_DIRS", str(empty))
    reset_settings_cache()
    knowledge.reset_knowledge_cache()

    output = search_knowledge_base.invoke({"query": "怎么启动"})

    assert output.startswith("错误: 知识库检索不可用")
    assert "知识库为空" in output


def test_index_rebuilds_after_doc_change(knowledge_dir: Path) -> None:
    (knowledge_dir / "a.md").write_text("# A\n只讲依赖安装。\n", encoding="utf-8")
    assert "a.md" in search_knowledge_base.invoke({"query": "依赖怎么装"})

    # 新增文档后指纹变化，索引应自动重建（否则 b.md 根本不在索引里）
    (knowledge_dir / "b.md").write_text("# B\n只讲安全边界与白名单。\n", encoding="utf-8")

    output = search_knowledge_base.invoke({"query": "安全边界在哪里"})
    assert "b.md" in output


def test_registry_toggles_rag_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "search_knowledge_base" in {tool.name for tool in get_tools()}

    monkeypatch.setenv("ENABLE_RAG_TOOL", "false")
    reset_settings_cache()

    assert "search_knowledge_base" not in {tool.name for tool in get_tools()}


def test_default_prompt_contains_hard_rule() -> None:
    """硬性规定必须写在默认系统提示词里（回归保护：防止被误删）。"""
    assert "硬性规定" in DEFAULT_SYSTEM_PROMPT
    assert "search_knowledge_base" in DEFAULT_SYSTEM_PROMPT
    assert "read_project_file" in DEFAULT_SYSTEM_PROMPT
    assert "search_knowledge_base" in PLANNER_PROMPT


def test_graph_executes_knowledge_tool(knowledge_dir: Path) -> None:
    """端到端：模型发出检索请求 → 工具节点真实执行 → 结果回填 ToolMessage。"""
    (knowledge_dir / "startup.md").write_text(
        "# 启动方式\nuvicorn 启动后端，streamlit 启动前端。\n", encoding="utf-8"
    )

    model = FakeChatModel(
        [
            "1. 检索文档\n2. 给出启动命令",
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"query": "这个项目怎么启动"},
                        "id": "call_rag",
                        "type": "tool_call",
                    }
                ],
            ),
            "检索结果显示：uvicorn 启动后端，streamlit 启动前端。",
            "最终回答：先启动后端，再启动前端。",
        ]
    )
    graph = build_graph(
        model=model,
        tools=get_tools(get_settings()),
        checkpointer=InMemorySaver(),
    )

    result = asyncio.run(
        graph.ainvoke(
            create_initial_state("这个项目怎么启动？", DEFAULT_SYSTEM_PROMPT),
            {"configurable": {"thread_id": "t-rag"}},
        )
    )

    tool_messages: list[Any] = [
        item for item in result["messages"] if isinstance(item, ToolMessage)
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "search_knowledge_base"
    assert "检索到" in tool_messages[0].content
    assert "startup.md" in tool_messages[0].content
    assert result["answer"].startswith("最终回答")
