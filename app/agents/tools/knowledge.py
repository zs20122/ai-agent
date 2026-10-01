"""知识库检索工具（RAG）：文档切分 → 本地向量化 → FAISS 语义检索。

为什么需要它：
文件工具只能“整篇读取”。遇到「怎么启动 / 有哪些功能 / 怎么配置」这类文档类问题，
模型倾向于直接 ``read_project_file`` 把 README 全文读进来——既浪费上下文，
又容易漏掉散落在多篇文档里的答案。本工具先检索出最相关的若干片段再交给模型作答。

设计约束：
- **懒加载**：sentence-transformers / torch / faiss 体积大且导入慢（数秒），
  只在首次真正检索时才 import，既保证服务启动速度，也让未装 RAG 依赖时服务照常运行；
- **失败不抛异常**：按项目约定返回 ``错误: ...`` 字符串，模型能看到原因并自行调整；
- **索引缓存**：按“文件清单 + 大小 + 修改时间 + 分块参数”生成指纹，文档变更后自动重建；
- **只读**：只读取 ``KNOWLEDGE_DIRS`` 下的文本文件，不写任何磁盘文件。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 参与索引的文本文件后缀（只纳入文档类文件，不索引源码）
KNOWLEDGE_SUFFIXES = (".md", ".txt", ".rst")
# 索引时跳过的目录
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "data",
}
# 单个文件大小上限，超过则跳过（避免把超大日志塞进索引）
MAX_FILE_BYTES = 512_000

# 进程内索引缓存（指纹一致时直接复用）
_CACHE: dict[str, Any] = {"fingerprint": None, "store": None, "chunks": 0, "sources": 0}


@dataclass(frozen=True)
class KnowledgeFile:
    """一个待索引的文档文件。"""

    path: Path
    relative: str
    size: int
    mtime: float


def reset_knowledge_cache() -> None:
    """清空索引缓存（测试或文档大规模变更后使用）。"""
    _CACHE["fingerprint"] = None
    _CACHE["store"] = None
    _CACHE["chunks"] = 0
    _CACHE["sources"] = 0


def _resolve_dirs(settings: Settings) -> list[Path]:
    """把配置里的知识库目录解析成绝对路径（相对路径按项目根目录解析）。"""
    resolved: list[Path] = []
    for raw in settings.knowledge_dirs or ["."]:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        if candidate.exists() and candidate not in resolved:
            resolved.append(candidate)
    return resolved


def collect_files(settings: Settings) -> list[KnowledgeFile]:
    """收集知识库目录下的文档文件（按路径排序，结果稳定）。"""
    found: dict[str, KnowledgeFile] = {}

    for base in _resolve_dirs(settings):
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for item in candidates:
            try:
                if not item.is_file() or item.suffix.lower() not in KNOWLEDGE_SUFFIXES:
                    continue
                if any(part in SKIP_DIRS for part in item.parts):
                    continue
                stat = item.stat()
            except OSError:  # 权限/竞态等：跳过该文件即可，不影响整体索引
                continue
            if stat.st_size > MAX_FILE_BYTES:
                logger.warning("跳过超大文档（%s 字节）：%s", stat.st_size, item)
                continue
            try:
                relative = item.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                relative = item.as_posix()
            found[relative] = KnowledgeFile(
                path=item, relative=relative, size=stat.st_size, mtime=stat.st_mtime
            )

    return [found[key] for key in sorted(found)]


def _fingerprint(files: list[KnowledgeFile], settings: Settings) -> str:
    """索引指纹：任一文件或分块参数变化即失效。"""
    parts = [f"{item.relative}:{item.size}:{int(item.mtime)}" for item in files]
    parts.append(f"cfg:{settings.embedding_model}")
    parts.append(f"chunk:{settings.knowledge_chunk_size}:{settings.knowledge_chunk_overlap}")
    return "|".join(parts)


def _load_embeddings(model_name: str) -> Any:
    """构造 Embedding 模型（唯一重依赖入口，测试中可替换）。

    优先使用官方独立包 ``langchain-huggingface``；未安装时退回随 langchain 发行版提供的
    ``langchain_community.embeddings.HuggingFaceEmbeddings``（该模块已标记 sunset，
    但仍是不额外安装依赖就能接入 FAISS 的路径）。

    Args:
        model_name: HuggingFace 模型名，或本地模型目录的绝对路径。
    """
    try:
        # 官方独立包（未随项目固定依赖安装，缺失时回落到 langchain-community 的实现）；
        # 用 importlib 动态导入：可选依赖不必进入静态分析的 import 图，
        # 未安装时也不会让 pyright / 类型检查报错。
        module: Any = importlib.import_module("langchain_huggingface")
    except ImportError:  # pragma: no cover - 取决于是否安装了官方新包
        module = importlib.import_module("langchain_community.embeddings")

    return module.HuggingFaceEmbeddings(
        model_name=model_name, encode_kwargs={"normalize_embeddings": True}
    )


def _build_store(settings: Settings) -> tuple[Any, int, int]:
    """读取文档、切分、向量化并构建 FAISS 索引。

    Returns:
        (FAISS 向量库, 分块数, 文件数)
    """
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    files = collect_files(settings)
    if not files:
        raise FileNotFoundError(
            f"知识库为空：{settings.knowledge_dirs} 下没有 {'/'.join(KNOWLEDGE_SUFFIXES)} 文档"
        )

    chunk_size = max(50, settings.knowledge_chunk_size)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=max(0, min(settings.knowledge_chunk_overlap, chunk_size - 1)),
        # 中文文档用中文标点做分隔符，避免把整句切碎
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for item in files:
        content = item.path.read_text(encoding="utf-8", errors="replace")
        for index, chunk in enumerate(splitter.split_text(content)):
            if not chunk.strip():
                continue
            texts.append(chunk)
            metadatas.append({"source": item.relative, "chunk": index + 1})

    if not texts:
        raise ValueError("知识库文档均无可索引内容（可能都是空白文件）")

    embeddings = _load_embeddings(settings.embedding_model)
    store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    logger.info(
        "知识库索引已构建：文件=%s 分块=%s 模型=%s",
        len(files),
        len(texts),
        settings.embedding_model,
    )
    return store, len(texts), len(files)


def _get_store(settings: Settings) -> tuple[Any, int, int]:
    """取索引：指纹一致则复用缓存，否则重建。"""
    fingerprint = _fingerprint(collect_files(settings), settings)
    if _CACHE["store"] is not None and _CACHE["fingerprint"] == fingerprint:
        return _CACHE["store"], int(_CACHE["chunks"]), int(_CACHE["sources"])

    store, chunks, sources = _build_store(settings)
    _CACHE.update(
        {"fingerprint": fingerprint, "store": store, "chunks": chunks, "sources": sources}
    )
    return store, chunks, sources


@tool
def search_knowledge_base(query: str, top_k: int = 0) -> str:
    """在项目知识库（README、docs/*.md 等文档）中做语义检索，返回最相关的片段。

    凡是涉及项目文档、使用说明、启动/运行方式、功能介绍、配置项、架构说明的问题，
    都必须先调用本工具，再基于返回片段作答；不要直接用 read_project_file 读整篇 README。

    Args:
        query: 检索问题，用中文自然语言描述即可，例如“这个项目怎么启动”。
        top_k: 返回片段数；传 0 表示使用配置默认值（KNOWLEDGE_TOP_K）。
    """
    if not (query or "").strip():
        return "错误: query 不能为空，请描述要查询的文档问题"

    settings = get_settings()
    limit = int(top_k) if int(top_k or 0) > 0 else max(1, settings.knowledge_top_k)

    try:
        store, chunks, sources = _get_store(settings)
    except Exception as exc:  # noqa: BLE001 - 转成可读错误，不让整轮对话中断
        logger.warning("知识库检索不可用：%s", exc)
        return (
            f"错误: 知识库检索不可用: {exc}。请检查 RAG 依赖"
            "（langchain-community / faiss-cpu / sentence-transformers）与 EMBEDDING_MODEL 配置。"
        )

    try:
        hits = store.similarity_search_with_score(query, k=max(1, limit))
    except Exception as exc:  # noqa: BLE001
        logger.warning("知识库检索失败：%s", exc)
        return f"错误: 检索失败: {exc}"

    if not hits:
        return "知识库中未检索到相关内容，请换一种问法，或改用文件工具查看具体文件。"

    logger.info(
        "search_knowledge_base 命中 %s 段：query=%r 来源=%s",
        len(hits),
        query[:60],
        [doc.metadata.get("source") for doc, _ in hits],
    )

    max_chars = max(200, settings.knowledge_max_chars)
    lines = [f"检索到 {len(hits)} 个相关片段（知识库共 {chunks} 个分块 / {sources} 个文件）：", ""]
    for rank, (doc, score) in enumerate(hits, start=1):
        content = doc.page_content.strip()
        if len(content) > max_chars:
            content = content[:max_chars] + "…（片段已截断）"
        lines.append(
            f"[片段 {rank}] 来源: {doc.metadata.get('source', '?')}"
            f"（第 {doc.metadata.get('chunk', '?')} 块） 距离: {score:.4f}"
        )
        lines.append(content)
        lines.append("")

    lines.append("以上片段取自项目知识库，请基于这些内容作答并注明来源文件。")
    return "\n".join(lines)
