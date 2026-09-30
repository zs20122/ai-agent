"""文件类工具：只允许访问 WORKSPACE_DIR 内部，防止路径穿越。

返回给模型的内容一律是字符串（成功为结果，失败为“错误: ...”说明），
这样模型能看到失败原因并自行纠正，而不是让整个图因异常中断。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from app.core.config import get_settings

# 单个文件最大读取字节数，避免把巨大文件塞进上下文
MAX_FILE_BYTES = 1_000_000
# 列目录时跳过的目录
_SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".pytest_cache"}


def workspace_root() -> Path:
    """返回（并确保存在）工作目录绝对路径。"""
    root = get_settings().workspace_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_within_workspace(relative_path: str) -> Path:
    """把相对路径解析到工作目录内，越界则抛 ValueError。"""
    root = workspace_root()
    cleaned = (relative_path or "").strip().strip('"').strip("'")
    candidate = (root / cleaned).resolve()

    if candidate != root and root not in candidate.parents:
        raise ValueError(f"路径越界：只允许访问工作目录 {root} 内的文件（收到 {relative_path!r}）")
    return candidate


@tool
def list_project_files(subdir: str = "", max_entries: int = 200) -> str:
    """列出工作目录下的文件（相对路径）。

    Args:
        subdir: 相对工作目录的子目录，留空表示工作目录根。
        max_entries: 最多返回的条目数，默认 200。
    """
    try:
        base = resolve_within_workspace(subdir)
    except ValueError as exc:
        return f"错误: {exc}"

    if not base.exists():
        return f"错误: 目录不存在: {subdir or '.'}"
    if not base.is_dir():
        return f"错误: 不是目录: {subdir}"

    root = workspace_root()
    entries: list[str] = []
    try:
        for item in sorted(base.rglob("*")):
            if len(entries) >= max(1, int(max_entries)):
                entries.append("... （已达 max_entries 上限，结果被截断）")
                break
            if any(part in _SKIP_DIRS for part in item.relative_to(root).parts):
                continue
            if item.is_file():
                entries.append(item.relative_to(root).as_posix())
    except OSError as exc:
        return f"错误: 列目录失败: {exc}"

    if not entries:
        return f"目录 {subdir or '.'} 下没有文件。"
    return "\n".join(entries)


@tool
def read_project_file(path: str, max_chars: int = 8000) -> str:
    """读取工作目录下某个文本文件的内容。

    Args:
        path: 相对工作目录的文件路径，例如 app/main.py。
        max_chars: 最多返回的字符数，默认 8000。
    """
    try:
        target = resolve_within_workspace(path)
    except ValueError as exc:
        return f"错误: {exc}"

    if not target.exists():
        return f"错误: 文件不存在: {path}"
    if not target.is_file():
        return f"错误: 不是文件: {path}"

    try:
        size = target.stat().st_size
        if size > MAX_FILE_BYTES:
            return f"错误: 文件过大（{size} 字节 > {MAX_FILE_BYTES} 字节），请改用其他方式处理"
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"错误: 读取失败: {exc}"

    limit = max(1, int(max_chars))
    if len(content) > limit:
        return content[:limit] + f"\n... （内容已截断，仅返回前 {limit} 个字符）"
    return content


FILE_TOOLS = [list_project_files, read_project_file]
