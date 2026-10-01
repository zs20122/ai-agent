"""工具层测试：路径越界防护、读取截断、命令白名单。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.agents.tools.file_ops import (
    list_project_files,
    read_project_file,
    resolve_within_workspace,
    workspace_root,
)
from app.agents.tools.registry import get_tools
from app.agents.tools.shell import ALLOWED_COMMANDS, run_whitelisted_command
from app.core.config import reset_settings_cache


def test_workspace_root_is_created(tmp_path: Path, monkeypatch: Any) -> None:
    target = tmp_path / "nested" / "workspace"
    monkeypatch.setenv("WORKSPACE_DIR", str(target))
    reset_settings_cache()

    assert not target.exists()
    assert workspace_root() == target.resolve()
    assert target.exists()


def test_read_and_list_files(workspace: Path) -> None:
    (workspace / "a.txt").write_text("hello", encoding="utf-8")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "b.py").write_text("print(1)", encoding="utf-8")

    assert read_project_file.invoke({"path": "a.txt"}) == "hello"

    listing = list_project_files.invoke({"subdir": ""})
    assert "a.txt" in listing
    assert "sub/b.py" in listing


def test_read_truncates_long_content(workspace: Path) -> None:
    (workspace / "big.txt").write_text("x" * 100, encoding="utf-8")

    output = read_project_file.invoke({"path": "big.txt", "max_chars": 10})

    assert output.startswith("x" * 10)
    assert "内容已截断" in output


def test_path_traversal_is_blocked(workspace: Path) -> None:
    with pytest.raises(ValueError):
        resolve_within_workspace("../outside.txt")

    assert read_project_file.invoke({"path": "../outside.txt"}).startswith("错误:")
    assert list_project_files.invoke({"subdir": ".."}).startswith("错误:")


def test_missing_file_returns_message(workspace: Path) -> None:
    assert read_project_file.invoke({"path": "nope.txt"}).startswith("错误: 文件不存在")


def test_shell_tool_disabled_by_default(workspace: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("ENABLE_SHELL_TOOL", "false")
    reset_settings_cache()

    output = run_whitelisted_command.invoke({"command": "python --version"})

    assert output.startswith("错误: 命令行工具未启用")


def test_shell_tool_rejects_non_whitelisted(workspace: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("ENABLE_SHELL_TOOL", "true")
    reset_settings_cache()

    output = run_whitelisted_command.invoke({"command": "curl http://example.com"})

    assert "不在白名单内" in output
    assert "python" in ALLOWED_COMMANDS


def test_registry_respects_shell_flag(workspace: Path, monkeypatch: Any) -> None:
    assert {tool.name for tool in get_tools()} == {
        "list_project_files",
        "read_project_file",
        "search_knowledge_base",
    }

    monkeypatch.setenv("ENABLE_SHELL_TOOL", "true")
    reset_settings_cache()

    assert "run_whitelisted_command" in {tool.name for tool in get_tools()}
