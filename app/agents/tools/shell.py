"""命令行工具（默认关闭，高风险，请谨慎开启）。

安全约束：
1. 只有 ``ENABLE_SHELL_TOOL=true`` 时才注册到工具列表；
2. 只允许白名单里的第一个参数（可执行文件名）；
3. 使用 ``shell=False`` + 参数列表执行，且工作目录固定为 WORKSPACE_DIR，
   避免 ``;`` ``&&`` ``|`` 等 shell 元字符被解释；
4. 强制超时，超时即杀掉子进程。
"""

from __future__ import annotations

import shlex
import subprocess

from langchain_core.tools import BaseTool, tool

from app.agents.tools.file_ops import workspace_root
from app.core.config import get_settings

# 允许执行的命令（仅命令名，不含路径与参数）
ALLOWED_COMMANDS = {"python", "pytest", "ruff", "mypy", "git"}

MAX_TIMEOUT_SECONDS = 60


@tool
def run_whitelisted_command(command: str, timeout_seconds: int = 20) -> str:
    """在工作目录内执行白名单命令并返回输出。

    仅当 ENABLE_SHELL_TOOL=true 时可用。允许的命令：python、pytest、ruff、mypy、git。

    Args:
        command: 完整命令字符串，例如 "python --version"。
        timeout_seconds: 超时秒数，最大 60。
    """
    settings = get_settings()
    if not settings.enable_shell_tool:
        return "错误: 命令行工具未启用（需要设置 ENABLE_SHELL_TOOL=true）"

    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return f"错误: 命令解析失败: {exc}"
    if not argv:
        return "错误: 命令为空"

    program = argv[0].strip()
    if program not in ALLOWED_COMMANDS:
        return f"错误: 命令 {program!r} 不在白名单内，允许：{sorted(ALLOWED_COMMANDS)}"

    timeout = min(max(1, int(timeout_seconds)), MAX_TIMEOUT_SECONDS)
    try:
        completed = subprocess.run(  # noqa: S603 - shell=False 且已做白名单校验
            argv,
            cwd=str(workspace_root()),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"错误: 命令执行超时（>{timeout} 秒）"
    except OSError as exc:
        return f"错误: 命令执行失败: {exc}"

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    return (
        f"退出码: {completed.returncode}\nstdout:\n{stdout or '<空>'}\nstderr:\n{stderr or '<空>'}"
    )


def build_shell_tool() -> BaseTool:
    """返回命令行工具实例（注册前请确认配置已开启）。"""
    return run_whitelisted_command
