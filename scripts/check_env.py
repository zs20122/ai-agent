"""环境体检脚本：由 scripts/run_dev.ps1 在启动前调用。

设计约束：
- 只依赖标准库，保证「尚未安装项目依赖的解释器」也能运行；
- 结论只写 stdout（不写 stderr），便于启动脚本直接展示，避免 PowerShell 5.1
  把原生命令的 stderr 变成终止性 NativeCommandError；
- 退出码：0 = 依赖齐全；1 = 缺少或损坏依赖。

用法::

    python scripts/check_env.py               # 快速探测（只查模块是否存在，毫秒级）
    python scripts/check_env.py --verbose     # 人类可读的详细报告（版本号）
    python scripts/check_env.py --verbose --imports   # 额外做真实导入，可发现损坏的安装
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from importlib.metadata import PackageNotFoundError, version

# 启动服务必需
REQUIRED = ("uvicorn", "fastapi", "langgraph")
# 可选，仅展示状态
OPTIONAL = ("langchain", "langchain-openai", "openai", "httpx", "watchfiles")

DIST_NAME = {"langchain-openai": "langchain-openai"}


def _missing_modules() -> list[str]:
    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def _broken_modules() -> list[str]:
    broken: list[str] = []
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - 体检脚本，任何异常都要如实报告
            broken.append(f"{name}({type(exc).__name__}: {exc})")
    return broken


def _package_version(name: str) -> str:
    try:
        return version(DIST_NAME.get(name, name))
    except PackageNotFoundError:
        return "未安装"
    except Exception:  # noqa: BLE001
        return "未知"


def main(argv: list[str]) -> int:
    verbose = "--verbose" in argv
    do_imports = "--imports" in argv

    if verbose:
        print(f"INTERPRETER: {sys.executable}")
        print(f"PYTHON     : {sys.version.split()[0]}")
        for name in REQUIRED:
            print(f"  [必需] {name} {_package_version(name)}")
        for name in OPTIONAL:
            note = "（缺失时 --reload 退化为轮询，功能不受影响）" if name == "watchfiles" else ""
            print(f"  [可选] {name} {_package_version(name)}{note}")

    if do_imports:
        broken = _broken_modules()
        if broken:
            print("DEPS_BROKEN: " + ", ".join(broken))
            return 1

    missing = _missing_modules()
    if missing:
        print("DEPS_MISSING: " + ", ".join(missing))
        return 1

    print("DEPS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
