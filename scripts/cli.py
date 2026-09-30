"""命令行调试入口。

用法（项目根目录下执行）::

    python scripts/cli.py "帮我看看 data/workspace 里有哪些文件"
    python scripts/cli.py "解释一下 app/agents/nodes.py" --stream
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 直接 `python scripts/cli.py` 时把项目根目录加入 sys.path，保证能 import app
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.schemas.chat import ChatRequest  # noqa: E402
from app.services.agent_service import AgentService  # noqa: E402


async def _run(question: str, session_id: str, stream: bool) -> int:
    service = AgentService()
    request = ChatRequest(message=question, session_id=session_id)

    if not stream:
        response = await service.chat(request)
        print("\n=== 执行计划 ===\n" + (response.plan or "<无>"))
        print("\n=== 最终回答 ===\n" + response.answer)
        print(f"\n(步数={response.steps}，模型={response.model})")
        return 0

    print("=== 流式输出 ===")
    async for chunk in service.stream(request):
        if chunk.event == "token":
            print(chunk.delta, end="", flush=True)
        elif chunk.event == "progress":
            text = chunk.delta.strip().replace("\n", " ")
            print(f"\n[{chunk.node}] {text[:200]}")
        elif chunk.event == "end":
            print("\n=== 结束 ===")
        elif chunk.event == "error":
            print(f"\n[错误] {chunk.delta}", file=sys.stderr)
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AI 研发助手 Agent 命令行入口")
    parser.add_argument("question", help="要提问的内容")
    parser.add_argument("--session-id", default="cli", help="会话 ID（相同 ID 共享多轮记忆）")
    parser.add_argument("--stream", action="store_true", help="使用 SSE 流式输出")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    if not settings.model_configured:
        print(
            "错误: 未配置 OPENAI_API_KEY，请先复制 .env.example 为 .env 并填写",
            file=sys.stderr,
        )
        return 2

    return asyncio.run(_run(args.question, args.session_id, args.stream))


if __name__ == "__main__":
    raise SystemExit(main())
