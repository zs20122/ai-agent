"""AI 研发助手 Agent —— 应用包。

分层约定（依赖方向单向，禁止反向 import）：
    api -> services -> agents -> llm / agents.tools
    core / schemas 作为横切能力可被任意层引用
"""

__version__ = "0.1.0"
