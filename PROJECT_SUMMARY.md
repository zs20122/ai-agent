# AI 研发助手 Agent —— 项目材料（简历描述 + 面试口述）

> 本文件中的每个数字与机制都对应仓库中的真实实现（见文末「材料与代码对应表」）。
> 最后核对时间：2026-09-30 ｜ 质量门禁实测：
> `ruff check .` → **All checks passed!** ｜ `ruff format --check .` → **全部文件已格式化（0 个待重排）**
> ｜ `python -m pytest -q` → **34 passed** ｜ `python -m pyright`（standard）→ **0 errors / 0 warnings**

---

## 一、简历项目描述（4 条 bullet）

**AI 研发助手 Agent** ｜ 个人项目 ｜ Python 3.10 · FastAPI · LangChain + LangGraph · Streamlit

- **基于 FastAPI + LangGraph 搭建分层 Agent 服务**：按 `api → services → agents → llm/tools` 单向依赖拆分为 28 个 Python 模块（接口层只做契约校验与 HTTP 错误语义，编排下沉到 service 层）；用 LangGraph `StateGraph` 编排 `planner → agent ⇄ tools → finalize` 工作流，条件边结合 `MAX_AGENT_STEPS` 限制工具调用轮次、`recursion_limit` 兜底防死循环；checkpointer 以 `thread_id = session_id` 提供多轮记忆；对外同时提供一次性 `POST /chat` 与 SSE 流式 `POST /chat/stream`（`start/progress/token/end/error` 五类事件，携带反缓冲响应头），并在未配置模型 Key 时保持服务正常启动（返回 503 / `ready` 为 `degraded`）。
- **实现 Function Calling 工具调用闭环**：用 `bind_tools` 把工具绑定到模型，tools 节点解析 AI 消息中的 `tool_calls` 逐个执行、按 `tool_call_id` 回填 `ToolMessage`（支持一轮多个工具调用）；未注册工具与工具执行异常统一降级为可读文本回填，从而保证 OpenAI 消息序列始终合法；工具集中在工具注册表注册，新增工具只改一处。
- **设计沙盒化安全机制**：文件工具通过 `resolve()` + 父目录校验拦截 `../` 路径穿越，并限制单文件 1 MB、单次读取 8000 字符、目录列出 200 条上限、跳过 `.venv/.git/__pycache__` 等目录；命令行工具**默认关闭**，开启后仅允许 `python/pytest/ruff/mypy/git` 白名单，采用 `shell=False` + 参数列表 + 固定工作目录 + 最长 60 秒强制超时，杜绝 `;`、`&&`、`|` 等 shell 元字符注入。
- **Streamlit 前端交付 + 工程化质量门禁**：中文气泡聊天页（`st.chat_message` / `st.chat_input`）、侧边栏切换后端地址与 `/ready` 状态探测（5 秒缓存）、会话 ID 管理与「执行计划 / 步数 / 模型」展开视图、全中文错误语义（未连接 / 503 / 超时）；全程在 **Ruff（lint + format）与 Pyright `standard` 零告警**约束下开发，并以 **34 个离线 pytest 用例**（假模型驱动，不需要 API Key、不访问网络）覆盖工具沙盒、工具调用闭环、步数上限降级、多轮记忆与 SSE 事件序列。

> **使用提示**：本项目没有真实用户量 / QPS / 准确率等运营指标，**不要自行补数字**。
> 版面紧张时优先保留第 1、3 条；第 2 条的「降级回填 / 消息序列合法」细节适合留到面试口述中展开。

---

## 二、面试口述介绍（约 2 分钟）

> 语速正常约 2 分钟；下面按时间分段，可按现场时间删减带补充说明的句子。

**① 定位（约 15 秒）**
「我用 Python 做了一个 AI 研发助手的 Agent 服务。后端是 FastAPI，工作流编排用 LangGraph，前端用 Streamlit 做了个聊天页。它能自己去列目录、读文件、按需执行白名单命令，然后基于结果回答；所有工具访问都被限制在一个工作目录的沙盒里。」

**② 架构主线（约 25 秒）**
「请求进来先过 FastAPI 接口层，接口层只做参数校验和 HTTP 错误语义转换，真正的业务在 service 层，再往下才是 LangGraph。图是四段：planner 先出一个简短计划，agent 做带工具的推理，如果模型返回了 `tool_calls` 就进 tools 节点执行、再回到 agent 形成循环，最后由 finalize 统一生成回答；条件边会卡最大步数防止无限循环。多轮记忆靠 checkpointer，我把 `thread_id` 直接绑成 `session_id`，所以同一个会话天然带上下文。对外我给了两个接口：一次性返回的 `/chat` 和 SSE 流式的 `/chat/stream`——流式会把 planner、agent 的中间输出作为 `progress` 事件、finalize 的逐字增量作为 `token` 事件下发。」

**③ 最大的难点（约 40 秒）**
「最难的是工具调用和步数上限撞在一起的那条路径。模型在最后一步返回了 `tool_calls`，但这时已经到上限，我强制进 finalize。第一次测是能出答案的，但我发现同一个会话**再问第二句就 400 了**——因为 OpenAI 要求带 `tool_calls` 的 assistant 消息后面必须紧跟对应数量的 tool 消息，续聊时历史被整体重放，序列非法，模型网关直接拒掉。我一开始只在收尾提示词里把这些悬空调用降级成文本，结果对当次回答有用、对状态没用：图状态里那条非法消息还在。最后改成两件事一起做——用 `RemoveMessage` 把悬空消息真从图状态里删掉，再写回一条降级后的文本，让模型知道『哪些工具没跑成』。这个不变量我还固化成测试断言：遍历整个消息列表，任何 `tool_calls` 都必须有配对的 `ToolMessage`，正常结束和超限收尾都必须成立。」

**④ 怎么排错（约 25 秒）**
「我的习惯是先分清『是代码错还是环境/工具错』，再动手改。比如流式那块我一直拿不到最终回答的逐字增量，我没有猜，而是先把 `astream` 的元数据打出来看，发现要按 `langgraph_node` 判断当前是 finalize 还是中间节点，并顺手加了兜底：拿不到增量就直接读图的最终状态，避免元数据缺失时返回空。还有一次遇到命令行检查 0 告警、编辑器面板却几十条的情况，我也没有直接按告警改代码，而是去读静态分析工具自己的日志，发现它在扫描工作区之外的临时脚本，于是把配置收敛成单一来源。总结起来就是：**先复现、先看证据，再改代码**，这样不会改出新问题。」

**⑤ 最有成就感的地方（约 15 秒）**
「最有成就感的不是把功能跑通，而是把『沙盒不能绕过、消息序列永远合法』这两件事用测试钉死了——34 个用例全部离线运行，不填 API Key、不联网就能跑，默认关闭的命令行工具、路径穿越、白名单、步数上限的降级路径都有覆盖。我也把 Ruff、Pyright 和 pytest 当成同一道门禁，每次改完都要求全绿，这才敢在上面继续加节点和工具。」

### 可能被追问的问题与回答要点

| 追问 | 回答要点 |
| --- | --- |
| 为什么手写 tools 节点，不用现成的 ReAct Agent？ | 需要精确控制步数上限、工具失败的回填格式，以及超限时对消息的清理；手写之后每一步都可断言——测试里就是直接断言模型被调用的次数和它实际看到的提示词内容。 |
| 沙盒真的防得住吗？ | 诚实说明边界：防路径穿越、防 shell 元字符注入是有效的；但它不是容器级隔离，白名单里的 `python` 本身仍可执行代码，所以默认关闭、只在可信环境开启，生产应换成容器或降权子进程。 |
| 会话记忆为什么用内存实现？ | 明确是开发期选择：`InMemorySaver` 重启即丢、多副本不共享；换 `SqliteSaver` / `PostgresSaver` 只需改 checkpointer 工厂一处，service 层无感知（扩展点已写进 README）。 |
| 流式为什么用 SSE 而不是 WebSocket？ | 场景是单向服务端推送，SSE 更简单、可复用普通 HTTP 与反向代理；代价是只能服务端到客户端，且必须处理代理缓冲（已在响应头里显式关闭缓冲）。 |
| 并发与性能怎么考虑？ | 说明现状：单实例、无鉴权、前端用同步 `requests`，属于骨架级；要上线会补并发限流、请求级超时预算、持久化检查点与离线评估集。 |

---

## 附录 A：材料与代码对应表（自查 / 答辩用）

| 材料中的说法 | 对应代码或测试 |
| --- | --- |
| 分层 `api → services → agents → llm/tools`，28 个模块 | `app/main.py`、`app/api/`、`app/services/agent_service.py`、`app/agents/`、`app/llm/factory.py`、`app/memory/`、`app/schemas/`、`app/core/` |
| 图编排 `planner → agent ⇄ tools → finalize` | `app/agents/graph.py`（`build_graph` / `get_graph`）、`app/agents/state.py` |
| 条件边限制步数 + 递归上限兜底 | `app/agents/nodes.py::route_after_agent`；`app/core/config.py`（`max_agent_steps=6`、`recursion_limit=max(steps*2+4, 10)`） |
| Function Calling 回填 `ToolMessage` | `app/agents/nodes.py::make_tools_node`（`tool_call_id` 配对；未注册工具与执行异常降级为 `错误: ...` 文本） |
| 超限时清理悬空 `tool_calls` | `app/agents/nodes.py::_sanitize_for_finalize` + `RemoveMessage`（README「工作流」有同段说明） |
| 多轮记忆（`thread_id = session_id`） | `app/services/agent_service.py::_run_config`；`app/memory/checkpointer.py`（`InMemorySaver`） |
| SSE 五类事件 + 反缓冲响应头 | `app/api/v1/endpoints/chat.py`（`SSE_HEADERS`、`event_stream`、`_format_sse`）；`app/services/agent_service.py::stream` |
| 未配置 Key 仍可启动（503 / `degraded`） | `app/api/v1/endpoints/chat.py`（`RuntimeError → 503`）、`app/api/v1/endpoints/health.py`、`app/llm/factory.py` |
| 路径穿越拦截、1 MB、8000 字符、200 条 | `app/agents/tools/file_ops.py`（`resolve_within_workspace`、`MAX_FILE_BYTES`、`_SKIP_DIRS`） |
| 命令白名单、`shell=False`、60 秒超时、默认关闭 | `app/agents/tools/shell.py`（`ALLOWED_COMMANDS`、`MAX_TIMEOUT_SECONDS`）、`app/agents/tools/registry.py`（按配置注册） |
| 前端 5 秒状态缓存、会话管理、计划展开 | `frontend/app.py`（`backend_status`、`new_session_id`、`render_message`） |
| 34 个离线用例 | `tests/`：`test_health.py`、`test_graph.py`、`test_tools.py`、`test_agent_service.py`、`test_frontend.py`（配 `fakes.py` 假模型） |
| Ruff / Pyright 配置单一来源 | `pyproject.toml`：`[tool.ruff]`（行宽 100、目标 py310）、`[tool.pyright]`（standard） |

---

## 附录 B：一键复核命令（在项目根目录执行）

```powershell
# 1) 静态检查与格式（期望：All checks passed! / already formatted）
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .

# 2) 类型检查（与 VS Code 的 Pylance 同源；离线或代理环境先复用系统 Node）
$env:PYRIGHT_PYTHON_GLOBAL_NODE='1'; .\.venv\Scripts\python.exe -m pyright

# 3) 离线测试（不需要 API Key，不访问网络）
.\.venv\Scripts\python.exe -m pytest -q
```

本次实测：`All checks passed!` ／ 全部文件均已格式化（0 个待重排） ／ `0 errors, 0 warnings` ／ `34 passed`。

> 关于 `ruff format` 的文件计数：`pyproject.toml` 未自定义 `include`，Ruff 的默认扫描范围除 Python 文件外还包含 Markdown 中的代码块，
> 所以它统计的是「40 个 `.py` + 仓库内的 `.md`」。这个数字会随环境漂移（例如 `.pytest_cache/` 被 `.gitignore` 忽略后，
> 其内部的 `README.md` 就不再计入，计数由 43 变 42）——**判断标准是「是否有文件需要重排」，应为 0**。
> 本文件的代码块以 `powershell` 标注，不参与 Python 格式化。

