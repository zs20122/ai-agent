"""AI 研发助手 —— Streamlit 聊天前端。

作用：把 FastAPI 后端的 ``POST /api/v1/chat`` 包装成一个中文气泡聊天页面。

启动方式（需先启动后端）::

    # 终端 1：后端
    powershell -ExecutionPolicy Bypass -File scripts\\run_dev.ps1
    # 终端 2：前端
    .\\.venv\\Scripts\\python.exe -m streamlit run frontend/app.py

说明：
- 前端由 Streamlit 进程通过 ``requests`` 直接调用后端，属于服务端调用，
  因此不涉及浏览器跨域（CORS）；
- 同一个 ``session_id`` 对应后端 checkpointer 的同一 thread_id，可保留多轮记忆；
- **会话与历史全自动**，页面加载顺序固定为：读 URL 查询参数 → 拿到会话 ID
  （没有就新生成并写回地址栏）→ 拉取历史 → 渲染页面，因此按 F5 刷新即可看到
  之前的全部聊天记录，不需要任何手动操作；
- 历史来自 ``GET /api/v1/chat/history/{session_id}``（``CHECKPOINT_BACKEND=sqlite``
  时是落盘的历史），同一个会话只拉一次（由 ``history_session`` 记住），
  对话过程中的重跑不会覆盖已有气泡；
- **气泡只画「用户的提问」和「AI 的最终回答」**：后端默认就把 SystemMessage（系统提示词）、
  ToolMessage（工具结果）与带 tool_calls 的中间步骤过滤掉，前端再按轮次只保留最后一条回答，
  因此工具调用过程、检索片段、agent 草稿都不会出现成气泡（排查可用后端
  ``?include_internal=true``）；
- 侧边栏只保留「后端地址 / 连接状态 / 当前会话 ID」，以及一个「开始新会话」按钮；
- 对话只用非流式接口 ``/api/v1/chat``（SSE 流式接口 ``/api/v1/chat/stream`` 见 README）。
"""

from __future__ import annotations

import re
import uuid
from typing import Any
from urllib.parse import quote

import requests
import streamlit as st

DEFAULT_BACKEND = "http://127.0.0.1:8000"
CHAT_PATH = "/api/v1/chat"
HISTORY_PATH = "/api/v1/chat/history/{session_id}"
READY_PATH = "/api/v1/ready"
# 会话 ID 也是地址栏查询参数名；长度上限与后端 Path 校验保持一致（1-64）
SESSION_ID_QUERY_PARAM = "session_id"
SESSION_ID_MAX_LEN = 64
# 模型生成可能较慢（尤其带多轮工具调用），超时给足
REQUEST_TIMEOUT = 180
STATUS_TIMEOUT = 5
# 读历史只查一次会话存储，给个小超时即可
HISTORY_TIMEOUT = 30
# 前端最多载入最近多少条历史消息（后端上限 500）
HISTORY_LIMIT = 200
# 前端版本号：显示在侧边栏，便于确认浏览器跑的是不是最新代码
FRONTEND_REVISION = "r5 · 只渲染提问与最终回答"
# 内容为空 / 只有 HTML 时的兜底文案，保证气泡里一定看得见东西
EMPTY_CONTENT_HINT = "（这条历史消息没有文本内容）"
HTML_FALLBACK_HINT = "该内容主要是 HTML 标签（Markdown 默认会过滤掉），下面按源代码显示："
# 判定「只有 HTML」用：标签 / 注释 / 去掉它们之后的可见文本
_HTML_TAG_RE = re.compile(r"<[a-zA-Z!/][^>]*>")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# 角色别名：后端已归一化，这里再兜一层，避免角色不认识就把消息丢掉
_ROLE_ALIASES = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "tool": "tool",
    "function": "tool",
    "system": "system",
}


class FrontendError(Exception):
    """面向用户展示的错误（消息已是可直接阅读的中文）。"""


def new_session_id() -> str:
    """生成短会话 ID（后端限制最长 64 字符）。"""
    return uuid.uuid4().hex[:12]


def clean_session_id(value: Any) -> str:
    """规整会话 ID：去掉首尾空白。

    空值、超长（后端上限 64）、含空白或控制字符的值都视为不合法，返回空字符串，
    由调用方决定是「新生成」还是「提示用户」。会话 ID 会进入 URL 与后端路径，
    所以这里顺手做一次边界检查，避免把明显非法的值发给后端换回 422。
    """
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or len(text) > SESSION_ID_MAX_LEN:
        return ""
    if any(char.isspace() or ord(char) < 32 for char in text):
        return ""
    return text


def session_id_from_query(value: Any) -> str:
    """从 URL 查询参数里取会话 ID（``st.query_params.get`` 可能返回 list）。"""
    if isinstance(value, list):
        return clean_session_id(value[0]) if value else ""
    return clean_session_id(value)


def sync_url(session_id: str) -> None:
    """把会话 ID 写进浏览器地址栏的查询参数（刷新 / 分享链接都能回到同一会话）。"""
    if session_id and st.query_params.get(SESSION_ID_QUERY_PARAM) != session_id:
        st.query_params[SESSION_ID_QUERY_PARAM] = session_id


def switch_session(session_id: str) -> None:
    """切换到指定会话：清空气泡、写入 URL，下一次脚本运行会重新载入它的历史。

    ``history_session`` 置空表示「该会话的历史还没拉」，页面顶部的水合逻辑会重新拉取。
    """
    st.session_state.session_id = session_id
    st.session_state.messages = []
    st.session_state.history_session = None
    st.session_state.history_notice = ""
    st.session_state.history_error = ""
    st.session_state.history_empty = False
    st.session_state.history_count = 0
    st.session_state.history_filtered = 0
    st.session_state.history_bubbles = 0
    st.session_state.history_mismatch = False
    sync_url(session_id)


def resolve_session_id() -> str:
    """确定本次页面要用的会话 ID，并把状态初始化好（页面加载的第一步）。

    会话 ID 优先级：**URL 查询参数**（刷新 / 分享链接 / 手动改地址栏）> 本次浏览器会话
    已记住的 ID > 新生成。URL 上的 ID 与当前不一致时以 URL 为准，并清空气泡等重新拉历史；
    URL 上没有（或非法）时才生成新的，然后把它写回地址栏，保证下一次刷新还能回到同一会话。

    返回值就是最终生效的 ``session_id``，可直接交给 ``hydrate_history`` 使用。
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history_session" not in st.session_state:
        st.session_state.history_session = None
    if "history_notice" not in st.session_state:
        st.session_state.history_notice = ""
    if "history_error" not in st.session_state:
        st.session_state.history_error = ""
    if "history_empty" not in st.session_state:
        st.session_state.history_empty = False
    if "history_count" not in st.session_state:
        st.session_state.history_count = 0
    if "history_filtered" not in st.session_state:
        st.session_state.history_filtered = 0
    if "history_bubbles" not in st.session_state:
        st.session_state.history_bubbles = 0
    if "history_mismatch" not in st.session_state:
        st.session_state.history_mismatch = False

    current = str(st.session_state.get("session_id") or "")
    from_url = session_id_from_query(st.query_params.get(SESSION_ID_QUERY_PARAM))

    if from_url and from_url != current:
        switch_session(from_url)
    elif not current:
        st.session_state.session_id = new_session_id()

    session_id = str(st.session_state.session_id)
    sync_url(session_id)
    return session_id


def _detail_of(response: requests.Response) -> str:
    """从后端的错误响应里取出可读的 detail 文本。"""
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "").strip()[:300] or "无响应内容"
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)[:300]


def ask_backend(base_url: str, message: str, session_id: str) -> tuple[str, dict]:
    """调用后端对话接口。

    Args:
        base_url: 后端根地址，例如 ``http://127.0.0.1:8000``。
        message: 用户输入。
        session_id: 会话 ID（决定后端的多轮记忆分组）。

    Returns:
        ``(answer, meta)``；meta 含 ``plan`` / ``steps`` / ``model``，用于界面附带展示。

    Raises:
        FrontendError: 网络不通、超时或后端返回非 200。
    """
    url = base_url.rstrip("/") + CHAT_PATH
    payload = {"message": message, "session_id": session_id}

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectTimeout as exc:
        raise FrontendError(f"连接后端超时：{url}（后端是否已启动？）") from exc
    except requests.exceptions.ReadTimeout as exc:
        raise FrontendError(
            f"后端 {REQUEST_TIMEOUT} 秒内没有返回，模型可能仍在生成；请稍后重试或查看后端日志。"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise FrontendError(
            f"连不上后端 {url}。请先在另一个终端启动后端：\n"
            "`powershell -ExecutionPolicy Bypass -File scripts\\run_dev.ps1`"
        ) from exc
    except requests.exceptions.RequestException as exc:  # 兜底：其余 requests 异常
        raise FrontendError(f"请求失败：{exc}") from exc

    if response.status_code == 503:
        raise FrontendError(
            "后端未配置模型凭据（返回 503）。请在项目根目录的 `.env` 里填写 "
            "`OPENAI_API_KEY` 后重启后端。"
        )
    if response.status_code != 200:
        raise FrontendError(f"后端返回 {response.status_code}：{_detail_of(response)}")

    data = response.json()
    answer = str(data.get("answer") or "").strip()
    if not answer:
        answer = "（后端没有返回有效回答，请查看后端日志）"

    meta = {
        "model": str(data.get("model") or ""),
        "plan": str(data.get("plan") or "").strip(),
        "steps": int(data.get("steps") or 0),
    }
    return answer, meta


@st.cache_data(ttl=5, show_spinner=False)
def backend_status(base_url: str) -> tuple[bool, str]:
    """探测后端就绪状态（只读 ``/api/v1/ready``，结果缓存 5 秒，避免每次重跑都发请求）。"""
    url = base_url.rstrip("/") + READY_PATH
    try:
        response = requests.get(url, timeout=STATUS_TIMEOUT)
    except requests.exceptions.RequestException:
        return False, "未连接"
    if response.status_code != 200:
        return False, f"异常（HTTP {response.status_code}）"

    try:
        data = response.json()
    except ValueError:
        return True, "已连接"
    if data.get("model_configured"):
        return True, f"已连接（模型 {data.get('model', '-')}）"
    return True, "已连接（未配置 API Key，对话会返回 503）"


def fetch_history(base_url: str, session_id: str, limit: int = HISTORY_LIMIT) -> dict[str, Any]:
    """读取后端保存的会话历史。

    Args:
        base_url: 后端根地址，例如 ``http://127.0.0.1:8000``。
        session_id: 要读取的会话 ID（后端 checkpointer 的 thread_id）。
        limit: 最多取回最近多少条消息（后端上限 500）。

    Returns:
        后端返回的原始 JSON：``messages`` / ``total_messages`` / ``message_count`` /
        ``truncated`` / ``backend``。会话不存在时 ``messages`` 为空列表。

    Raises:
        FrontendError: 网络不通、超时或后端返回非 200。
    """
    url = base_url.rstrip("/") + HISTORY_PATH.format(session_id=quote(session_id, safe=""))

    try:
        response = requests.get(url, params={"limit": limit}, timeout=HISTORY_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        raise FrontendError(f"读取历史失败：{exc}") from exc

    if response.status_code != 200:
        raise FrontendError(f"后端返回 {response.status_code}：{_detail_of(response)}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise FrontendError("后端返回的历史不是合法 JSON") from exc

    return payload if isinstance(payload, dict) else {}


def history_to_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """把历史响应转成聊天气泡用的消息列表（**只有用户提问与 AI 最终回答**）。

    后端接口已经把 ``SystemMessage``（系统提示词）、``ToolMessage``（工具执行结果）与
    带 ``tool_calls`` 的中间步骤过滤掉了，前端再兜一层：按 user 消息切分轮次，
    每轮只取**最后一条有文字**的 assistant 消息作为回答；agent 的草稿与工具过程
    一律丢弃，不渲染成聊天气泡（需要看完整过程请直接调后端 ``?include_internal=true``）。

    保证：只要历史里有问答内容，气泡正文就一定非空；一轮里没有文字回答时只画用户的提问。
    """
    items = payload.get("messages") or []
    bubbles: list[dict[str, Any]] = []

    for turn in _split_turns(items):
        bubbles.append(
            {"role": "user", "content": turn["question"] or EMPTY_CONTENT_HINT, "meta": {}}
        )
        if turn["answer"] is not None:
            bubbles.append({"role": "assistant", "content": turn["answer"], "meta": {}})

    if not bubbles and items:
        # 兜底：历史里没有 user 消息时（只有 system / tool 等）也按原始顺序挑出能画的
        bubbles = _bubbles_from_raw(items)

    return bubbles


def _bubbles_from_raw(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """兜底渲染：按原始顺序挑出可渲染的消息，**依然只画用户提问与 AI 回答**。

    system / tool / 只调工具没有文字的 assistant 全部跳过；正文为空也跳过。
    因此返回空列表就说明这段历史里确实没有任何「问 & 答」内容（侧边栏会给出说明）。
    """
    bubbles: list[dict[str, Any]] = []

    for item in items:
        role = _normalize_role(item)
        content = str(item.get("content") or "").strip()

        if role not in ("user", "assistant") or item.get("tool_calls") or not content:
            continue
        if role == "assistant" and bubbles and bubbles[-1]["role"] == "assistant":
            bubbles.pop()  # 同一轮里只留最后一条回答（草稿被覆盖）
        bubbles.append({"role": role, "content": content, "meta": {}})

    return bubbles


def _normalize_role(item: dict[str, Any]) -> str:
    """归一化角色：优先 ``role``，兼容 ``type``；不认识的角色按助手内容处理。"""
    raw = str(item.get("role") or item.get("type") or "").strip().lower()
    return _ROLE_ALIASES.get(raw, "assistant")


def _split_turns(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 user 消息切分轮次，只挑出每轮的最终回答（内部过程不参与）。"""
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for item in items:
        role = _normalize_role(item)
        content = str(item.get("content") or "").strip()

        if role == "user":
            current = {"question": content, "answer": None}
            turns.append(current)
            continue
        if current is None:
            continue  # 没有归属轮次的消息（system / 工具过程等）不参与气泡

        if role == "assistant" and content and not item.get("tool_calls"):
            current["answer"] = content  # 同一轮里后出现的回答覆盖前面的草稿

    return turns


def hydrate_history(base_url: str, session_id: str) -> None:
    """从后端拉取指定会话的历史，并**写入 ``st.session_state.messages``** 供气泡渲染。

    设计要点：
    - 同一个会话只拉一次（用 ``history_session`` 记住），所以对话过程中的每次重跑
      都不会覆盖界面上已有的气泡；
    - 自愈：万一气泡列表被清空而历史还在（``history_count`` 有值），就重新拉一次，
      避免出现“有历史、界面却空白”的中间态；
    - 拉取失败只在侧边栏提示，不清空已有气泡、也不打断页面（后端没启动时刷新页面仍可用）；
    - 无论能否渲染成气泡，都把「原始条数 / 渲染气泡数」记进 session_state 并显示在侧边栏，
      这样一眼就能分清是后端没数据、还是前端没渲染。
    """
    already = st.session_state.get("history_session") == session_id
    bubbles_lost = (
        already
        and int(st.session_state.get("history_count") or 0) > 0
        and not st.session_state.messages
    )
    if already and not bubbles_lost:
        return

    st.session_state.history_session = session_id
    try:
        payload = fetch_history(base_url, session_id)
    except FrontendError as exc:
        st.session_state.history_error = str(exc)
        return

    count = int(payload.get("message_count") or 0)
    filtered = int(payload.get("filtered_messages") or 0)
    history_messages = history_to_messages(payload)

    st.session_state.history_error = ""
    st.session_state.messages = history_messages
    st.session_state.history_count = count
    st.session_state.history_filtered = filtered
    st.session_state.history_bubbles = len(history_messages)
    st.session_state.history_empty = not history_messages

    if not count:
        st.session_state.history_notice = ""
        st.session_state.history_mismatch = False
        return

    if not history_messages:
        # 后端有消息、前端却渲染不出气泡：明确报出来，不静默白屏
        st.session_state.history_mismatch = True
        st.session_state.history_notice = (
            f"已读取 {count} 条历史消息，但没有可渲染的对话内容（只有系统提示词或工具过程）"
        )
        return

    st.session_state.history_mismatch = False
    notice = f"已从后端载入 {count} 条对话消息"
    if filtered:
        notice += f"（另有 {filtered} 条系统提示 / 工具结果 / 中间步骤已过滤）"
    notice += f"，渲染 {len(history_messages)} 个气泡（{payload.get('backend', '-')} 后端）"
    if payload.get("truncated"):
        notice += f"；该会话共 {payload.get('total_messages', 0)} 条，这里只显示最近部分"
    st.session_state.history_notice = notice


def render_text(content: Any) -> None:
    """把一条消息的正文画进当前气泡，保证「有内容就一定看得见」。

    正常正文走 Markdown（``st.write`` 对字符串同样是 Markdown，这里直接用
    ``st.markdown`` 是为了能接管兜底）；如果正文**只有 HTML 标签**——``st.markdown``
    默认会过滤 HTML，气泡就会看起来完全空白——改成按源代码显示；正文为空则给提示语。
    """
    text = str(content or "").strip()
    if not text:
        st.caption(EMPTY_CONTENT_HINT)
        return
    if _looks_like_html_only(text):
        st.caption(HTML_FALLBACK_HINT)
        st.code(text, language=None)
        return
    st.markdown(text)


def _looks_like_html_only(text: str) -> bool:
    """判断正文是否「含 HTML 标签，但去掉标签与注释后几乎没有可见文字」。"""
    if not _HTML_TAG_RE.search(text):
        return False
    return not _HTML_TAG_RE.sub("", _HTML_COMMENT_RE.sub("", text)).strip()


def render_message(message: dict[str, Any]) -> None:
    """渲染一条消息：用户/助手气泡；助手消息可展开查看这一轮的执行计划。

    历史气泡只画「用户的提问」与「AI 的最终回答」——工具调用、检索片段、agent 草稿
    都是内部过程，既不进气泡也不在这里折叠展示（需要排查时直接调后端
    ``GET /api/v1/chat/history/{session_id}?include_internal=true``）。
    """
    is_user = str(message.get("role") or "") == "user"
    role = str(message.get("role") or "assistant")
    with st.chat_message(role, avatar="🧑" if is_user else "🤖"):
        render_text(message.get("content"))
        meta = message.get("meta") or {}
        if meta.get("plan"):
            with st.expander(
                f"执行计划（{meta.get('steps', 0)} 步，模型 {meta.get('model', '-')}）"
            ):
                render_text(meta["plan"])


def sidebar_connection() -> str:
    """侧边栏「连接设置」：后端地址 + 连接状态，返回本次要用的后端地址。"""
    with st.sidebar:
        st.subheader("连接设置")
        base_url = st.text_input("后端地址", value=DEFAULT_BACKEND)

        ok, status_text = backend_status(base_url)
        if ok:
            st.success(f"后端状态：{status_text}")
        else:
            st.error(f"后端状态：{status_text}")
            st.caption(
                "请先启动后端：`powershell -ExecutionPolicy Bypass -File scripts\\run_dev.ps1`"
            )
    return base_url


def sidebar_session(session_id: str) -> None:
    """侧边栏「当前会话」：显示会话 ID、历史载入结果与诊断信息，并提供「开始新会话」。

    历史不需要任何手动操作——页面每次加载都会自动拉取当前会话的历史，
    所以这里只展示状态：失败提示、载入条数（原始条数 / 渲染气泡数），或空会话说明。
    """
    with st.sidebar:
        st.divider()
        st.caption(
            f"会话 ID：`{session_id}`（相同 ID 共享多轮记忆，已写入地址栏 "
            f"`?{SESSION_ID_QUERY_PARAM}=`；刷新即自动恢复历史）"
        )
        if st.button("开始新会话"):
            switch_session(new_session_id())
            st.rerun()

        if st.session_state.history_error:
            st.error(f"历史载入失败：{st.session_state.history_error}")
        elif st.session_state.history_mismatch:
            st.warning(st.session_state.history_notice)
        elif st.session_state.history_notice:
            st.success(st.session_state.history_notice)
        elif st.session_state.history_empty:
            st.caption(f"会话 `{session_id}` 暂无历史消息。")

        # 诊断信息：返回的对话条数、被过滤的内部条数与气泡数分开显示，出现空白时一眼看出卡在哪一步
        filtered = int(st.session_state.history_filtered)
        filtered_detail = f"（另有 {filtered} 条内部消息已过滤）" if filtered else ""
        st.caption(
            f"历史诊断：后端返回 {int(st.session_state.history_count)} 条对话消息"
            f"{filtered_detail} → 渲染 {int(st.session_state.history_bubbles)} 个气泡；"
            f"前端版本 {FRONTEND_REVISION}"
        )

        st.divider()
        st.caption(
            "接口为一次性返回，最长等待 180 秒；气泡只显示提问与最终回答，"
            "工具过程请查看后端日志或用 `?include_internal=true` 调历史接口。"
        )


def render_chat(base_url: str) -> None:
    """渲染界面：**先**把 ``st.session_state.messages`` 里的每一条画成气泡，再放输入框。

    注意：这个循环在每次重跑（页面刷新、按钮点击、提交问题）都会完整执行一遍，
    历史气泡因此始终从 session_state 重建，F5 之后也能直接看到过去的问答。
    """
    for message in st.session_state.messages:
        render_message(message)

    prompt = st.chat_input("请输入你的问题，例如：data/workspace 里有哪些文件？")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt, "meta": {}})
    with st.chat_message("user", avatar="🧑"):
        render_text(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Agent 正在思考……"):
            try:
                answer, meta = ask_backend(base_url, prompt, st.session_state.session_id)
            except FrontendError as exc:
                answer, meta = f"⚠️ {exc}", {}
        render_text(answer)
        if meta.get("plan"):
            with st.expander(
                f"执行计划（{meta.get('steps', 0)} 步，模型 {meta.get('model', '-')}）"
            ):
                render_text(meta["plan"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})


def main() -> None:
    """页面入口，固定按「读 URL → 拿到会话 ID → 拉历史 → 渲染页面」执行。

    这样浏览器里按 F5 时，URL 上的 ``?session_id=`` 会先被读回，
    历史随即从后端自动加载并渲染，用户看不到任何输入框或按钮操作。
    """
    st.set_page_config(page_title="AI 研发助手", page_icon="🤖", layout="centered")
    st.title("AI 研发助手")

    # ① 读 URL 查询参数 → ② 拿到会话 ID（URL 里没有就新生成并写回地址栏）
    session_id = resolve_session_id()

    # 侧边栏：后端地址 + 连接状态（历史与对话都发给它）
    base_url = sidebar_connection()

    # ③ 自动拉取该会话历史（同一会话只拉一次，失败只在侧边栏提示、不打断页面）
    hydrate_history(base_url, session_id)

    # 侧边栏：当前会话信息 + 「开始新会话」
    sidebar_session(session_id)

    # ④ 渲染页面：历史气泡 + 输入框
    render_chat(base_url)


if __name__ == "__main__":
    main()
