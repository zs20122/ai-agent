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
- 只用非流式接口 ``/api/v1/chat``（SSE 流式接口 ``/api/v1/chat/stream`` 见 README）。
"""

from __future__ import annotations

import uuid

import requests
import streamlit as st

DEFAULT_BACKEND = "http://127.0.0.1:8000"
CHAT_PATH = "/api/v1/chat"
READY_PATH = "/api/v1/ready"
# 模型生成可能较慢（尤其带多轮工具调用），超时给足
REQUEST_TIMEOUT = 180
STATUS_TIMEOUT = 5


class FrontendError(Exception):
    """面向用户展示的错误（消息已是可直接阅读的中文）。"""


def new_session_id() -> str:
    """生成短会话 ID（后端限制最长 64 字符）。"""
    return uuid.uuid4().hex[:12]


def init_state() -> None:
    """初始化会话状态：会话 ID + 消息历史。"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []


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


def render_message(message: dict) -> None:
    """渲染一条消息：用户/助手气泡，助手消息可展开查看执行计划。"""
    is_user = message.get("role") == "user"
    with st.chat_message(message.get("role", "assistant"), avatar="🧑" if is_user else "🤖"):
        st.markdown(message.get("content", ""))
        meta = message.get("meta") or {}
        if meta.get("plan"):
            with st.expander(
                f"执行计划（{meta.get('steps', 0)} 步，模型 {meta.get('model', '-')}）"
            ):
                st.markdown(meta["plan"])


def main() -> None:
    st.set_page_config(page_title="AI 研发助手", page_icon="🤖", layout="centered")
    init_state()

    st.title("AI 研发助手")

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

        st.divider()
        st.caption(f"会话 ID：`{st.session_state.session_id}`（相同 ID 共享多轮记忆）")
        if st.button("开始新会话"):
            st.session_state.session_id = new_session_id()
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("接口为一次性返回，最长等待 180 秒；工具调用过程可查看后端日志。")

    for message in st.session_state.messages:
        render_message(message)

    prompt = st.chat_input("请输入你的问题，例如：data/workspace 里有哪些文件？")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Agent 正在思考……"):
            try:
                answer, meta = ask_backend(base_url, prompt, st.session_state.session_id)
            except FrontendError as exc:
                answer, meta = f"⚠️ {exc}", {}
        st.markdown(answer)
        if meta.get("plan"):
            with st.expander(
                f"执行计划（{meta.get('steps', 0)} 步，模型 {meta.get('model', '-')}）"
            ):
                st.markdown(meta["plan"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})


if __name__ == "__main__":
    main()
