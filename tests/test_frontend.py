"""前端（frontend/app.py）离线测试：请求构造、响应解析与错误语义。

全部离线：用 monkeypatch 替换 ``requests.post`` / ``requests.get``，不访问网络。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pytest
import requests
from streamlit.testing.v1 import AppTest

from frontend import app as frontend

APP_FILE = Path(frontend.__file__).resolve()


class _FakeResponse:
    """最小可用的假响应对象（只需 status_code / text / json()）。"""

    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("响应不是合法 JSON")
        return self._payload


@pytest.fixture(autouse=True)
def _clear_status_cache() -> Any:
    """backend_status 带 st.cache_data，用例之间必须清缓存，否则会串数据。"""
    frontend.backend_status.clear()
    yield
    frontend.backend_status.clear()


def _patch_post(monkeypatch: pytest.MonkeyPatch, response: Any, captured: dict[str, Any]) -> None:
    def fake_post(url: str, json: Any = None, timeout: int = 0) -> Any:
        captured.update(url=url, json=json, timeout=timeout)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(frontend.requests, "post", fake_post)


def test_ask_backend_builds_request_and_parses_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch_post(
        monkeypatch,
        _FakeResponse(
            200,
            {
                "session_id": "s1",
                "answer": "  你好，我是 AI 研发助手  ",
                "plan": "1. 列目录",
                "steps": 2,
                "model": "deepseek-chat",
            },
        ),
        captured,
    )

    # 尾部斜杠不应产生 //api/v1/chat
    answer, meta = frontend.ask_backend("http://127.0.0.1:8000/", "hi", "s1")

    assert answer == "你好，我是 AI 研发助手"
    assert meta == {"model": "deepseek-chat", "plan": "1. 列目录", "steps": 2}
    assert captured["url"] == "http://127.0.0.1:8000/api/v1/chat"
    assert captured["json"] == {"message": "hi", "session_id": "s1"}
    assert captured["timeout"] == frontend.REQUEST_TIMEOUT


def test_ask_backend_blank_answer_falls_back_to_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_post(monkeypatch, _FakeResponse(200, {"answer": "   "}), {})

    answer, meta = frontend.ask_backend("http://127.0.0.1:8000", "hi", "s1")

    assert "没有返回有效回答" in answer
    assert meta["steps"] == 0
    assert meta["plan"] == ""


def test_ask_backend_connection_error_suggests_starting_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_post(monkeypatch, requests.exceptions.ConnectionError("refused"), {})

    with pytest.raises(frontend.FrontendError) as excinfo:
        frontend.ask_backend("http://127.0.0.1:8000", "hi", "s1")

    message = str(excinfo.value)
    assert "连不上后端" in message
    assert "run_dev.ps1" in message


def test_ask_backend_read_timeout_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_post(monkeypatch, requests.exceptions.ReadTimeout("slow"), {})

    with pytest.raises(frontend.FrontendError) as excinfo:
        frontend.ask_backend("http://127.0.0.1:8000", "hi", "s1")

    assert "没有返回" in str(excinfo.value)


def test_ask_backend_503_explains_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_post(monkeypatch, _FakeResponse(503, {"detail": "未配置 OPENAI_API_KEY"}), {})

    with pytest.raises(frontend.FrontendError) as excinfo:
        frontend.ask_backend("http://127.0.0.1:8000", "hi", "s1")

    message = str(excinfo.value)
    assert "503" in message
    assert "OPENAI_API_KEY" in message


def test_ask_backend_other_error_status_keeps_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_post(monkeypatch, _FakeResponse(500, {"detail": "内部错误"}), {})

    with pytest.raises(frontend.FrontendError) as excinfo:
        frontend.ask_backend("http://127.0.0.1:8000", "hi", "s1")

    message = str(excinfo.value)
    assert "500" in message
    assert "内部错误" in message


def test_ask_backend_non_json_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_post(monkeypatch, _FakeResponse(502, None, "<html>Bad Gateway</html>"), {})

    with pytest.raises(frontend.FrontendError) as excinfo:
        frontend.ask_backend("http://127.0.0.1:8000", "hi", "s1")

    assert "Bad Gateway" in str(excinfo.value)


def test_backend_status_reports_disconnected(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, timeout: int = 0) -> Any:
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(frontend.requests, "get", fake_get)

    ok, text = frontend.backend_status("http://127.0.0.1:8000")

    assert ok is False
    assert text == "未连接"


def test_backend_status_reports_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        frontend.requests,
        "get",
        lambda url, timeout=0: _FakeResponse(
            200, {"status": "ready", "model": "deepseek-chat", "model_configured": True}
        ),
    )

    ok, text = frontend.backend_status("http://127.0.0.1:8000")

    assert ok is True
    assert "deepseek-chat" in text


def test_backend_status_warns_when_model_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        frontend.requests,
        "get",
        lambda url, timeout=0: _FakeResponse(
            200, {"status": "degraded", "model": "", "model_configured": False}
        ),
    )

    ok, text = frontend.backend_status("http://127.0.0.1:8000")

    assert ok is True
    assert "503" in text


def test_new_session_id_is_short_and_unique() -> None:
    first = frontend.new_session_id()
    second = frontend.new_session_id()

    assert len(first) == 12
    assert first != second
    assert len(first) <= 64  # 后端 session_id 上限


def _patch_get(monkeypatch: pytest.MonkeyPatch, response: Any, captured: dict[str, Any]) -> None:
    def fake_get(url: str, params: Any = None, timeout: int = 0) -> Any:
        captured.update(url=url, params=params, timeout=timeout)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(frontend.requests, "get", fake_get)


def test_fetch_history_builds_url_and_passes_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    payload = {
        "session_id": "demo",
        "backend": "sqlite",
        "total_messages": 2,
        "message_count": 2,
        "truncated": True,
        "messages": [{"role": "user", "content": "你好"}],
    }
    _patch_get(monkeypatch, _FakeResponse(200, payload), captured)

    # 尾部斜杠不应产生 //api/v1/...；会话 ID 需要做 URL 编码
    result = frontend.fetch_history("http://127.0.0.1:8000/", "demo 会话/1", limit=50)

    assert result == payload
    assert (
        captured["url"] == "http://127.0.0.1:8000/api/v1/chat/history/demo%20%E4%BC%9A%E8%AF%9D%2F1"
    )
    assert captured["params"] == {"limit": 50}
    assert captured["timeout"] == frontend.HISTORY_TIMEOUT


def test_fetch_history_empty_session_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, _FakeResponse(200, {"session_id": "new", "messages": []}), {})

    result = frontend.fetch_history("http://127.0.0.1:8000", "new")

    assert result["messages"] == []


def test_fetch_history_connection_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, requests.exceptions.ConnectionError("refused"), {})

    with pytest.raises(frontend.FrontendError) as excinfo:
        frontend.fetch_history("http://127.0.0.1:8000", "demo")

    assert "读取历史失败" in str(excinfo.value)


def test_fetch_history_error_status_keeps_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, _FakeResponse(503, {"detail": "不支持的 CHECKPOINT_BACKEND"}), {})

    with pytest.raises(frontend.FrontendError) as excinfo:
        frontend.fetch_history("http://127.0.0.1:8000", "demo")

    message = str(excinfo.value)
    assert "503" in message
    assert "CHECKPOINT_BACKEND" in message


def test_fetch_history_non_json_and_non_dict_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, _FakeResponse(502, None, "<html>Bad Gateway</html>"), {})
    with pytest.raises(frontend.FrontendError) as excinfo:
        frontend.fetch_history("http://127.0.0.1:8000", "demo")
    assert "Bad Gateway" in str(excinfo.value)

    _patch_get(monkeypatch, _FakeResponse(200, ["不是对象"]), {})
    assert frontend.fetch_history("http://127.0.0.1:8000", "demo") == {}


def test_history_to_messages_keeps_only_question_and_final_answer() -> None:
    """同一轮的多条 assistant 消息：只留最后一条当回答，工具过程与草稿不进气泡。"""
    payload = {
        "messages": [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "", "tool_calls": ["list_project_files"]},
            {
                "role": "tool",
                "content": "data/workspace 下的文件：demo.py",
                "name": "list_project_files",
            },
            {"role": "assistant", "content": "推理结论"},
            {"role": "assistant", "content": "第一轮回答"},
            {"role": "user", "content": "第二轮问题"},
            {"role": "assistant", "content": "第二轮回答"},
        ]
    }

    bubbles = frontend.history_to_messages(payload)

    assert [(item["role"], item["content"]) for item in bubbles] == [
        ("user", "第一轮问题"),
        ("assistant", "第一轮回答"),
        ("user", "第二轮问题"),
        ("assistant", "第二轮回答"),
    ]
    assert all(item["meta"] == {} for item in bubbles), "内部过程不再写进气泡的 meta"
    rendered = "\n".join(str(item["content"]) for item in bubbles)
    for internal in ["我是助手", "你是助手", "list_project_files", "推理结论"]:
        assert internal not in rendered, "系统提示词 / 工具过程 / 草稿都不能出现在气泡里"


def test_history_to_messages_skips_internal_process_messages() -> None:
    """只有 system / tool / 只调工具的 assistant 时，不画任何气泡（内部过程不是聊天内容）。"""
    assert frontend.history_to_messages({}) == []

    assert frontend.history_to_messages({"messages": [{"role": "system", "content": "说明"}]}) == []

    bubbles = frontend.history_to_messages(
        {
            "messages": [
                {"role": "system", "content": "你是助手"},
                {"role": "assistant", "content": "", "tool_calls": ["search_knowledge_base"]},
                {"role": "tool", "content": "片段：RAG 链路", "name": "search_knowledge_base"},
            ]
        }
    )

    assert bubbles == [], "这段历史里没有任何「问 & 答」，气泡列表应为空（侧边栏会说明原因）"


def test_history_to_messages_renders_real_backend_shape() -> None:
    """真实后端那种 7 条形状只画「提问 + 最终回答」两个气泡（工具过程不显示）。"""
    payload = _seven_message_history("2fe17b27f503")

    bubbles = frontend.history_to_messages(payload)

    assert [(item["role"], item["content"]) for item in bubbles] == [
        ("user", "请检索项目文档，告诉我 RAG 链路是怎么设计的？"),
        ("assistant", "# 项目的 RAG 链路设计"),
    ]
    assert all(str(item["content"]).strip() for item in bubbles), (
        "气泡正文不能为空，否则界面上看起来就是空白"
    )


def test_history_to_messages_never_returns_blank_bubbles() -> None:
    """没有文字回答的轮次只画用户的提问；空正文用提示语兜底，绝不出现空气泡。"""
    payload = {
        "messages": [
            {"role": "user", "content": "帮我看看目录"},
            {"role": "assistant", "content": "   ", "tool_calls": ["list_project_files"]},
            {"role": "tool", "content": "demo.py", "name": "list_project_files"},
        ]
    }

    only_question = frontend.history_to_messages(payload)

    assert [(item["role"], item["content"]) for item in only_question] == [
        ("user", "帮我看看目录")
    ], "这一轮没有文字回答，就不画助手气泡（不拿工具过程凑数）"

    blank_question = frontend.history_to_messages(
        {"messages": [{"role": "user", "content": "   "}, {"role": "assistant", "content": "回答"}]}
    )

    assert [item["content"] for item in blank_question] == [frontend.EMPTY_CONTENT_HINT, "回答"]
    assert all(str(item["content"]).strip() for item in blank_question)


def test_history_to_messages_normalizes_legacy_roles() -> None:
    """兼容 LangChain 原生角色名（human / ai）与只带 ``type`` 字段的消息。"""
    payload = {
        "messages": [
            {"type": "human", "content": "问题"},
            {"type": "ai", "content": "回答"},
        ]
    }

    assert [(item["role"], item["content"]) for item in frontend.history_to_messages(payload)] == [
        ("user", "问题"),
        ("assistant", "回答"),
    ]


def test_looks_like_html_only_detects_invisible_markdown() -> None:
    """纯 HTML 正文会被 Markdown 过滤成空白，必须能识别出来改成代码块显示。"""
    assert frontend._looks_like_html_only("<div><span></span></div>")
    assert frontend._looks_like_html_only("<!-- 只有注释 -->")
    assert frontend._looks_like_html_only('<img src="a.png">')
    assert not frontend._looks_like_html_only("# 正常 Markdown 标题")
    assert not frontend._looks_like_html_only("<p>有可见文字</p>")
    assert not frontend._looks_like_html_only("```html\n<div>代码块里的 HTML 不会被过滤</div>\n```")


def test_clean_session_id_keeps_trimmed_valid_values() -> None:
    assert frontend.clean_session_id("  demo  ") == "demo"
    assert frontend.clean_session_id("f932c20ea340") == "f932c20ea340"
    assert frontend.clean_session_id("我的会话") == "我的会话"
    assert frontend.clean_session_id("a" * frontend.SESSION_ID_MAX_LEN) == "a" * 64


def test_clean_session_id_rejects_invalid_values() -> None:
    """空值 / 非字符串 / 超长 / 含空白或控制字符的会话 ID 都视为不合法。"""
    for value in ["", "   ", None, 123, ["demo"], "a b", "a\nb", "a" * 65, "de\tmo"]:
        assert frontend.clean_session_id(value) == "", f"{value!r} 应被判为不合法"


def test_session_id_from_query_handles_list_and_missing() -> None:
    """URL 查询参数可能是 str / list / None，统一收敛成合法 ID 或空串。"""
    assert frontend.session_id_from_query("demo") == "demo"
    assert frontend.session_id_from_query(["demo", "other"]) == "demo"
    assert frontend.session_id_from_query([]) == ""
    assert frontend.session_id_from_query(None) == ""
    assert frontend.session_id_from_query("   ") == ""


def _fake_history_payload(session_id: str, questions: list[str]) -> dict[str, Any]:
    """构造一份后端历史响应：每个问题一轮，只含 user + assistant（内部过程已被后端过滤）。"""
    messages: list[dict[str, Any]] = []
    for question in questions:
        messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"回答：{question}"},
            ]
        )
    return {
        "session_id": session_id,
        "backend": "sqlite",
        "total_messages": len(messages),
        "message_count": len(messages),
        # 每轮被过滤掉 3 条内部消息：系统提示词 + 调工具步骤 + 工具返回
        "filtered_messages": 3 * len(questions),
        "include_internal": False,
        "truncated": False,
        "messages": messages,
    }


class _FakeBackend:
    """离线假后端：记录调用，并按 URL 返回就绪 / 历史数据（不访问网络）。"""

    def __init__(
        self,
        history: dict[str, list[str]] | None = None,
        *,
        offline: bool = False,
        payloads: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.history = history or {}
        # 直接给出原始历史响应（用于构造畸形 / 没有 user 消息的历史，绕过 _fake_history_payload）
        self.payloads = payloads or {}
        self.offline = offline
        self.calls: list[str] = []

    def get(self, url: str, params: Any = None, timeout: int = 0) -> Any:
        self.calls.append(url)
        if self.offline:
            raise requests.exceptions.ConnectionError("refused")
        if "/chat/history/" in url:
            session_id = unquote(url.rsplit("/", 1)[-1])
            if session_id in self.payloads:
                return _FakeResponse(200, self.payloads[session_id])
            questions = self.history.get(session_id, [])
            if not questions:
                return _FakeResponse(
                    200,
                    {
                        "session_id": session_id,
                        "backend": "memory",
                        "total_messages": 0,
                        "message_count": 0,
                        "filtered_messages": 0,
                        "include_internal": False,
                        "truncated": False,
                        "messages": [],
                    },
                )
            return _FakeResponse(200, _fake_history_payload(session_id, questions))
        return _FakeResponse(
            200, {"status": "ready", "model": "fake-model", "model_configured": True}
        )

    @property
    def history_requests(self) -> list[str]:
        """按顺序返回请求过的历史会话 ID。"""
        return [unquote(url.rsplit("/", 1)[-1]) for url in self.calls if "/chat/history/" in url]


def _seven_message_history(session_id: str) -> dict[str, Any]:
    """真实后端曾经返回过的那种 7 条原始形状（含 system / tool / 草稿 / 重复回答）。

    后端现在默认只返回「提问 + 最终回答」，这份 payload 用来验证**前端即使拿到未过滤的
    数据也只画问答气泡**。
    """
    return {
        "session_id": session_id,
        "backend": "sqlite",
        "total_messages": 7,
        "message_count": 7,
        "filtered_messages": 0,
        "include_internal": True,
        "truncated": False,
        "messages": [
            {"role": "system", "content": "你是资深 AI 研发助手"},
            {"role": "user", "content": "请检索项目文档，告诉我 RAG 链路是怎么设计的？"},
            {
                "role": "assistant",
                "content": "我先检索知识库，避免凭空回答。",
                "tool_calls": ["search_knowledge_base", "search_knowledge_base"],
            },
            {
                "role": "tool",
                "content": "检索到 4 个相关片段",
                "name": "search_knowledge_base",
            },
            {
                "role": "tool",
                "content": "检索到 4 个相关片段（requirements）",
                "name": "search_knowledge_base",
            },
            {"role": "assistant", "content": "# 项目的 RAG 链路设计"},
            {"role": "assistant", "content": "# 项目的 RAG 链路设计"},
        ],
    }


def _filtered_history_payload(session_id: str) -> dict[str, Any]:
    """后端现在的默认返回：同一个会话只回「提问 + 最终回答」两条，另 5 条内部消息被过滤。"""
    return {
        "session_id": session_id,
        "backend": "sqlite",
        "total_messages": 2,
        "message_count": 2,
        "filtered_messages": 5,
        "include_internal": False,
        "truncated": False,
        "messages": [
            {"role": "user", "content": "请检索项目文档，告诉我 RAG 链路是怎么设计的？"},
            {"role": "assistant", "content": "# 项目的 RAG 链路设计"},
        ],
    }


def _run_ui(
    monkeypatch: pytest.MonkeyPatch,
    backend: _FakeBackend,
    query: dict[str, str] | None = None,
) -> AppTest:
    """离线跑一遍前端脚本（等价于浏览器打开页面），返回 AppTest 供断言。"""
    monkeypatch.setattr(frontend.requests, "get", backend.get)
    at = AppTest.from_file(str(APP_FILE), default_timeout=30)
    for key, value in (query or {}).items():
        at.query_params[key] = value
    at.run()
    assert not at.exception, f"前端脚本抛出异常：{at.exception}"
    return at


def _sidebar_button(at: AppTest, label: str) -> Any:
    return next(item for item in at.sidebar.button if item.label == label)


def _url_session(at: AppTest) -> str:
    """读取 AppTest 模拟地址栏里的会话 ID（多值参数用 list 表示，复用生产端归一化）。"""
    return frontend.session_id_from_query(at.query_params.get(frontend.SESSION_ID_QUERY_PARAM))


def test_ui_restores_session_and_history_from_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """刷新页面（地址栏带着 ?session_id=demo）时，自动从后端恢复历史气泡。"""
    backend = _FakeBackend({"demo": ["第一轮问题", "第二轮问题"]})

    at = _run_ui(monkeypatch, backend, query={frontend.SESSION_ID_QUERY_PARAM: "demo"})

    assert at.session_state["session_id"] == "demo"
    assert [(item["role"], item["content"]) for item in at.session_state["messages"]] == [
        ("user", "第一轮问题"),
        ("assistant", "回答：第一轮问题"),
        ("user", "第二轮问题"),
        ("assistant", "回答：第二轮问题"),
    ]
    assert backend.history_requests == ["demo"]
    # 地址栏里的会话 ID 保持可用（刷新、收藏、分享链接都能回到同一会话）
    assert _url_session(at) == "demo"


def test_ui_generates_session_id_and_writes_it_to_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """URL 里没有会话 ID 时新生成一个，并写回地址栏。"""
    backend = _FakeBackend()

    at = _run_ui(monkeypatch, backend)

    session_id = at.session_state["session_id"]
    assert len(session_id) == 12
    assert _url_session(at) == session_id
    assert backend.history_requests == [session_id]
    assert at.session_state["messages"] == []


def test_ui_replaces_invalid_session_id_from_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """地址栏被塞了非法会话 ID（超长）时不照单全收，改成新生成的 ID。"""
    backend = _FakeBackend()

    at = _run_ui(monkeypatch, backend, query={frontend.SESSION_ID_QUERY_PARAM: "x" * 65})

    assert at.session_state["session_id"] != "x" * 65
    assert len(_url_session(at)) == 12


def test_ui_rerun_keeps_bubbles_and_does_not_refetch_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """对话过程中的每次重跑都不重复拉历史、也不清空气泡（刷新后气泡仍在的等价验证）。"""
    backend = _FakeBackend({"demo": ["第一轮问题"]})
    at = _run_ui(monkeypatch, backend, query={frontend.SESSION_ID_QUERY_PARAM: "demo"})
    before = list(at.session_state["messages"])

    at.run()  # 等价于用户在页面里随便点一下触发的重跑

    assert not at.exception
    assert at.session_state["messages"] == before
    assert backend.history_requests == ["demo"], "同一会话只应拉一次历史"


def test_ui_start_new_session_writes_new_id_to_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """「开始新会话」换新 ID：地址栏同步、气泡清空、不残留上一个会话的历史。"""
    backend = _FakeBackend({"demo": ["第一轮问题"]})
    at = _run_ui(monkeypatch, backend, query={frontend.SESSION_ID_QUERY_PARAM: "demo"})
    assert at.session_state["messages"]

    _sidebar_button(at, "开始新会话").click()
    at.run()

    assert not at.exception
    new_id = at.session_state["session_id"]
    assert new_id != "demo"
    assert len(new_id) == 12
    assert _url_session(at) == new_id
    assert at.session_state["messages"] == []
    assert backend.history_requests == ["demo", new_id]


def test_ui_sidebar_has_no_manual_history_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    """侧边栏只剩「后端地址」输入框与「开始新会话」按钮，历史全自动加载。"""
    backend = _FakeBackend()

    at = _run_ui(monkeypatch, backend)

    assert [item.label for item in at.sidebar.text_input] == ["后端地址"]
    assert [item.label for item in at.sidebar.button] == ["开始新会话"]
    texts = " ".join(item.value for item in at.sidebar.caption)
    assert "历史会话 ID" not in texts
    assert "载入历史" not in texts
    # 页面只对当前会话发起一次历史请求，说明确实只剩这一条自动路径
    assert backend.history_requests == [at.session_state["session_id"]]


def test_ui_history_failure_is_reported_without_breaking_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """后端不可用时：只在侧边栏提示，气泡保持为空，脚本不抛异常。"""
    backend = _FakeBackend(offline=True)

    at = _run_ui(monkeypatch, backend, query={frontend.SESSION_ID_QUERY_PARAM: "demo"})

    assert at.session_state["session_id"] == "demo"
    assert at.session_state["messages"] == []
    assert any("历史载入失败" in item.value for item in at.sidebar.error)


def test_ui_draws_every_loaded_history_message_on_each_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """侧边栏报的条数与气泡数一致时，聊天区必须真的画出气泡，且每次重跑都逐条重画。"""
    backend = _FakeBackend(payloads={"demo": _filtered_history_payload("demo")})

    at = _run_ui(monkeypatch, backend, query={frontend.SESSION_ID_QUERY_PARAM: "demo"})

    messages = at.session_state["messages"]
    assert [(item["role"], item["content"]) for item in messages] == [
        ("user", "请检索项目文档，告诉我 RAG 链路是怎么设计的？"),
        ("assistant", "# 项目的 RAG 链路设计"),
    ]
    assert at.session_state["history_count"] == 2
    assert at.session_state["history_filtered"] == 5
    assert at.session_state["history_bubbles"] == 2
    assert len(at.chat_message) == len(messages) == 2, "每条 session_state.messages 都要有气泡"
    assert any("渲染 2 个气泡" in item.value for item in at.sidebar.success)
    assert any(
        "另有 5 条系统提示 / 工具结果 / 中间步骤已过滤" in item.value for item in at.sidebar.success
    ), "侧边栏要说清楚被过滤掉的内部消息有多少条"
    assert any(
        "后端返回 2 条对话消息（另有 5 条内部消息已过滤） → 渲染 2 个气泡" in item.value
        for item in at.sidebar.caption
    )

    rendered = [str(item.value) for item in at.markdown]
    assert any("请检索项目文档" in text for text in rendered), "用户气泡正文要画出来"
    assert any("# 项目的 RAG 链路设计" in text for text in rendered), "助手气泡正文要画出来"

    at.run()  # 等价于按 F5 / 点击按钮触发的重跑

    assert not at.exception
    assert len(at.chat_message) == len(at.session_state["messages"]) == 2
    assert any("# 项目的 RAG 链路设计" in str(item.value) for item in at.markdown)
    assert backend.history_requests == ["demo"], "重跑只重画气泡，不重复拉历史"


def test_ui_renders_only_question_and_answer_bubbles(monkeypatch: pytest.MonkeyPatch) -> None:
    """即使后端返回未过滤的原始历史（7 条），页面也只画「提问 + 最终回答」两个气泡。"""
    backend = _FakeBackend(payloads={"demo": _seven_message_history("demo")})

    at = _run_ui(monkeypatch, backend, query={frontend.SESSION_ID_QUERY_PARAM: "demo"})

    messages = at.session_state["messages"]
    assert [(item["role"], item["content"]) for item in messages] == [
        ("user", "请检索项目文档，告诉我 RAG 链路是怎么设计的？"),
        ("assistant", "# 项目的 RAG 链路设计"),
    ]
    assert len(at.chat_message) == 2
    assert all(item["meta"] == {} for item in messages), "历史气泡不应带执行过程折叠区"

    rendered = "\n".join(str(item.value) for item in at.markdown)
    for internal in ["你是资深 AI 研发助手", "search_knowledge_base", "检索到 4 个相关片段"]:
        assert internal not in rendered, "系统提示词 / 工具调用 / 检索片段都不能渲染成气泡"


def test_ui_reloads_history_when_bubbles_are_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """自愈：气泡被清空但历史还在时，重跑会重新拉取，避免出现“有历史却空白”。"""
    backend = _FakeBackend({"demo": ["第一轮问题"]})
    at = _run_ui(monkeypatch, backend, query={frontend.SESSION_ID_QUERY_PARAM: "demo"})
    assert len(at.session_state["messages"]) == 2

    at.session_state["messages"] = []  # 模拟中间态：气泡没了、history_session 还在
    at.run()

    assert not at.exception
    assert len(at.session_state["messages"]) == 2
    assert backend.history_requests == ["demo", "demo"], "气泡丢失时应自愈重拉一次"


def test_ui_does_not_render_internal_process_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """历史里只有系统提示词与工具过程时，一个气泡都不画，并在侧边栏说明原因。"""
    payload = {
        "session_id": "demo",
        "backend": "sqlite",
        "total_messages": 3,
        "message_count": 3,
        "filtered_messages": 0,
        "include_internal": True,
        "truncated": False,
        "messages": [
            {"role": "system", "content": "你是助手"},
            {"role": "assistant", "content": "", "tool_calls": ["search_knowledge_base"]},
            {"role": "tool", "content": "片段：RAG 链路", "name": "search_knowledge_base"},
        ],
    }
    backend = _FakeBackend(payloads={"demo": payload})

    at = _run_ui(monkeypatch, backend, query={frontend.SESSION_ID_QUERY_PARAM: "demo"})

    assert at.session_state["history_bubbles"] == 0
    assert at.session_state["history_mismatch"] is True
    assert len(at.chat_message) == 0, "内部过程不渲染成气泡"
    assert any("没有可渲染的对话内容" in item.value for item in at.sidebar.warning)

    rendered = "\n".join(str(item.value) for item in at.markdown)
    assert "search_knowledge_base" not in rendered
    assert "片段：RAG 链路" not in rendered


def test_ui_html_only_answer_is_shown_as_code_instead_of_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型输出只有 HTML 标签时（Markdown 会整段过滤掉），改用代码块显示，避免空气泡。"""
    html_only = '<div class="card"><span></span></div><!-- 空卡片 -->'
    payload = {
        "session_id": "demo",
        "backend": "sqlite",
        "total_messages": 2,
        "message_count": 2,
        "truncated": False,
        "messages": [
            {"role": "user", "content": "给我一段 HTML"},
            {"role": "assistant", "content": html_only},
        ],
    }
    backend = _FakeBackend(payloads={"demo": payload})

    at = _run_ui(monkeypatch, backend, query={frontend.SESSION_ID_QUERY_PARAM: "demo"})

    assert len(at.chat_message) == 2
    assert [item.value for item in at.code] == [html_only]
    assert any(frontend.HTML_FALLBACK_HINT in str(item.value) for item in at.caption)


def test_ui_warns_when_backend_reports_messages_but_nothing_renders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """后端报有条数却一条消息都没返回时，侧边栏给 warning，而不是静默白屏。"""
    payload = {
        "session_id": "demo",
        "backend": "sqlite",
        "total_messages": 2,
        "message_count": 2,
        "truncated": False,
        "messages": [],
    }
    backend = _FakeBackend(payloads={"demo": payload})

    at = _run_ui(monkeypatch, backend, query={frontend.SESSION_ID_QUERY_PARAM: "demo"})

    assert at.session_state["history_bubbles"] == 0
    assert at.session_state["history_mismatch"] is True
    assert len(at.chat_message) == 0
    assert any("没有可渲染的对话内容" in item.value for item in at.sidebar.warning)
