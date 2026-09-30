"""前端（frontend/app.py）离线测试：请求构造、响应解析与错误语义。

全部离线：用 monkeypatch 替换 ``requests.post`` / ``requests.get``，不访问网络。
"""

from __future__ import annotations

from typing import Any

import pytest
import requests

from frontend import app as frontend


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
