"""接口层测试：健康检查 / 就绪探针 / 缺配置时的错误语义。"""

from __future__ import annotations

from typing import Any


def test_root_returns_service_info(client: Any) -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["api"] == "/api/v1"
    assert body["docs"] == "/docs"


def test_health_ok(client: Any) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "AI 研发助手 Agent"


def test_ready_is_degraded_without_api_key(client: Any) -> None:
    response = client.get("/api/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_configured"] is False
    assert body["hint"]


def test_ready_is_ready_with_api_key(client: Any, monkeypatch: Any) -> None:
    from app.core.config import reset_settings_cache

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    reset_settings_cache()

    body = client.get("/api/v1/ready").json()

    assert body["status"] == "ready"
    assert body["model_configured"] is True


def test_chat_without_api_key_returns_503(client: Any) -> None:
    response = client.post(
        "/api/v1/chat",
        json={"message": "你好", "session_id": "test-503"},
    )

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_chat_rejects_empty_message(client: Any) -> None:
    response = client.post("/api/v1/chat", json={"message": ""})

    assert response.status_code == 422
