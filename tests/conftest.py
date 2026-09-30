"""pytest 公共固件。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections.abc import Iterator  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from app.core.config import reset_settings_cache  # noqa: E402
from app.llm.factory import reset_chat_model_cache  # noqa: E402
from app.memory.checkpointer import reset_checkpointer_cache  # noqa: E402


def _reset_all() -> None:
    reset_settings_cache()
    reset_chat_model_cache()
    reset_checkpointer_cache()


@pytest.fixture(autouse=True)
def clean_caches() -> Iterator[None]:
    """每个用例前后清空单例缓存，避免用例之间互相污染。"""
    _reset_all()
    yield
    _reset_all()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 WORKSPACE_DIR 指向临时目录，避免污染真实 data/workspace。"""
    target = tmp_path / "workspace"
    target.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("WORKSPACE_DIR", str(target))
    reset_settings_cache()
    return target


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """FastAPI 测试客户端（默认场景：未配置 API Key）。"""
    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_settings_cache()

    with TestClient(create_app()) as test_client:
        yield test_client
