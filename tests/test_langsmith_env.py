import os

import pytest

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.observability.tracing import apply_langsmith_runtime_env


def test_apply_langsmith_runtime_env_sets_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "pytest-langsmith-project")
    get_settings.cache_clear()
    settings = get_settings()
    apply_langsmith_runtime_env(settings)
    assert os.environ["LANGSMITH_API_KEY"] == "ls-test-key"
    assert os.environ["LANGSMITH_PROJECT"] == "pytest-langsmith-project"
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"


def test_apply_langsmith_skips_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    from multi_agent_research_lab.core.config import Settings

    s = Settings.model_construct(langsmith_api_key=None)
    apply_langsmith_runtime_env(s)
    assert "LANGCHAIN_TRACING_V2" not in os.environ
