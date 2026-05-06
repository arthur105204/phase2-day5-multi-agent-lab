"""Tracing hooks.

LangSmith: when `LANGSMITH_API_KEY` is set in settings / `.env`, call
`apply_langsmith_runtime_env` early (e.g. from CLI `_init`) and use `wrap_openai` in
`LLMClient` so chat completions appear in the LangSmith project.

Langfuse / OpenTelemetry remain optional extensions.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multi_agent_research_lab.core.config import Settings


def apply_langsmith_runtime_env(settings: "Settings") -> None:
    """Copy LangSmith settings into process env for the LangSmith / OpenAI wrappers."""

    if not settings.langsmith_api_key:
        return
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    TODO(student): Replace or augment with Langfuse or OpenTelemetry provider spans.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
