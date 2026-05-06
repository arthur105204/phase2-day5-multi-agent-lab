"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.observability.tracing import apply_langsmith_runtime_env


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """OpenAI-compatible chat client (cloud OpenAI or local servers such as Ollama)."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retries and HTTP timeout."""

        settings = get_settings()
        if not settings.openai_api_key and not settings.openai_base_url:
            raise StudentTodoError(
                "Configure OPENAI_API_KEY (cloud) or OPENAI_BASE_URL "
                "(local OpenAI-compatible server)."
            )

        return _complete_with_retry(
            api_key=settings.openai_api_key or "local-not-a-secret",
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            timeout_seconds=float(settings.timeout_seconds),
            max_attempts=settings.llm_max_retries,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )


def _complete_with_retry(
    *,
    api_key: str,
    base_url: str | None,
    model: str,
    timeout_seconds: float,
    max_attempts: int,
    system_prompt: str,
    user_prompt: str,
) -> LLMResponse:
    @retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    )
    def _call() -> LLMResponse:
        settings = get_settings()
        apply_langsmith_runtime_env(settings)
        client: OpenAI = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        if settings.langsmith_api_key:
            from langsmith.wrappers import wrap_openai

            client = wrap_openai(client)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        return LLMResponse(content=content, input_tokens=input_tokens, output_tokens=output_tokens)

    return _call()
