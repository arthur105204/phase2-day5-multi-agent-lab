"""Application configuration.

Keep config small and explicit. Do not read environment variables directly in agents.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")

    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="multi-agent-research-lab",
        validation_alias="LANGSMITH_PROJECT",
    )

    tavily_api_key: str | None = Field(default=None, validation_alias="TAVILY_API_KEY")

    max_iterations: int = Field(default=6, ge=1, le=20, validation_alias="MAX_ITERATIONS")
    timeout_seconds: int = Field(default=60, ge=5, le=600, validation_alias="TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=3, ge=1, le=10, validation_alias="LLM_MAX_RETRIES")
    min_final_answer_chars: int = Field(
        default=80,
        ge=20,
        validation_alias="MIN_FINAL_ANSWER_CHARS",
    )
    benchmark_usd_per_1k_input_tokens: float | None = Field(
        default=None,
        ge=0,
        validation_alias="BENCHMARK_USD_PER_1K_INPUT_TOKENS",
    )
    benchmark_usd_per_1k_output_tokens: float | None = Field(
        default=None,
        ge=0,
        validation_alias="BENCHMARK_USD_PER_1K_OUTPUT_TOKENS",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
