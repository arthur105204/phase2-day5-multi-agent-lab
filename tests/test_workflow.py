import pytest

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


def test_workflow_completes_with_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    get_settings.cache_clear()

    long_stub = (
        "Stub LLM answer with enough length to satisfy validation. "
        "Second sentence for structure and traceability in tests."
    )

    def fake_complete(_self: object, system_prompt: str, user_prompt: str) -> LLMResponse:
        _ = system_prompt
        _ = user_prompt
        return LLMResponse(content=long_stub, input_tokens=3, output_tokens=4)

    def fake_search(_self: object, query: str, max_results: int = 5) -> list[SourceDocument]:
        _ = query
        _ = max_results
        return [
            SourceDocument(
                title="Example",
                url="https://example.com/lab",
                snippet="Offline-friendly snippet for workflow tests.",
            )
        ]

    monkeypatch.setattr(LLMClient, "complete", fake_complete)
    monkeypatch.setattr(SearchClient, "search", fake_search)

    state = ResearchState(request=ResearchQuery(query="What is GraphRAG in one paragraph?"))
    out = MultiAgentWorkflow().run(state)

    assert out.final_answer
    assert out.research_notes
    assert out.analysis_notes
    assert not out.errors
    assert "researcher" in out.route_history
    assert "writer" in out.route_history
