from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark


def test_run_benchmark_adds_quality_and_notes() -> None:
    def runner(_q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query="Explain X with citations"))
        state.sources = [
            SourceDocument(title="A", url="https://a", snippet="aaa"),
            SourceDocument(title="B", url="https://b", snippet="bbb"),
        ]
        state.final_answer = "Answer body. Sources: [1] https://a"
        return state

    _state, metrics = run_benchmark("unit", "q", runner)
    assert metrics.latency_seconds >= 0
    assert metrics.quality_score is not None
    assert metrics.quality_score >= 0
    assert metrics.notes

