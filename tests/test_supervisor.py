from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems briefly"))


def test_supervisor_routes_to_researcher_when_research_missing() -> None:
    state = _state()
    SupervisorAgent().run(state)
    assert state.next_route == "researcher"


def test_supervisor_routes_to_analyst_after_research_notes() -> None:
    state = _state()
    state.research_notes = "notes"
    SupervisorAgent().run(state)
    assert state.next_route == "analyst"


def test_supervisor_routes_to_writer_after_analysis_notes() -> None:
    state = _state()
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    SupervisorAgent().run(state)
    assert state.next_route == "writer"


def test_supervisor_routes_done_when_final_answer_exists() -> None:
    state = _state()
    state.final_answer = "done"
    SupervisorAgent().run(state)
    assert state.next_route == "done"
