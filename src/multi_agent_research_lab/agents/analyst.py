"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, *, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        notes = state.research_notes or ""
        system = (
            "You are an analyst. Extract key claims, tensions between sources, gaps in evidence, "
            "and risks. Use short bullets. Audience: "
            f"{state.request.audience}."
        )
        user = f"Original question:\n{state.request.query}\n\nResearch notes:\n{notes}"
        resp = self._llm.complete(system, user)
        state.analysis_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes,
                metadata={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
            )
        )
        state.add_trace_event(
            "analyst",
            {"tokens_in": resp.input_tokens, "tokens_out": resp.output_tokens},
        )
        return state
