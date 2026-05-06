"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self._llm = llm or LLMClient()
        self._search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        max_src = state.request.max_sources
        state.sources = self._search.search(state.request.query, max_results=max_src)

        lines = []
        for i, s in enumerate(state.sources, start=1):
            url = s.url or "(no url)"
            lines.append(f"[{i}] {s.title} — {url}\n    {s.snippet}")

        bundle = "\n".join(lines)
        system = (
            "You are a careful research assistant. Summarize the provided sources into structured "
            "bullet notes. Attribute claims to source numbers like [1]. If a source is weak or "
            "mock/offline, say so briefly."
        )
        user = (
            f"User question:\n{state.request.query}\n\nSources:\n{bundle}\n\nWrite research notes."
        )
        resp = self._llm.complete(system, user)
        state.research_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
            )
        )
        state.add_trace_event(
            "researcher",
            {
                "sources": len(state.sources),
                "tokens_in": resp.input_tokens,
                "tokens_out": resp.output_tokens,
            },
        )
        return state
