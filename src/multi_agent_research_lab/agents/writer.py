"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, *, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        settings = get_settings()
        research = state.research_notes or ""
        analysis = state.analysis_notes or ""
        source_lines = []
        for i, s in enumerate(state.sources, start=1):
            url = s.url or ""
            source_lines.append(f"[{i}] {s.title} {url}".strip())

        system = (
            "You are the lead author. Write a clear final answer for the user. "
            "Ground statements in the notes; include a 'Sources' section listing [n] lines. "
            f"If output is shorter than {settings.min_final_answer_chars} characters, expand with "
            "structure (overview, key points, limitations) while staying honest about uncertainty."
        )
        user = (
            f"Question:\n{state.request.query}\n\n"
            f"Research notes:\n{research}\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Available citations:\n" + "\n".join(source_lines)
        )
        resp = self._llm.complete(system, user)
        state.final_answer = resp.content.strip()
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
            )
        )
        state.add_trace_event(
            "writer",
            {"tokens_in": resp.input_tokens, "tokens_out": resp.output_tokens},
        )
        return state
