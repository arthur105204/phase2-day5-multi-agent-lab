"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Set `next_route` to researcher | analyst | writer | done."""

        if state.final_answer:
            state.next_route = "done"
            state.add_trace_event("supervisor", {"next": "done", "reason": "final_answer_present"})
            return state

        if not state.research_notes:
            state.next_route = "researcher"
        elif not state.analysis_notes:
            state.next_route = "analyst"
        else:
            state.next_route = "writer"

        state.add_trace_event("supervisor", {"next": state.next_route})
        return state
