"""Multi-agent workflow (imperative orchestration with guardrails)."""

from time import monotonic

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Runs Supervisor -> workers until done or a guardrail trips."""

    def build(self) -> object:
        """Describe the workflow graph.

        The reference lab implementation uses an explicit loop (no LangGraph compile step).
        """

        return {
            "mode": "imperative",
            "nodes": ["supervisor", "researcher", "analyst", "writer"],
            "edges": "supervisor routes to one worker or done",
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute up to `MAX_ITERATIONS` supervisor cycles within `TIMEOUT_SECONDS`."""

        settings = get_settings()
        deadline = monotonic() + float(settings.timeout_seconds)
        supervisor = SupervisorAgent()
        workers = {
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
        }

        for step in range(settings.max_iterations):
            if monotonic() > deadline:
                state.errors.append("Guardrail: workflow exceeded TIMEOUT_SECONDS")
                state.add_trace_event("workflow", {"event": "timeout", "step": step})
                break

            state = supervisor.run(state)
            route = state.next_route
            if not route:
                state.errors.append("Guardrail: supervisor did not set next_route")
                break

            state.record_route(route)
            state.add_trace_event("workflow", {"step": step, "route": route})

            if route == "done":
                break

            worker = workers.get(route)
            if worker is None:
                state.errors.append(f"Guardrail: unknown route {route!r}")
                break

            try:
                state = worker.run(state)
            except Exception as exc:  # noqa: BLE001 — lab boundary: record and stop
                state.errors.append(f"Agent failure ({route}): {exc}")
                state.add_trace_event("workflow", {"event": "agent_error", "route": route})
                break

            if (
                route == "writer"
                and state.final_answer
                and len(state.final_answer) < settings.min_final_answer_chars
            ):
                state.add_trace_event(
                    "validation",
                    {
                        "event": "short_final_answer_retry",
                        "length": len(state.final_answer),
                        "min": settings.min_final_answer_chars,
                    },
                )
                try:
                    state = WriterAgent().run(state)
                except Exception as exc:  # noqa: BLE001
                    state.errors.append(f"Writer retry failed: {exc}")
                    break
        else:
            state.errors.append(
                "Guardrail: reached MAX_ITERATIONS without a clean stop "
                "(check logs / partial outputs)."
            )
            state.add_trace_event("workflow", {"event": "max_iterations_exhausted"})

        return state
