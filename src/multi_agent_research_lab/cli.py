"""Command-line entrypoint for the lab starter."""

from time import perf_counter
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import apply_langsmith_runtime_env
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    apply_langsmith_runtime_env(settings)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline that calls the configured LLM once."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    started = perf_counter()
    try:
        llm = LLMClient()
        resp = llm.complete(
            "You are a careful research assistant. Answer clearly for the stated audience.",
            f"Audience: {request.audience}\n\nQuestion:\n{request.query}",
        )
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Configuration needed", style="red"))
        raise typer.Exit(code=1) from exc

    elapsed = perf_counter() - started
    state.final_answer = resp.content
    state.add_trace_event(
        "baseline",
        {
            "latency_seconds": elapsed,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
        },
    )
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))
    tok_in = resp.input_tokens
    tok_out = resp.output_tokens
    console.print(f"[dim]Latency: {elapsed:.2f}s | tokens in/out: {tok_in}/{tok_out}[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow (Supervisor + Researcher + Analyst + Writer)."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Configuration needed", style="yellow"))
        raise typer.Exit(code=2) from exc

    if result.errors:
        err_text = "\n".join(result.errors)
        console.print(
            Panel.fit(err_text, title="Run completed with issues", style="yellow"),
        )
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
