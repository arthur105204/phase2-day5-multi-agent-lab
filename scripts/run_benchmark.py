"""Run a small benchmark suite and write `reports/benchmark_report.md`.

This script is intentionally simple and works with either:
- Cloud OpenAI (OPENAI_API_KEY)
- Local OpenAI-compatible server (OPENAI_BASE_URL), e.g. Ollama
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient


@dataclass(frozen=True)
class RunSummary:
    name: str
    query: str
    latency_seconds: float
    tokens_in: int | None
    tokens_out: int | None
    errors: list[str]


def _sum_tokens(state: ResearchState) -> tuple[int | None, int | None]:
    ins: int = 0
    outs: int = 0
    seen_any = False
    for r in state.agent_results:
        meta = r.metadata or {}
        ti = meta.get("input_tokens")
        to = meta.get("output_tokens")
        if isinstance(ti, int):
            ins += ti
            seen_any = True
        if isinstance(to, int):
            outs += to
            seen_any = True
    # Baseline runner stores tokens in `trace` to avoid coupling to AgentResult/AgentName.
    for ev in state.trace:
        payload = (ev or {}).get("payload") if isinstance(ev, dict) else None
        if not isinstance(payload, dict):
            continue
        ti = payload.get("input_tokens")
        to = payload.get("output_tokens")
        if isinstance(ti, int):
            ins += ti
            seen_any = True
        if isinstance(to, int):
            outs += to
            seen_any = True
    if not seen_any:
        return None, None
    return ins, outs


def _baseline_runner(query: str) -> ResearchState:
    req = ResearchQuery(query=query)
    state = ResearchState(request=req)
    llm = LLMClient()
    resp = llm.complete(
        "You are a careful research assistant. Answer clearly for the stated audience.",
        f"Audience: {req.audience}\n\nQuestion:\n{req.query}",
    )
    state.final_answer = resp.content
    state.add_trace_event(
        "baseline",
        {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
    )
    return state


def _multi_agent_runner(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


def _run(name: str, query: str, runner) -> RunSummary:
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    ti, to = _sum_tokens(state)
    return RunSummary(
        name=name,
        query=query,
        latency_seconds=latency,
        tokens_in=ti,
        tokens_out=to,
        errors=state.errors,
    )


def _render_md(items: list[RunSummary]) -> str:
    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("- **Date**: (fill in)")
    lines.append("- **Model**: from `.env` (`OPENAI_MODEL`)")
    lines.append("- **Search**: Tavily if `TAVILY_API_KEY` set, otherwise mock sources")
    lines.append("")
    lines.append("## Results (raw)")
    lines.append("")
    for it in items:
        lines.append(f"### {it.name}")
        lines.append(f"- **query**: {it.query}")
        lines.append(f"- **latency_seconds**: {it.latency_seconds:.2f}")
        lines.append(f"- **tokens_in**: {it.tokens_in}")
        lines.append(f"- **tokens_out**: {it.tokens_out}")
        lines.append(f"- **errors**: {it.errors if it.errors else '[]'}")
        lines.append("")
    lines.append("## Notes")
    lines.append("- Add a short quality comparison (0-10) and rationale.")
    lines.append(
        "- Paste LangSmith run links or include screenshots under "
        "`reports/langsmith_screenshots/`."
    )
    lines.append("")
    lines.append("## Failure modes & fixes")
    lines.append(
        "- **Mock sources -> low-quality research**: set `TAVILY_API_KEY` to fetch real sources."
    )
    lines.append(
        "- **High latency/tokens**: lower `max_sources`, truncate snippets, "
        "or reduce output length requirements."
    )
    lines.append(
        "- **PowerShell line continuation**: use one-line commands or PowerShell backtick."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    queries = [
        "Explain multi-agent systems in 5 bullet points.",
        "Research GraphRAG state-of-the-art and write a 300-word summary.",
    ]

    items: list[RunSummary] = []
    metrics: list[BenchmarkMetrics] = []
    for q in queries:
        b_state, b_metrics = run_benchmark("baseline", q, _baseline_runner)
        items.append(_run("baseline", q, (lambda s=b_state: (lambda _q: s))()))
        metrics.append(b_metrics)

        m_state, m_metrics = run_benchmark("multi-agent", q, _multi_agent_runner)
        items.append(_run("multi-agent", q, (lambda s=m_state: (lambda _q: s))()))
        metrics.append(m_metrics)

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_report.md").write_text(_render_md(items), encoding="utf-8")

    (out_dir / "benchmark_report_table.md").write_text(
        render_markdown_report(metrics),
        encoding="utf-8",
    )

    print("Wrote reports/benchmark_report.md and reports/benchmark_report_table.md")


if __name__ == "__main__":
    main()

