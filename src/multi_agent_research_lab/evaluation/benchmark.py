"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")


def _sum_tokens(state: ResearchState) -> tuple[int | None, int | None]:
    total_in = 0
    total_out = 0
    seen_any = False

    for r in state.agent_results:
        meta = r.metadata or {}
        ti = meta.get("input_tokens")
        to = meta.get("output_tokens")
        if isinstance(ti, int):
            total_in += ti
            seen_any = True
        if isinstance(to, int):
            total_out += to
            seen_any = True

    for ev in state.trace:
        payload = ev.get("payload") if isinstance(ev, dict) else None
        if not isinstance(payload, dict):
            continue
        ti = payload.get("input_tokens")
        to = payload.get("output_tokens")
        if isinstance(ti, int):
            total_in += ti
            seen_any = True
        if isinstance(to, int):
            total_out += to
            seen_any = True

    if not seen_any:
        return None, None
    return total_in, total_out


def _citation_coverage(state: ResearchState) -> float | None:
    """Return fraction of sources cited at least once in final_answer."""

    if not state.sources:
        return None
    text = state.final_answer or ""
    cited: set[int] = set()
    for m in _CITATION_RE.finditer(text):
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        if 1 <= idx <= len(state.sources):
            cited.add(idx)
    return len(cited) / len(state.sources)


def _quality_heuristic(state: ResearchState, citation_cov: float | None) -> float:
    """Cheap heuristic score (0-10) without calling another LLM."""

    score = 0.0
    answer = state.final_answer or ""

    if answer.strip():
        score += 4.0
    if len(answer) >= 300:
        score += 2.0
    if "sources" in answer.lower():
        score += 1.0

    if citation_cov is not None:
        if citation_cov >= 0.6:
            score += 2.0
        elif citation_cov > 0:
            score += 1.0

    if state.errors:
        score -= 2.0

    if score < 0:
        return 0.0
    if score > 10:
        return 10.0
    return score


def _estimate_cost_usd(tokens_in: int | None, tokens_out: int | None) -> float | None:
    settings = get_settings()
    in_price = settings.benchmark_usd_per_1k_input_tokens
    out_price = settings.benchmark_usd_per_1k_output_tokens
    if in_price is None and out_price is None:
        return None
    if tokens_in is None and tokens_out is None:
        return None
    cost = 0.0
    if in_price is not None and tokens_in is not None:
        cost += (tokens_in / 1000.0) * in_price
    if out_price is not None and tokens_out is not None:
        cost += (tokens_out / 1000.0) * out_price
    return cost


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and produce basic quality/cost/citation metrics."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    tok_in, tok_out = _sum_tokens(state)
    cit_cov = _citation_coverage(state)
    est_cost = _estimate_cost_usd(tok_in, tok_out)
    quality = _quality_heuristic(state, cit_cov)

    notes_parts: list[str] = []
    if tok_in is not None or tok_out is not None:
        notes_parts.append(f"tokens_in={tok_in} tokens_out={tok_out}")
    if cit_cov is not None:
        notes_parts.append(f"citation_coverage={cit_cov:.2f}")
    if state.errors:
        notes_parts.append(f"errors={len(state.errors)}")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=est_cost,
        quality_score=quality,
        notes=" | ".join(notes_parts),
    )
    return state, metrics
