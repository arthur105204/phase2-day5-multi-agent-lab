"""Search client for ResearcherAgent."""

import logging
from typing import Any

import httpx

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Tavily when configured; otherwise deterministic mock sources for offline lab runs."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        settings = get_settings()
        if settings.tavily_api_key:
            return _tavily_search(query, max_results=max_results, api_key=settings.tavily_api_key)
        return _mock_sources(query, max_results)


def _mock_sources(query: str, max_results: int) -> list[SourceDocument]:
    q = query.strip()[:80]
    return [
        SourceDocument(
            title=f"Mock reading note {i + 1} ({q})",
            url=f"https://example.invalid/lab-mock/{i + 1}",
            snippet=(
                f"Offline placeholder snippet {i + 1}. "
                f"Set TAVILY_API_KEY for real search. Topic: {q}."
            ),
            metadata={"mock": True},
        )
        for i in range(max_results)
    ]


def _tavily_search(query: str, *, max_results: int, api_key: str) -> list[SourceDocument]:
    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("Tavily search failed, falling back to mock sources: %s", exc)
        return _mock_sources(query, max_results)

    results = data.get("results") or []
    out: list[SourceDocument] = []
    for item in results[:max_results]:
        out.append(
            SourceDocument(
                title=str(item.get("title") or "Untitled"),
                url=item.get("url"),
                snippet=str(item.get("content") or item.get("snippet") or ""),
                metadata={"raw_score": item.get("score")},
            )
        )
    return out or _mock_sources(query, max_results)
