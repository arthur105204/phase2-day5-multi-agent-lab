# Lab 20: Multi-Agent Research System Starter

Starter repo cho bài lab **Multi-Agent Systems**: xây dựng hệ thống nghiên cứu gồm **Supervisor + Researcher + Analyst + Writer** và benchmark với single-agent baseline.

> Repo này gồm **implementation tham chiếu** cho baseline (single LLM call), multi-agent loop (Supervisor + 3 worker), `LLMClient` (OpenAI-compatible), và `SearchClient` (Tavily hoặc mock). Phần mở rộng (Critic, LangGraph compile, benchmark nâng cao) vẫn để `TODO(student)` ở vài file.

## What you can do with this repo

- **Run a single-agent baseline**: 1 LLM call, prints answer + latency/tokens.
- **Run a multi-agent workflow**: Supervisor routes through Researcher → Analyst → Writer, returns a JSON `ResearchState`.
- **Trace LLM calls in LangSmith**: set `LANGSMITH_API_KEY` and runs will appear under your `LANGSMITH_PROJECT`.
- **Benchmark** baseline vs multi-agent: use `scripts/run_benchmark.py` + `reports/benchmark_report.md`.

## Learning outcomes

Sau 2 giờ lab, học viên cần có thể:

1. Thiết kế role rõ ràng cho nhiều agent.
2. Xây dựng shared state đủ thông tin cho handoff.
3. Thêm guardrail tối thiểu: max iterations, timeout, retry/fallback, validation.
4. Trace được luồng chạy và giải thích agent nào làm gì.
5. Benchmark single-agent vs multi-agent theo quality, latency, cost.

## Architecture mục tiêu

```text
User Query
   |
   v
Supervisor / Router
   |------> Researcher Agent  -> research_notes
   |------> Analyst Agent     -> analysis_notes
   |------> Writer Agent      -> final_answer
   |
   v
Trace + Benchmark Report
```

## Cấu trúc repo

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/              # Supervisor + workers (+ Critic TODO)
│   ├── core/                # Config, state, schemas, errors
│   ├── graph/               # Multi-agent workflow (loop có guardrails)
│   ├── services/            # LLM, search, storage clients
│   ├── evaluation/          # Benchmark / report (mở rộng)
│   ├── observability/       # Logging / trace_span hook
│   └── cli.py               # CLI entrypoint
├── configs/                 # YAML configs for lab variants
├── docs/                    # Lab guide, rubric, design notes
├── tests/                   # Unit tests
├── notebooks/               # Optional notebook entrypoint
├── scripts/                 # Helper scripts
├── reports/                 # Deliverables (benchmark report + trace evidence)
├── .env.example             # Environment variables template
├── pyproject.toml           # Python project config
├── Dockerfile               # Containerized dev/runtime
└── Makefile                 # Common commands
```

## Key concepts (how it works)

- **State passing**: Every agent reads/writes a shared `ResearchState` (`core/state.py`).
- **Supervisor routing**: `SupervisorAgent` sets `state.next_route` to one of: `researcher`, `analyst`, `writer`, `done`.
- **Guardrails**: `MultiAgentWorkflow` enforces `MAX_ITERATIONS` and `TIMEOUT_SECONDS`, records failures to `state.errors`.
- **Search**: `SearchClient` uses Tavily if `TAVILY_API_KEY` is set, otherwise a deterministic mock (offline).
- **Tracing**: `LLMClient` wraps the OpenAI client via LangSmith `wrap_openai` when `LANGSMITH_API_KEY` is present.

## Quickstart

### 1. Tạo môi trường

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

`pip` cần extras trong ngoặc vuông và **bọc bằng dấu ngoặc kép** trên Windows: `".[dev]"` (không dùng `pip install -e dev`).

### 2. Cấu hình model và search

Mở `.env`:

- **Cloud OpenAI**: `OPENAI_API_KEY`, `OPENAI_MODEL`.
- **Local (Ollama, OpenAI-compatible)**: `OPENAI_BASE_URL` (ví dụ `http://localhost:11434/v1`), `OPENAI_MODEL` (tên model Ollama). `OPENAI_API_KEY` có thể để trống.
- **Search thật (tuỳ chọn)**: `TAVILY_API_KEY`. Nếu không có, `SearchClient` dùng **mock sources** để lab vẫn chạy offline.

Ví dụ `.env` cho Ollama:

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.1:8b
OPENAI_API_KEY=
```

### 3. Guardrails (env)

Các biến sau được đọc trong `core/config.py` và áp dụng trong `LLMClient` / `graph/workflow.py`:

| Biến | Ý nghĩa |
|------|---------|
| `MAX_ITERATIONS` | Số vòng tối đa (mỗi vòng: Supervisor + tối đa một worker). Hết vòng mà chưa dừng sạch sẽ ghi lỗi guardrail. |
| `TIMEOUT_SECONDS` | Giới hạn thời gian wall-clock cho cả workflow; hết giờ sẽ dừng và ghi `errors`. |
| `LLM_MAX_RETRIES` | Số lần retry (tenacity) khi LLM gặp rate limit / timeout / lỗi kết nối. |
| `MIN_FINAL_ANSWER_CHARS` | Sau bước Writer, nếu `final_answer` quá ngắn sẽ **retry Writer một lần** (validation tối thiểu). |

**LangSmith:** đặt `LANGSMITH_API_KEY` và `LANGSMITH_PROJECT` trong `.env`. Ứng dụng gọi `apply_langsmith_runtime_env` khi khởi động CLI và bọc `OpenAI` bằng `langsmith.wrappers.wrap_openai`, nên mỗi `chat.completions` xuất hiện trong project (sau vài giây). Kiểm tra tại `https://smith.langchain.com`.

**Benchmark cost (optional):** nếu bạn muốn ước lượng USD trong benchmark table, set:

```env
BENCHMARK_USD_PER_1K_INPUT_TOKENS=0.00015
BENCHMARK_USD_PER_1K_OUTPUT_TOKENS=0.00060
```

### 4. Smoke test

```bash
pytest
python -m multi_agent_research_lab.cli --help
```

Trên Windows nếu chưa cài `make`, chạy trực tiếp `pytest` (tương đương target `test` trong `Makefile`).

### 5. Chạy baseline (single-agent)

**PowerShell (một dòng)** — không dùng `\` xuống dòng như Bash:

```powershell
python -m multi_agent_research_lab.cli baseline --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Baseline gọi `LLMClient` một lần và in latency / token (nếu provider trả về).

### 6. Chạy multi-agent

```powershell
python -m multi_agent_research_lab.cli multi-agent --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Output là JSON `ResearchState` (trace, route_history, notes, errors nếu có).

## Benchmark & deliverables

### 1) Run benchmark script

```powershell
python scripts\run_benchmark.py
```

Outputs:

- `reports/benchmark_report_table.md`: table of latency/cost/quality/notes
- `reports/benchmark_report.md`: template to paste links + summarize results

### 2) Evidence for tracing

- Put screenshots under `reports/langsmith_screenshots/` (already supported), or
- Paste LangSmith run links into `reports/benchmark_report.md`.

## Milestones trong 2 giờ lab

| Thời lượng | Milestone | File gợi ý |
|---:|---|---|
| 0-15' | Setup, baseline chạy thật | `cli.py`, `services/llm_client.py` |
| 15-45' | Supervisor + workflow + guardrails | `agents/supervisor.py`, `graph/workflow.py` |
| 45-75' | Researcher / Analyst / Writer + search | `agents/*.py`, `services/search_client.py` |
| 75-95' | Trace + benchmark single vs multi | `observability/tracing.py`, `evaluation/benchmark.py` |
| 95-115' | Peer review theo rubric | `docs/peer_review_rubric.md` |
| 115-120' | Exit ticket | `docs/lab_guide.md` |

## Quy ước production trong repo

- Tách rõ `agents`, `services`, `core`, `graph`, `evaluation`, `observability`.
- Không hard-code API key trong code.
- Tất cả input/output chính dùng Pydantic schema.
- Có type hints, linting, formatting, unit test tối thiểu.
- Có logging/tracing hook ngay từ đầu.
- Không để agent chạy vô hạn: dùng `MAX_ITERATIONS`, `TIMEOUT_SECONDS`.
- Có benchmark report thay vì chỉ demo output đẹp.

## Mở rộng (TODO trong code)

```bash
grep -R "TODO(student)" -n src tests docs
```

Gợi ý:

1. **CriticAgent** — fact-check / coverage (`agents/critic.py`).
2. **LangGraph** — thay loop trong `MultiAgentWorkflow` bằng graph compile (`graph/workflow.py`).
3. **Tracing** — LangSmith / Langfuse trong `observability/tracing.py`.
4. **Benchmark** — cost, quality rubric, citation coverage (`evaluation/benchmark.py`, `evaluation/report.py`).

## Deliverables

Học viên nộp:

1. GitHub repo cá nhân.
2. Screenshot trace hoặc link trace.
3. `reports/benchmark_report.md` so sánh single vs multi-agent.
4. Một đoạn giải thích failure mode và cách fix.

## Troubleshooting

- **PowerShell multi-line commands fail**: don't use Bash `\` line continuation. Use one-line commands or PowerShell backtick.
- **Sources are `example.invalid`**: `TAVILY_API_KEY` not set, so search runs in mock/offline mode.
- **No LangSmith traces**: check `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`, then wait a few seconds and refresh the project page.

## References

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Agents SDK orchestration/handoffs — https://developers.openai.com/api/docs/guides/agents/orchestration
- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- Langfuse tracing — https://langfuse.com/docs
