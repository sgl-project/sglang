"""
Multi-stage forked workflow with synthesis — measures what's lost when SGLang is used
as a plain HTTP endpoint instead of via @sgl.function programs.

`industry_sweep` does `fork(N)` over a long shared prefix (sector context + question)
followed by a synthesis stage that aggregates the per-fork outputs.
`baseline_industry_sweep` runs the same workload as independent /generate POSTs (the
OpenAI-API-style path; no program structure visible to the server). The --benchmark
flag times both and prints the speedup ratio.

The program-aware path captures both prefix caching via the radix tree and fork-level
concurrency — both of which the raw-API path loses, falling back to runtime
hash-based prefix detection only.

Demo content is semiconductor research (NVDA / AMD / AVGO / INTC), sourced from
https://chenyu-li.info/blog/19/en/ — "Will AI CapEx Pay for Itself?" (Mar 2026).

Usage:
    python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 --port 30000
    python multi_stage_fork.py             # run the pattern
    python multi_stage_fork.py --benchmark # SGLang frontend vs raw-API baseline
"""

import argparse
import time

import requests

import sglang as sgl

# Stub data — publicly-known chipmaker facts. In production, load from a fundamentals pipeline.

SECTOR_CONTEXT = """You are analyzing AI infrastructure semiconductor companies under
an AI CapEx ROI framework.

Framework summary:
- Hyperscalers will spend $2.6-5.6T on AI infrastructure 2024-2035
- Three revenue scenarios (A=optimistic, B=moderate, C=pessimistic) × three capital
  intensity paths (Low, Mid, High) = 9 outcome combinations
- Capital intensity matters more than revenue for ROI
- NVIDIA benefits from high intensity; hyperscalers benefit from low intensity
- Every dollar of efficiency that improves cloud AI profitability is a dollar
  NVIDIA doesn't collect

Investment analysis dimensions:
- Compute architecture (training vs inference, dense vs MoE)
- Customer concentration and sticky workloads
- Software moat (CUDA, ROCm, Gaudi software stacks)
- Capital allocation discipline
- Margin trajectory under price competition
- Pricing power in supply-constrained vs supply-abundant scenarios
"""

SECTOR_COMPANIES = {
    "NVDA": """NVIDIA — dominant AI training silicon supplier. FY25 (calendar 2024)
revenue $130.5B (+114% YoY); datacenter segment $115B (+217% YoY). Gross margin 75%
peak. Market cap $4.45T per the blog framework analysis. H100/H200 generation has been
the dominant training workhorse 2023-2025. B100/B200 Blackwell generation in production
ramp. CUDA software stack and developer ecosystem are the moat. Bull case: AI capex
continues; bear case: gross margin compression as AMD gains share + inference workloads
commoditize.""",
    "AMD": """AMD — second-source AI silicon disruptor. FY24 revenue $25.8B (+14% YoY);
datacenter segment $12.6B (+94% YoY). MI300X launched Q4 2023; MI325X / MI355 in
production. Named hyperscaler customers: Microsoft (Azure ND-MI300X instances), Meta
(Llama inference), Oracle. ROCm software stack maturing; SGLang has first-class MI300X
support per AMD's published ROCm blog. Investment thesis: AMD becomes credible inference
alternative to NVIDIA; gross margin expansion as scale builds.""",
    "AVGO": """Broadcom — AI custom ASIC + networking. FY24 AI-specific revenue ~$12.2B
(rapid growth from Google TPU partnership and Meta MTIA partnership). Custom-silicon
strategy: each major hyperscaler designs its own accelerator + Broadcom manufactures.
Networking (Tomahawk switches, Jericho routers) is critical infrastructure for AI
clusters. Investment thesis: capture customer-specific compute spend that NVIDIA can't
serve via generic GPUs; networking is the moat-by-default.""",
    "INTC": """Intel — recovering from foundry pivot and AI execution gap. FY24 revenue
~$53B (declining); datacenter segment under pressure. Gaudi 3 AI accelerator launched
2024 but with limited customer traction. Foundry pivot (18A, 14A nodes) is multi-year
investment with no near-term AI silicon contribution. Investment thesis: deep value with
turnaround optionality; near-term competitive disadvantage in AI silicon vs NVDA/AMD.
Most negatively exposed to AI capex framework's high-intensity scenarios.""",
}

# industry_sweep — N companies fork off a shared sector context, then synthesize.


@sgl.function
def industry_sweep(s, sector_context, companies, research_question):
    """Parallel analysis across N companies sharing a sector-context prefix."""
    s += sector_context + "\n"
    s += "Research question: " + research_question + "\n\n"

    tickers = list(companies.keys())
    forks = s.fork(len(tickers))

    for i, ticker in enumerate(tickers):
        forks[i] += f"--- Company: {ticker} ---\n{companies[ticker]}\n\n"
        forks[i] += f"Analysis of {ticker} against the research question:\n"
        forks[i] += sgl.gen(
            f"analysis_{ticker}",
            max_tokens=400,
            stop="\n\n",
            temperature=0.3,
        )

    # Synthesis: gather per-company analyses, then rank
    s += "Per-company analyses gathered:\n"
    for i, ticker in enumerate(tickers):
        s += f"\n[{ticker}] " + forks[i][f"analysis_{ticker}"]
    s += "\n\nFinal ranking and top pick with rationale:\n"
    s += sgl.gen("sector_pick", max_tokens=400, temperature=0.2)


# Baseline: independent /generate POSTs without frontend program structure. The server
# falls back to runtime hash-based prefix detection (vLLM-style) rather than the
# program-aware caching the SGLang frontend enables. Used as the benchmark control.


def _raw_generate(
    text: str,
    max_tokens: int,
    temperature: float,
    stop: str = None,
    base_url: str = "http://localhost:30000",
    timeout: float = 300.0,
) -> str:
    """Make a single /generate POST without frontend program structure."""
    payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
        },
    }
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["text"]


def baseline_industry_sweep(
    sector_context: str, companies: dict, research_question: str
) -> dict:
    """Same workload as industry_sweep but as N independent /generate calls."""
    prefix = f"{sector_context}\nResearch question: {research_question}\n\n"

    per_company_analyses = {}
    for ticker, summary in companies.items():
        prompt = (
            prefix
            + f"--- Company: {ticker} ---\n{summary}\n\n"
            + f"Analysis of {ticker} against the research question:\n"
        )
        per_company_analyses[ticker] = _raw_generate(
            prompt, max_tokens=400, temperature=0.3, stop="\n\n"
        )

    # Synthesis call
    synthesis_prompt = prefix + "Per-company analyses gathered:\n"
    for ticker, analysis in per_company_analyses.items():
        synthesis_prompt += f"\n[{ticker}] {analysis}"
    synthesis_prompt += "\n\nFinal ranking and top pick with rationale:\n"
    sector_pick = _raw_generate(synthesis_prompt, max_tokens=400, temperature=0.2)

    return {"per_company": per_company_analyses, "sector_pick": sector_pick}


# Benchmark machinery

RESEARCH_QUESTION = (
    "Under the 3x3 capex/revenue framework, which AI infrastructure chipmaker "
    "has the most defensible position if AI revenue growth normalizes to the "
    "moderate trajectory?"
)


def _flush_cache(base_url: str = "http://localhost:30000"):
    """Reset SGLang server's radix tree between benchmark runs."""
    try:
        requests.post(f"{base_url}/flush_cache")
    except Exception as e:
        print(f"  (cache flush failed: {e}, continuing)")


def benchmark_industry_sweep():
    """Compare SGLang frontend vs OpenAI-equivalent baseline on industry_sweep."""
    print("=" * 70)
    print("BENCHMARK: industry_sweep")
    print("=" * 70)

    # Approach A: SGLang frontend (program-aware, fork-based)
    # NOTE: .run() returns non-blockingly after the Python body finishes
    # enqueueing IR nodes; we MUST call state.text() (which internally calls
    # self.sync() -> queue.join()) before stopping the timer, otherwise the
    # final sgl.gen("sector_pick") would still be in flight on the worker
    # thread when we measure elapsed time. See interpreter.py:404-406, 350-352.
    _flush_cache()
    start = time.perf_counter()
    state = industry_sweep.run(
        sector_context=SECTOR_CONTEXT,
        companies=SECTOR_COMPANIES,
        research_question=RESEARCH_QUESTION,
    )
    state.text()  # force sync — wait for all enqueued gens to complete
    sgl_elapsed = time.perf_counter() - start
    print(f"  SGLang frontend (program-aware): {sgl_elapsed:.2f}s")

    # Approach B: equivalent loop via raw /generate, no program structure
    _flush_cache()
    start = time.perf_counter()
    baseline_industry_sweep(
        sector_context=SECTOR_CONTEXT,
        companies=SECTOR_COMPANIES,
        research_question=RESEARCH_QUESTION,
    )
    baseline_elapsed = time.perf_counter() - start
    print(f"  OpenAI-equivalent (request-level): {baseline_elapsed:.2f}s")

    speedup = baseline_elapsed / sgl_elapsed if sgl_elapsed > 0 else float("nan")
    print(f"  Speedup: {speedup:.2f}x")
    print()
    print("  Note: speedup combines program-aware prefix caching via the radix tree")
    print("  with concurrent execution of the N forks. The OpenAI-equivalent baseline")
    print("  is sequential and uses runtime hash-based prefix detection.")
    return {
        "sgl_elapsed": sgl_elapsed,
        "baseline_elapsed": baseline_elapsed,
        "speedup": speedup,
    }


def driver_industry_sweep():
    print("=" * 70)
    print("industry_sweep — parallel cross-company analysis")
    print("=" * 70)
    state = industry_sweep.run(
        sector_context=SECTOR_CONTEXT,
        companies=SECTOR_COMPANIES,
        research_question=RESEARCH_QUESTION,
    )
    print(state.text())
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run SGLang frontend vs OpenAI-equivalent baseline comparison",
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:30000",
        help="SGLang server URL (default: %(default)s)",
    )
    args = parser.parse_args()

    sgl.set_default_backend(sgl.RuntimeEndpoint(args.backend_url))

    if args.benchmark:
        benchmark_industry_sweep()
    else:
        driver_industry_sweep()
