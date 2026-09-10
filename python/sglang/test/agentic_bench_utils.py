"""Utilities for running agentic multi-turn serving benchmarks.

Nightly perf tests in :mod:`test.registered` use these helpers to drive
``sglang.benchmark.serving --dataset-name agentic-trace`` over a concurrency
sweep and turn the resulting JSONL into a markdown report.

Agentic coding replay differs from the fixed-shape ``bench_one_batch_server``
sweeps the other nightly perf tests run: each conversation is a session whose
history grows turn by turn, so the interesting numbers are inter-token latency
under a growing context and how much of each turn's prompt the prefix cache
serves. ``NightlyBenchmarkRunner`` only drives ``bench_one_batch_server``
(batch size x input len x output len), which cannot express that, hence this
module.
"""

import json
import os
import random
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

# Env var pointing at a pre-built agentic trace JSON. When unset the caller
# falls back to `write_agentic_coding_trace`, so CI has a workload without
# depending on a corpus it cannot download.
AGENTIC_TRACE_PATH_ENV = "SGLANG_AGENTIC_TRACE_PATH"

# Rough BPE tokens-per-word for the code-and-prose mix below. Only used to
# turn the token budgets in `AgenticTraceSpec` into word counts; the replay
# itself never depends on hitting the budget exactly (`AgenticTraceDataset`
# treats each conversation's `prompt_tokens` as informational).
_TOKENS_PER_WORD = 1.3

_MODULES = (
    "scheduler",
    "tokenizer_manager",
    "radix_cache",
    "hicache_storage",
    "attention_backend",
    "moe_runner",
    "speculative_worker",
    "memory_pool",
    "model_runner",
    "server_args",
)

_SYMBOLS = (
    "allocate_token_slots",
    "match_prefix",
    "evict_host_pages",
    "prepare_for_decode",
    "resolve_future_tokens",
    "build_forward_batch",
    "capture_cuda_graph",
    "flush_write_queue",
    "select_experts",
    "verify_draft_tokens",
)

_TOOL_NAMES = ("read_file", "grep", "edit_file", "run_tests", "bash", "list_dir")


def _lorem_words(rng: random.Random, num_words: int) -> str:
    """Deterministic filler that tokenizes like source code and tracebacks.

    Realistic token shapes matter here: the replay measures prefill cost, and a
    stream of identical short words would compress into far fewer tokens per
    word than a real agent transcript.
    """
    parts = []
    while len(parts) < num_words:
        parts.extend(
            (
                f"{rng.choice(_MODULES)}.{rng.choice(_SYMBOLS)}",
                f"line={rng.randint(10, 4000)}",
                rng.choice(
                    (
                        "returns",
                        "expects",
                        "raises",
                        "asserts",
                        "wraps",
                        "reuses",
                        "invalidates",
                    )
                ),
                rng.choice(
                    (
                        "the page-aligned KV block",
                        "a host-tier prefetch",
                        "the draft token budget",
                        "an evicted radix node",
                        "the chunked prefill window",
                    )
                ),
            )
        )
    return " ".join(parts[:num_words])


def _block(rng: random.Random, num_tokens: int) -> str:
    return _lorem_words(rng, max(1, int(num_tokens / _TOKENS_PER_WORD)))


@dataclass(frozen=True)
class AgenticTraceSpec:
    """Shape of a synthesized agentic-coding corpus.

    Defaults approximate a mid-sized coding-agent session: a shared agent
    scaffold, a per-session repository context that dominates the first turn,
    and tool-output deltas that grow the history every turn.
    """

    num_conversations: int = 24
    turns_per_conversation: int = 8
    # Identical across conversations, so it models the cross-session prefix
    # every coding agent re-sends.
    system_prompt_tokens: int = 2048
    # Unique per conversation: the bulk of turn 1, and the prefix that every
    # later turn in the same session should hit in cache.
    repo_context_tokens: int = 32768
    # Tool output appended each turn.
    turn_tokens: int = 4096

    def first_turn_tokens(self) -> int:
        return self.system_prompt_tokens + self.repo_context_tokens + self.turn_tokens


def _system_prompt(spec: AgenticTraceSpec) -> str:
    rng = random.Random(0)
    tools = ", ".join(_TOOL_NAMES)
    head = (
        "You are a coding agent working inside a large Python and C++ "
        f"repository. You have these tools available: {tools}. Investigate "
        "before editing, keep patches minimal, and run the tests you touch. "
        "Reference implementation notes follow.\n"
    )
    return head + _block(rng, spec.system_prompt_tokens)


def write_agentic_coding_trace(
    path: str,
    spec: AgenticTraceSpec = AgenticTraceSpec(),
    seed: int = 42,
) -> str:
    """Write a synthetic agentic-coding trace in ``agentic-trace`` JSON form.

    The real SemiAnalysis/OpenHands corpora these benchmarks were built against
    are gated, so CI synthesizes a corpus with the same structure instead: a
    long shared scaffold, a long per-session context, and short tool-output
    deltas. Absolute throughput is therefore not comparable to a run over a
    real corpus, but the run-to-run comparison a nightly regression check needs
    holds, because the corpus is a deterministic function of ``spec`` and
    ``seed``.

    Returns the path written.
    """
    system_prompt = _system_prompt(spec)
    conversations = []

    for conv_index in range(spec.num_conversations):
        rng = random.Random(seed + conv_index)
        repo_context = _block(rng, spec.repo_context_tokens)
        turns = [
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Task {conv_index}: a regression landed in "
                            f"{rng.choice(_MODULES)}. Repository context "
                            f"follows.\n{repo_context}\n"
                            f"{_block(rng, spec.turn_tokens)}"
                        ),
                    },
                ],
                "prompt_tokens": spec.first_turn_tokens(),
            }
        ]

        for turn_index in range(1, spec.turns_per_conversation):
            tool = rng.choice(_TOOL_NAMES)
            turns.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Output of {tool} (call {turn_index}):\n"
                                f"{_block(rng, spec.turn_tokens)}"
                            ),
                        }
                    ],
                    "prompt_tokens": spec.first_turn_tokens()
                    + turn_index * spec.turn_tokens,
                }
            )

        conversations.append(turns)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "source": "sglang-synthetic-agentic-coding",
                    "seed": seed,
                    **{k: getattr(spec, k) for k in spec.__dataclass_fields__},
                },
                "conversations": conversations,
            },
            f,
        )
    return path


def resolve_agentic_trace(
    result_dir: str,
    spec: AgenticTraceSpec = AgenticTraceSpec(),
    seed: int = 42,
) -> str:
    """Return a trace path, preferring a real corpus over the synthetic one."""
    override = os.environ.get(AGENTIC_TRACE_PATH_ENV)
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(
                f"{AGENTIC_TRACE_PATH_ENV}={override} does not point to a file"
            )
        print(f"Using agentic trace from {AGENTIC_TRACE_PATH_ENV}: {override}")
        return override

    path = os.path.join(result_dir, "agentic_coding_trace.json")
    write_agentic_coding_trace(path, spec=spec, seed=seed)
    print(
        f"Synthesized agentic trace at {path} "
        f"({spec.num_conversations} conversations x "
        f"{spec.turns_per_conversation} turns, "
        f"~{spec.first_turn_tokens()} tokens on turn 1). "
        f"Set {AGENTIC_TRACE_PATH_ENV} to replay a real corpus instead."
    )
    return path


@dataclass
class AgenticBenchPoint:
    """One concurrency point of an agentic sweep.

    ``bench_serving`` reports no input throughput for multi-turn replay (it
    cannot attribute a prompt length to a turn it did not construct), so the
    prefill side shows up as cache hit rate rather than input tok/s.
    """

    concurrency: int
    conversations: int
    completed_turns: int
    duration_s: float
    output_throughput: float
    mean_ttft_ms: float
    p99_ttft_ms: float
    mean_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    achieved_concurrency: float
    accept_length: Optional[float] = None
    cache_hit_rate_pct: Optional[float] = None
    host_cached_tokens: Optional[int] = None
    raw: Dict = field(default_factory=dict, repr=False)


def _parse_bench_record(record: Dict, concurrency: int, conversations: int):
    cache_report = record.get("cache_report") or {}
    return AgenticBenchPoint(
        concurrency=concurrency,
        conversations=conversations,
        completed_turns=record.get("completed", 0),
        duration_s=record.get("duration", 0.0),
        output_throughput=record.get("output_throughput", 0.0),
        mean_ttft_ms=record.get("mean_ttft_ms", 0.0),
        p99_ttft_ms=record.get("p99_ttft_ms", 0.0),
        mean_itl_ms=record.get("mean_itl_ms", 0.0),
        p99_itl_ms=record.get("p99_itl_ms", 0.0),
        mean_e2e_latency_ms=record.get("mean_e2e_latency_ms", 0.0),
        achieved_concurrency=record.get("concurrency", 0.0),
        accept_length=record.get("accept_length"),
        cache_hit_rate_pct=cache_report.get("cache_hit_rate_pct"),
        host_cached_tokens=cache_report.get("host_cached_tokens"),
        raw=record,
    )


def run_agentic_concurrency_sweep(
    base_url: str,
    model_path: str,
    trace_path: str,
    result_dir: str,
    concurrencies: Sequence[int],
    conversations_per_slot: int = 2,
    max_turns: Optional[int] = None,
    output_len: Optional[int] = None,
    timeout: int = 3600,
    extra_bench_args: Optional[List[str]] = None,
) -> List[AgenticBenchPoint]:
    """Replay the trace once per concurrency level against a running server.

    ``conversations_per_slot`` scales the conversation count with concurrency so
    every point replays the same number of sequential waves; a fixed count would
    make low-concurrency points many times longer than high-concurrency ones.

    In CI ``bench_serving`` flushes the server cache before each measured run
    (see ``should_flush_cache`` in ``sglang.benchmark.serving``), so each point
    starts cold and measures only the prefix reuse it generates itself.
    """
    os.makedirs(result_dir, exist_ok=True)
    points: List[AgenticBenchPoint] = []

    for concurrency in concurrencies:
        conversations = concurrency * conversations_per_slot
        output_file = os.path.join(result_dir, f"agentic_conc{concurrency}.jsonl")
        if os.path.exists(output_file):
            os.remove(output_file)

        command = [
            "python3",
            "-m",
            "sglang.benchmark.serving",
            "--backend",
            "sglang-oai-chat",
            "--base-url",
            base_url,
            "--model",
            model_path,
            "--tokenizer",
            model_path,
            "--dataset-name",
            "agentic-trace",
            "--dataset-path",
            trace_path,
            "--num-prompts",
            str(conversations),
            "--max-concurrency",
            str(concurrency),
            "--warmup-requests",
            "1",
            "--cache-report",
            "--output-file",
            output_file,
            "--trust-remote-code",
        ]
        if max_turns is not None:
            command += ["--agentic-max-turns", str(max_turns)]
        if output_len is not None:
            command += ["--sharegpt-output-len", str(output_len)]
        if extra_bench_args:
            command += list(extra_bench_args)

        print(f"Running agentic replay at concurrency {concurrency}: {command}")
        result = subprocess.run(command, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"Agentic replay failed at concurrency {concurrency} "
                f"(exit code {result.returncode})"
            )
        if not os.path.exists(output_file):
            raise RuntimeError(
                f"Agentic replay at concurrency {concurrency} wrote no results "
                f"to {output_file}"
            )

        with open(output_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if not records:
            raise RuntimeError(f"{output_file} contains no benchmark records")

        point = _parse_bench_record(records[-1], concurrency, conversations)
        if point.completed_turns == 0:
            raise RuntimeError(
                f"Agentic replay at concurrency {concurrency} completed no turns"
            )
        points.append(point)

    return points


def run_agentic_benchmark(
    model_path: str,
    base_url: str,
    server_args: List[str],
    result_dir: str,
    concurrencies: Sequence[int],
    trace_spec: AgenticTraceSpec = AgenticTraceSpec(),
    conversations_per_slot: int = 2,
    max_turns: Optional[int] = None,
    output_len: Optional[int] = None,
    server_launch_timeout: int = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    bench_timeout: int = 3600,
    env: Optional[dict] = None,
) -> List[AgenticBenchPoint]:
    """Launch a server, run the concurrency sweep against it, then tear it down."""
    trace_path = resolve_agentic_trace(result_dir, spec=trace_spec)

    process = popen_launch_server(
        model=model_path,
        base_url=base_url,
        other_args=server_args,
        timeout=server_launch_timeout,
        env=env,
    )
    try:
        return run_agentic_concurrency_sweep(
            base_url=base_url,
            model_path=model_path,
            trace_path=trace_path,
            result_dir=result_dir,
            concurrencies=concurrencies,
            conversations_per_slot=conversations_per_slot,
            max_turns=max_turns,
            output_len=output_len,
            timeout=bench_timeout,
        )
    finally:
        kill_process_tree(process.pid)


def generate_agentic_markdown_report(
    points: Sequence[AgenticBenchPoint], header: str
) -> str:
    """Render a sweep as a markdown table for the GitHub step summary."""
    summary = f"### {header}\n"
    summary += (
        "| concurrency | conversations | turns | duration (s) | "
        "output throughput (tok/s) | mean TTFT (ms) | p99 TTFT (ms) | "
        "mean ITL (ms) | p99 ITL (ms) | mean E2E (ms) | cache hit (%) | "
        "accept len |\n"
    )
    summary += (
        "| ----------- | ------------- | ----- | ------------ | "
        "------------------------- | -------------- | ------------- | "
        "------------- | ------------ | ------------- | ------------- | "
        "---------- |\n"
    )

    for p in points:
        cache_hit = (
            f"{p.cache_hit_rate_pct:.1f}" if p.cache_hit_rate_pct is not None else "n/a"
        )
        accept = f"{p.accept_length:.2f}" if p.accept_length else "n/a"
        summary += (
            f"| {p.concurrency} | {p.conversations} | {p.completed_turns} | "
            f"{p.duration_s:.1f} | {p.output_throughput:.2f} | "
            f"{p.mean_ttft_ms:.2f} | {p.p99_ttft_ms:.2f} | "
            f"{p.mean_itl_ms:.2f} | {p.p99_itl_ms:.2f} | "
            f"{p.mean_e2e_latency_ms:.2f} | {cache_hit} | {accept} |\n"
        )

    return summary
