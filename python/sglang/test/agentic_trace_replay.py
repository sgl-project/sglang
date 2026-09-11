"""Replay an agentic trace turn by turn, generating each turn's recorded reply length.

``bench_serving --dataset-name agentic-trace`` already walks a multi-turn
conversation round by round, but it asks every round for the same number of
output tokens. Agentic replies span two orders of magnitude inside a single
trajectory (20 to 3693 tokens in the corpus
:mod:`sglang.test.agentic_trace_utils` converts), and the next turn's prompt
contains the previous reply, so one uniform length both misstates the decode
work and makes the replayed prompt drift away from the recorded one.

This driver replays the same trace files while asking each round for the reply
length its turn recorded. It is deliberately a thin loop on top of
``bench_serving``: the HTTP/streaming path, TTFT/ITL accounting, cache reporting
and the metrics summary are the shared implementation, and only the per-turn
sequencing is new. That also keeps the numbers comparable with the rest of the
``bench_serving`` fleet.

``run_agentic_replay`` publishes ``bench_serving``'s process-global argument bag
(``set_global_args``), so a process that also runs ``sglang.bench_serving``
must set it back.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from sglang.benchmark.serving import (
    RequestFuncInput,
    RequestFuncOutput,
    async_request_openai_chat_completions,
    calculate_metrics,
    flush_server_cache,
    get_auth_headers,
    set_global_args,
)

# The replay is a chat workload by construction: it depends on the server
# applying the chat template and on assistant turns accumulating in the history.
BACKEND = "sglang-oai-chat"

# Same fallback as AgenticTraceDataset, for traces without recorded lengths.
DEFAULT_OUTPUT_LEN = 220

DEFAULT_PROGRESS_INTERVAL = 30.0


@dataclass
class ReplayTurn:
    """One round: the messages it adds, and the reply the trace recorded."""

    messages: List[Dict[str, str]]
    output_len: int
    prompt_tokens: int


def load_conversations(
    trace_path: str,
    num_conversations: int = 0,
    max_turns: Optional[int] = None,
    output_len: Optional[int] = None,
) -> List[List[ReplayTurn]]:
    """Load an agentic-trace JSON document into replayable conversations.

    Accepts the format ``AgenticTraceDataset`` reads: a ``conversations`` list
    of turn lists, each turn holding the ``messages`` it contributes and,
    optionally, the ``output_tokens`` recorded for its reply. ``output_len``
    overrides the recorded lengths with a uniform value.
    """
    with open(trace_path, "r", encoding="utf-8") as f:
        document = json.load(f)

    conversations = []
    for raw_conversation in document.get("conversations", []):
        if num_conversations and len(conversations) >= num_conversations:
            break

        turns = []
        for turn in raw_conversation:
            if max_turns and len(turns) >= max_turns:
                break
            messages = turn.get("messages")
            if not messages:
                continue
            turns.append(
                ReplayTurn(
                    messages=[
                        {"role": m["role"], "content": m["content"]} for m in messages
                    ],
                    output_len=int(
                        output_len or turn.get("output_tokens") or DEFAULT_OUTPUT_LEN
                    ),
                    prompt_tokens=int(turn.get("prompt_tokens", 0)),
                )
            )
        if turns:
            conversations.append(turns)

    if not conversations:
        raise ValueError(f"No usable conversations in {trace_path}")
    return conversations


def _bench_serving_globals(cache_report: bool) -> Namespace:
    """The flags ``bench_serving``'s chat request function reads off its globals."""
    return Namespace(
        disable_stream=False,
        # Ask for exactly the recorded number of tokens, so the replayed decode
        # length is the trace's and not the model's stopping behaviour.
        disable_ignore_eos=False,
        cache_report=cache_report,
        print_requests=False,
        header=None,
    )


def _request_input(
    turn: ReplayTurn,
    history: List[Dict[str, str]],
    api_url: str,
    model: str,
    output_len: Optional[int] = None,
) -> RequestFuncInput:
    return RequestFuncInput(
        prompt=list(history),
        api_url=api_url,
        prompt_len=turn.prompt_tokens,
        output_len=output_len or turn.output_len,
        model=model,
        lora_name="",
        image_data=None,
        extra_request_body={},
    )


class _Progress:
    """Rate-limited progress lines; a silent 15-minute CI step is unreadable."""

    def __init__(self, total_conversations: int, total_turns: int, interval: float):
        self.total_conversations = total_conversations
        self.total_turns = total_turns
        self.interval = interval
        self.started = time.perf_counter()
        self.last_print = self.started
        self.conversations = 0
        self.turns = 0

    def turn_done(self):
        self.turns += 1
        self._maybe_print()

    def conversation_done(self):
        self.conversations += 1
        self._maybe_print(force=self.conversations == self.total_conversations)

    def _maybe_print(self, force: bool = False):
        now = time.perf_counter()
        if not force and now - self.last_print < self.interval:
            return
        self.last_print = now
        print(
            f"[agentic-replay] {self.conversations}/{self.total_conversations} "
            f"conversations, {self.turns}/{self.total_turns} turns, "
            f"{now - self.started:.0f}s elapsed",
            flush=True,
        )


@contextlib.asynccontextmanager
async def _lane(semaphore: Optional[asyncio.Semaphore]):
    if semaphore is None:
        yield
        return
    async with semaphore:
        yield


async def _replay_conversation(
    conversation: List[ReplayTurn],
    api_url: str,
    model: str,
    semaphore: Optional[asyncio.Semaphore],
    progress: _Progress,
) -> List[RequestFuncOutput]:
    """Replay one trajectory: each turn waits for the previous reply.

    The semaphore is held for the whole trajectory, which is also how
    ``bench_serving`` bounds its multi-turn replay: one lane, one in-flight
    request, so the limit is both conversations and requests in flight.
    """
    history: List[Dict[str, str]] = []
    outputs: List[RequestFuncOutput] = []

    async with _lane(semaphore):
        for turn in conversation:
            history.extend(turn.messages)
            output = await async_request_openai_chat_completions(
                _request_input(turn, history, api_url, model)
            )
            outputs.append(output)
            progress.turn_done()
            if not output.success:
                # Every later turn replays this reply as context, so the rest
                # of the trajectory would measure a prompt the trace never
                # contained.
                break
            history.append({"role": "assistant", "content": output.generated_text})

    progress.conversation_done()
    return outputs


async def _replay_all(
    conversations: List[List[ReplayTurn]],
    api_url: str,
    model: str,
    max_concurrency: Optional[int],
    progress_interval: float,
) -> List[RequestFuncOutput]:
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    progress = _Progress(
        total_conversations=len(conversations),
        total_turns=sum(len(c) for c in conversations),
        interval=progress_interval,
    )
    per_conversation = await asyncio.gather(
        *(
            _replay_conversation(conversation, api_url, model, semaphore, progress)
            for conversation in conversations
        )
    )
    return [output for outputs in per_conversation for output in outputs]


def _server_accept_length(base_url: str) -> Optional[float]:
    """Speculative accept length, the way ``bench_serving`` reads it."""
    try:
        response = requests.get(base_url + "/server_info", headers=get_auth_headers())
        if response.status_code != 200:
            return None
        server_info = response.json()
        if "decode" in server_info:
            server_info = server_info["decode"][0]
        states = server_info.get("internal_states") or []
        return states[0].get("avg_spec_accept_length") if states else None
    except Exception:
        return None


def _cache_report(outputs: List[RequestFuncOutput], total_prompt_tokens: int) -> dict:
    """Prefix-cache hits over the prompt tokens the replay actually sent.

    ``total_prompt_tokens`` is summed per turn, not per conversation: the
    prompt of turn ``k`` is re-sent in full, and counting only a trajectory's
    first prompt would report a hit rate above 100%.
    """
    total_cached = total_device = total_host = total_storage = 0
    storage_backend = None
    has_details = False
    for output in outputs:
        if not output.success:
            continue
        total_cached += output.cached_tokens
        details = output.cached_tokens_details
        if details:
            has_details = True
            total_device += details.get("device") or 0
            total_host += details.get("host") or 0
            total_storage += details.get("storage") or 0
            storage_backend = storage_backend or details.get("storage_backend")

    hit_rate = total_cached / total_prompt_tokens * 100 if total_prompt_tokens else 0.0
    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_cached_tokens": total_cached,
        "cache_hit_rate_pct": round(hit_rate, 2),
        "device_cached_tokens": total_device if has_details else None,
        "host_cached_tokens": total_host if has_details else None,
        "storage_cached_tokens": total_storage if total_storage else None,
        "storage_backend": storage_backend,
    }


def run_agentic_replay(
    base_url: str,
    model: str,
    tokenizer,
    trace_path: str,
    num_conversations: int = 0,
    max_turns: Optional[int] = None,
    output_len: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    warmup: bool = True,
    flush_cache: bool = True,
    cache_report: bool = True,
    progress_interval: float = DEFAULT_PROGRESS_INTERVAL,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Replay ``trace_path`` against ``base_url`` and return the usual metrics.

    The result carries ``bench_serving``'s own key names, plus the replay's own
    ``total_turns`` / ``failed`` counts, and is appended to ``output_file`` as
    one JSON line when given.
    """
    conversations = load_conversations(
        trace_path,
        num_conversations=num_conversations,
        max_turns=max_turns,
        output_len=output_len,
    )
    total_turns = sum(len(c) for c in conversations)
    api_url = base_url.rstrip("/") + "/v1/chat/completions"
    set_global_args(_bench_serving_globals(cache_report))

    print(
        f"#Conversations: {len(conversations)} (turns={total_turns}, "
        f"max turns/conv={max(len(c) for c in conversations)})\n"
        f"#Output tokens per turn: "
        + (
            f"uniform {output_len}"
            if output_len
            else "recorded, min={} max={} avg={:.1f}".format(
                min(t.output_len for c in conversations for t in c),
                max(t.output_len for c in conversations for t in c),
                sum(t.output_len for c in conversations for t in c) / total_turns,
            )
        ),
        flush=True,
    )

    if warmup:
        first = conversations[0][0]
        print("Starting warmup with 1 conversation turn...", flush=True)
        asyncio.run(
            async_request_openai_chat_completions(
                _request_input(
                    first,
                    list(first.messages),
                    api_url,
                    model,
                    output_len=min(first.output_len, 32),
                )
            )
        )
    if flush_cache:
        # Keep the warmup's prefix out of the measured run.
        flush_server_cache(base_url, BACKEND)
        time.sleep(1.0)

    start = time.perf_counter()
    outputs = asyncio.run(
        _replay_all(conversations, api_url, model, max_concurrency, progress_interval)
    )
    duration = time.perf_counter() - start

    accept_length = _server_accept_length(base_url)
    metrics, _ = calculate_metrics(
        input_requests=None,
        outputs=outputs,
        dur_s=duration,
        tokenizer=tokenizer,
        backend=BACKEND,
        accept_length=accept_length,
    )

    # calculate_metrics only counts input tokens for single-turn rows, so the
    # prompt side is summed here from the turns that actually completed.
    completed_turns = [output for output in outputs if output.success]
    total_input_tokens = sum(output.prompt_len for output in completed_turns)

    result = {
        "backend": BACKEND,
        "dataset_name": "agentic-trace",
        "max_concurrency": max_concurrency,
        "duration": duration,
        "completed": metrics.completed,
        "total_turns": total_turns,
        "failed": total_turns - metrics.completed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": metrics.total_output,
        "total_output_tokens_retokenized": metrics.total_output_retokenized,
        "request_throughput": metrics.request_throughput,
        "input_throughput": total_input_tokens / duration,
        "output_throughput": metrics.output_throughput,
        "total_throughput": (total_input_tokens + metrics.total_output) / duration,
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
        "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "concurrency": metrics.concurrency,
        "accept_length": accept_length,
        "errors": sorted({o.error for o in outputs if not o.success and o.error})[:3],
    }
    if cache_report:
        result["cache_report"] = _cache_report(outputs, total_input_tokens)

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    return result
