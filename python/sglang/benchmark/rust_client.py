"""Rust load-generation client for the sglang-native `/generate` backend.

Thin wrapper around the `sglang_bench` extension (built from
`rust/sglang-bench`). Python prepares the workload — payload JSON, arrival
offsets — and computes the metrics; the extension owns the hot loop (pacing,
concurrency limiting, HTTP streaming, SSE/JSON parsing, per-chunk timing) on
tokio threads, off the GIL. The GIL is crossed twice per run: `start_run` and
`results()`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from typing import TYPE_CHECKING, Dict, List, Optional

import msgspec
import orjson

if TYPE_CHECKING:
    from tqdm.asyncio import tqdm

    from sglang.benchmark.serving import RequestFuncOutput

_PBAR_POLL_INTERVAL_S = 0.1


class PreparedRequest(msgspec.Struct, frozen=True):
    """One `/generate` request with its body already serialized, so payload
    semantics are decided by the caller and never diverge from the Python
    client path."""

    payload: bytes
    prompt_len: int
    output_len: int
    arrival_offset_s: float
    routing_key: Optional[str] = None


def is_available() -> bool:
    return importlib.util.find_spec("sglang_bench") is not None


async def run_requests_rust(
    *,
    requests: List[PreparedRequest],
    api_url: str,
    headers: Dict[str, str],
    routing_key_header: str,
    max_concurrency: Optional[int],
    cache_report: bool,
    pbar: Optional[tqdm],
) -> List[RequestFuncOutput]:
    """Run all requests through the Rust client; block only on a coarse
    progress poll. Outputs come back in submission order."""
    import sglang_bench

    from sglang.benchmark.serving import RequestFuncOutput

    # The Rust side reports times relative to the instant `start_run` is
    # entered; anchor re-bases `start_time` onto `time.perf_counter()`.
    anchor = time.perf_counter()
    run = sglang_bench.start_run(
        [
            (r.payload, r.prompt_len, r.output_len, r.arrival_offset_s, r.routing_key)
            for r in requests
        ],
        api_url=api_url,
        headers=list(headers.items()),
        routing_key_header=routing_key_header,
        max_concurrency=max_concurrency,
        cache_report=cache_report,
    )

    reported = 0
    while not run.is_done():
        await asyncio.sleep(_PBAR_POLL_INTERVAL_S)
        if pbar is not None:
            completed = run.completed()
            if completed > reported:
                pbar.update(completed - reported)
                reported = completed
    if pbar is not None and run.total() > reported:
        pbar.update(run.total() - reported)

    outputs = []
    for raw in run.results():
        output = RequestFuncOutput()
        output.generated_text = raw.generated_text
        output.success = raw.success
        output.latency = raw.latency
        output.ttft = raw.ttft
        output.itl = raw.itl
        output.prompt_len = raw.prompt_len
        output.error = raw.error
        output.output_len = raw.output_len
        output.start_time = anchor + raw.start_time
        output.cached_tokens = raw.cached_tokens
        if raw.cached_tokens_details_json is not None:
            output.cached_tokens_details = orjson.loads(raw.cached_tokens_details_json)
        output.spec_accept_length = raw.spec_accept_length
        if not output.success:
            print(f"{output.error=}")
        outputs.append(output)
    return outputs
