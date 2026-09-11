"""Cache/scheduling consistency test: score the same token contexts repeatedly
and require bitwise-identical greedy output tokens and top-k logprobs.

The workload generates a continuation for each independently sampled prefix,
then submits shuffled requests starting from different cuts of that
continuation. This scores each shared context multiple times after different
batching and cache histories and compares the resulting tokens and logprobs, so
a hybrid-cache (SWA + Mamba/sconv) or overlap-scheduler bug that corrupts a
reused context shows up as a diverging token or logprob rather than as a small
accuracy drop no eval would catch.

The engine is put under additional pressure in three ways: probabilistic timing
jitter is enqueued after CUDA event record and wait operations to amplify stream
races, ``max_total_tokens`` constrains the KV cache, and periodic request
retractions introduce further scheduling and cache disruption.

This assumes a batch-invariant serving path: reducing the same context under a
different batch shape must produce the same bits, or the differences reported are
in reduction order rather than in cache and scheduling behaviour.
"""

from __future__ import annotations

import gc
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import globaltimer, smid

if TYPE_CHECKING:
    from sglang.srt.entrypoints.engine import Engine


@triton.jit
def _random_jitter_kernel(
    enabled: tl.tensor,
    PROBABILITY: tl.constexpr,
    MAX_TIME_US: tl.constexpr,
) -> None:
    if tl.load(enabled) != 0:
        start = globaltimer().to(tl.uint64)
        sample = start ^ (smid().to(tl.uint64) << 32)
        sample ^= sample << 13
        sample ^= sample >> 7
        sample ^= sample << 17
        if sample.to(tl.uint32) < int(PROBABILITY * 2**32):
            sample ^= sample << 13
            sample ^= sample >> 7
            sample ^= sample << 17
            duration_ns = (
                sample.to(tl.uint32).to(tl.uint64) * (MAX_TIME_US * 1_000 + 1)
            ) >> 32
            now = globaltimer().to(tl.uint64)
            while now - start < duration_ns:
                now = globaltimer().to(tl.uint64)


def _random_jitter(enabled: torch.Tensor, probability: float, max_time_us: int) -> None:
    kernel: Any = _random_jitter_kernel
    kernel[(1,)](
        enabled,
        PROBABILITY=probability,
        MAX_TIME_US=max_time_us,
        num_warps=1,
    )


def _run_scheduler_process_with_jitter(
    server_args: Any,
    port_args: Any,
    gpu_id: int,
    *scheduler_args: Any,
    jitter_probability: float,
    jitter_max_time_us: int,
    **scheduler_kwargs: Any,
) -> None:
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    original_record = torch.cuda.Event.record
    original_wait = torch.cuda.Event.wait
    original_capture_end = torch.cuda.CUDAGraph.capture_end
    original_flush_cache = Scheduler.flush_cache
    capture_streams: dict[tuple[torch.device, int], torch.cuda.Stream] = {}
    probability = jitter_probability / server_args.tp_size
    jitter_active = False

    # Compile and load the kernel before an event hook can run during CUDA
    # graph capture, where Triton compilation is not allowed. Captured kernels
    # retain this device address and read its current value at every replay.
    with torch.cuda.device(gpu_id):
        enabled = torch.zeros(1, dtype=torch.uint8, device="cuda")
        _random_jitter(enabled, probability, jitter_max_time_us)
        torch.cuda.synchronize()

    def wrap_event(original: Any) -> Any:
        def with_jitter(
            event: torch.cuda.Event,
            stream: torch.cuda.Stream | None = None,
        ) -> None:
            original(event, stream)
            target_stream = torch.cuda.current_stream() if stream is None else stream
            with torch.cuda.stream(target_stream):
                capturing = torch.cuda.is_current_stream_capturing()
                if capturing or jitter_active:
                    _random_jitter(enabled, probability, jitter_max_time_us)
            if capturing:
                capture_streams[(target_stream.device, target_stream.cuda_stream)] = (
                    target_stream
                )

        return with_jitter

    def flush_cache(self: Any, empty_cache: bool = True) -> bool:
        nonlocal jitter_active
        success = original_flush_cache(self, empty_cache)
        if success and not jitter_active:
            enabled.fill_(1)
            torch.cuda.synchronize()
            jitter_active = True
        return success

    torch.cuda.Event.record = wrap_event(original_record)
    torch.cuda.Event.wait = wrap_event(original_wait)
    Scheduler.flush_cache = flush_cache

    # A post-record delay can extend a captured side stream past the
    # event that originally joined it. Join those streams before ending
    # capture; eager execution needs no corresponding special case.
    def capture_end(self: torch.cuda.CUDAGraph) -> None:
        origin_stream = torch.cuda.current_stream()
        origin_key = (origin_stream.device, origin_stream.cuda_stream)
        try:
            for key, source_stream in capture_streams.items():
                if key == origin_key:
                    continue
                done = torch.cuda.Event()
                original_record(done, source_stream)
                original_wait(done, origin_stream)
            original_capture_end(self)
        finally:
            capture_streams.clear()

    torch.cuda.CUDAGraph.capture_end = capture_end
    print(
        "Installed stream-sync jitter "
        f"(probability={jitter_probability}, world_size={server_args.tp_size}, "
        f"per_rank_probability={probability}, "
        f"time_us=uniform[0,{jitter_max_time_us}], starts_after_cache_flush=True)",
        flush=True,
    )

    run_scheduler_process(
        server_args,
        port_args,
        gpu_id,
        *scheduler_args,
        **scheduler_kwargs,
    )


@contextmanager
def get_jitter_engine(
    *,
    jitter_probability: float = 0.1,
    jitter_max_time_us: int = 10_000,
    retract_interval: int = 500,
    **engine_kwargs: Any,
) -> Iterator[Engine]:
    from sglang.srt.entrypoints.engine import Engine

    assert 0.0 <= jitter_probability <= 1.0, f"{jitter_probability=}"
    assert jitter_max_time_us > 0, f"{jitter_max_time_us=}"
    assert retract_interval > 0, f"{retract_interval=}"
    assert "model_path" in engine_kwargs, "model_path is required"

    # Force chunked prefill and optimistic admission into the deliberately
    # undersized pool; metrics let the workload verify that retraction occurred.
    engine_kwargs.setdefault("chunked_prefill_size", 512)
    engine_kwargs.setdefault("enable_metrics", True)
    engine_kwargs.setdefault("max_total_tokens", 20_480)
    engine_kwargs.setdefault("schedule_conservativeness", 0.05)

    if (
        engine_kwargs.get("enable_dp_attention")
        and "dist_init_addr" not in engine_kwargs
    ):
        import portpicker

        from sglang.srt.server_args import DP_ATTENTION_HANDSHAKE_PORT_DELTA

        port = portpicker.pick_unused_port_range(  # pyright: ignore[reportAttributeAccessIssue]
            DP_ATTENTION_HANDSHAKE_PORT_DELTA + 1
        )[0]
        engine_kwargs["dist_init_addr"] = f"127.0.0.1:{port}"

    scheduler_process = (
        partial(
            _run_scheduler_process_with_jitter,
            jitter_probability=jitter_probability,
            jitter_max_time_us=jitter_max_time_us,
        )
        if jitter_probability > 0
        else Engine.run_scheduler_process_func
    )

    class JitterEngine(Engine):
        run_scheduler_process_func = staticmethod(scheduler_process)

    engine = None
    test_env = {
        "SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS": "True",
        "SGLANG_TEST_RETRACT": "True",
        "SGLANG_TEST_RETRACT_INTERVAL": str(retract_interval),
    }
    original_test_env = {name: os.environ.get(name) for name in test_env}
    os.environ.update(test_env)
    # enable_metrics registers collectors on the process-global prometheus
    # registry, so a second engine in the same process (a CustomTestCase retry)
    # would die on re-registration and hide the original failure.
    from prometheus_client import REGISTRY as _prom_registry

    collectors_before = set(_prom_registry._collector_to_names)
    try:
        engine = JitterEngine(**engine_kwargs)
        yield engine
    finally:
        if engine is not None:
            engine.shutdown()
            del engine
            gc.collect()
            torch.cuda.empty_cache()
        for collector in set(_prom_registry._collector_to_names) - collectors_before:
            _prom_registry.unregister(collector)
        for name, value in original_test_env.items():
            if value is None:
                os.environ.pop(name)
            else:
                os.environ[name] = value


# --- internals -------------------------------------------------------------


class _Request(NamedTuple):
    prefix: int
    cut: int
    input_ids: list[int]
    max_new_tokens: int

    @property
    def label(self) -> str:
        return f"prefix{self.prefix}/cut{self.cut}"


# An observation key identifies a predicted continuation position within one
# independently sampled prefix and its baseline continuation.
_Key = tuple[int, int]
# (token id at position, top-k token->logprob)
_Obs = tuple[int, dict[int, float]]


def _topk_equal(a: dict[int, float], b: dict[int, float]) -> bool:
    """Compare top-k maps bitwise, tolerating rank-k boundary ties."""
    if len(a) != len(b):
        return False
    if any(a[token] != b[token] for token in a.keys() & b.keys()):
        return False
    return all(
        source[token] == min(other.values())
        for only, source, other in (
            (a.keys() - b.keys(), a, b),
            (b.keys() - a.keys(), b, a),
        )
        for token in only
    )


def _record_response(
    observations: dict[_Key, list[_Obs]],
    req: _Request,
    out: dict[str, Any],
    baseline: list[int] | None,
) -> list[int]:
    """Record every scored position of one response; returns its output ids.

    `baseline` is the prefix's expected greedy continuation (None while
    recording the baseline itself). Positions past the first divergence from
    the baseline are dropped: their context differs from every other request
    at the same nominal position, so comparisons there are meaningless. The
    divergence position itself still has identical context and is recorded —
    its top-k diff is exactly the interesting signal.
    """

    meta = out["meta_info"]

    out_ids = [int(t) for t in out["output_ids"]]
    out_lps = meta["output_token_logprobs"]
    out_top = meta["output_top_logprobs"]
    assert len(out_ids) == req.max_new_tokens, (
        f"{req.label}: got {len(out_ids)} output tokens, expected {req.max_new_tokens}"
    )
    expected = out_ids if baseline is None else baseline[req.cut :]
    for m, (tid, lp_entry, top) in enumerate(
        zip(out_ids, out_lps, out_top, strict=True)
    ):
        assert int(lp_entry[1]) == tid, f"{req.label}: output logprob misaligned at {m}"
        pos = len(req.input_ids) + m
        topk_map = {int(entry[1]): float(entry[0]) for entry in top}
        assert len(topk_map) == len(top), f"duplicate token ids in top-k: {top}"
        observations[(req.prefix, pos)].append((tid, topk_map))
        if tid != expected[m]:
            break
    return out_ids


def _generate_batch(
    engine: Engine, requests: list[_Request], topk: int
) -> list[dict[str, Any]]:
    """One batched generate call; per-request fields go through the batch
    API's list parameters, results come back in submission order."""
    outs = engine.generate(
        input_ids=[list(r.input_ids) for r in requests],
        sampling_params=[
            {"temperature": 0.0, "max_new_tokens": r.max_new_tokens, "ignore_eos": True}
            for r in requests
        ],
        return_logprob=[True] * len(requests),
        # -1 requests output logprobs only and permits full radix-cache reuse.
        logprob_start_len=[-1] * len(requests),
        top_logprobs_num=[topk] * len(requests),
    )
    results = list(outs)
    assert all(isinstance(r, dict) for r in results)
    return results


def _total_retracted_requests() -> float:
    from prometheus_client import CollectorRegistry
    from prometheus_client import multiprocess as prom_multiprocess

    assert "PROMETHEUS_MULTIPROC_DIR" in os.environ
    registry = CollectorRegistry()
    prom_multiprocess.MultiProcessCollector(registry)
    return sum(
        sample.value
        for metric in registry.collect()
        if "retracted_req" in metric.name
        for sample in metric.samples
        if sample.name.endswith("_total")
    )


def run_jitter_test(
    engine: Engine,
    *,
    num_unique_prefixes: int = 10,  # independently sampled random-token prefixes
    requests_per_prefix: int = 10,  # output-prefix cut requests per prefix
    prefix_len_min: int = 2048,  # prefix length range, inclusive
    prefix_len_max: int = 4096,
    new_tokens: int = 2048,  # baseline generation length
    topk: int = 8,  # how many logprob entries per position are compared
    seed: int = 0,  # seeds every draw: token sequence, lengths, cuts, shuffle
    min_retracted_requests: int = 10,
    vocab_size: int = 199_998,  # excludes special tokens / padded embedding tail
) -> None:
    """Run the workload on a caller-constructed engine and assert bitwise
    consistency across all overlapping observations."""
    assert min_retracted_requests >= 0
    prefix_rng = random.Random(seed + 1)
    prefix_lens = sorted(
        prefix_rng.randint(prefix_len_min, prefix_len_max)
        for _ in range(num_unique_prefixes)
    )
    # Random output-cut lengths per (prefix, cut) pair, capped so every
    # request still decodes at least 64 tokens; 0 stays in range (an exact
    # duplicate of the baseline scheduled in a different batch mix =
    # run-to-run check).
    cut_rng = random.Random(seed + 2)
    cut_lens = [
        [cut_rng.randint(0, new_tokens - 64) for _ in range(requests_per_prefix)]
        for _ in range(num_unique_prefixes)
    ]
    token_rng = random.Random(seed)
    prefix_tokens = [
        [token_rng.randrange(vocab_size) for _ in range(prefix_len)]
        for prefix_len in prefix_lens
    ]

    observations: dict[_Key, list[_Obs]] = defaultdict(list)

    # --- Phase 1: greedy baselines, one request per prefix; outputs define
    # the prefixes' canonical continuations ---
    t1 = time.monotonic()
    baseline_reqs = [
        _Request(prefix, 0, tokens, new_tokens)
        for prefix, tokens in enumerate(prefix_tokens)
    ]
    outs = _generate_batch(engine, baseline_reqs, topk)
    print(f"[timing] phase 1 (baselines): {time.monotonic() - t1:.1f}s")
    baselines = {}
    for req, out in zip(baseline_reqs, outs, strict=True):
        baselines[req.prefix] = _record_response(observations, req, out, baseline=None)

    # The scheduler only flushes when fully idle; the batch API can return
    # before the last request has fully drained, so retry briefly.
    for attempt in range(10):
        flush_result = engine.flush_cache()
        if getattr(flush_result, "success", True):
            break
        time.sleep(1.0)
    else:
        raise AssertionError(f"cache flush failed after retries: {flush_result}")
    retractions_before_jitter = (
        _total_retracted_requests() if min_retracted_requests else 0.0
    )

    # --- Phase 2: shuffled jitter batch — per prefix, requests from the
    # prefix plus a random-length cut of the baseline output, decoding out to
    # the baseline's total length ---
    jitter_reqs = [
        _Request(
            prefix,
            cut,
            tokens + baselines[prefix][:cut],
            new_tokens - cut,
        )
        for prefix, tokens in enumerate(prefix_tokens)
        for cut in cut_lens[prefix]
    ]
    random.Random(seed).shuffle(jitter_reqs)
    t2 = time.monotonic()
    jitter_outs = _generate_batch(engine, jitter_reqs, topk)
    print(f"[timing] phase 2 (jitter batch): {time.monotonic() - t2:.1f}s")
    for req, out in zip(jitter_reqs, jitter_outs, strict=True):
        _record_response(observations, req, out, baseline=baselines[req.prefix])

    # --- Cross-check every observation sharing a key ---
    num_cross_checked_positions = 0
    num_mismatches = 0
    mismatch_details: list[str] = []
    for key in sorted(observations):
        group = observations[key]
        if len(group) >= 2:
            num_cross_checked_positions += 1
        ref_tid, ref_topk = group[0]
        for tid, top in group[1:]:
            if tid == ref_tid and _topk_equal(ref_topk, top):
                continue
            num_mismatches += 1
            if len(mismatch_details) < 5:
                shared = ref_topk.keys() & top.keys()
                mismatch_details.append(
                    f"prefix={key[0]} pos={key[1]} token {ref_tid} vs {tid} "
                    f"max|dlogprob|="
                    f"{max((abs(ref_topk[t] - top[t]) for t in shared), default=0.0):.3e}"
                )
    print(
        f"checked {len(observations)} positions, "
        f"{num_cross_checked_positions} observed by >=2 requests, "
        f"{sum(len(g) for g in observations.values())} observations total, "
        f"{num_mismatches} mismatches"
    )
    # Guard against the test silently becoming vacuous: most continuation
    # positions must be observed by several requests.
    assert num_cross_checked_positions >= len(prefix_lens) * new_tokens // 2, (
        f"only {num_cross_checked_positions} positions were cross-checked; "
        "the request construction no longer produces overlapping observations"
    )
    assert num_mismatches == 0, "\n".join(
        [f"{num_mismatches} mismatching observations"] + mismatch_details
    )
    if min_retracted_requests:
        retractions_during_jitter = (
            _total_retracted_requests() - retractions_before_jitter
        )
        assert retractions_during_jitter >= min_retracted_requests, (
            f"expected at least {min_retracted_requests} retractions during the jitter "
            f"batch, got {retractions_during_jitter}"
        )
