"""Timing records for the MoE-LoRA laboratory.

The measurement engine is the repo's own kernel-benchmark harness,
``sglang.kernels.jit.benchmark.marker`` — CUDA-event timing, warmup, quantile
metrics, graph-replay mode, capacity-derived L2 rotation, and bandwidth from a
declared memory footprint.  This module adds only what the campaign's evidence
discipline needs on top of it (execution plan section 31.2):

1. a JSON sink of content-addressed records, so a number can be re-adjudicated
   without a rerun;
2. a declared measurement BOUNDARY on every record, because plan section 10
   forbids comparing across boundaries;
3. matched-base pairing, so a LoRA measurement carries the base-only
   denominator measured in the SAME session rather than a remembered one;
4. an explicit cache-state selector, because plan section 7.4 requires both a
   cold and a hot reading and the default rotation only produces the former.

Nothing here decides anything.  Selection lives in the gate packets.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from typing import Any, Callable

import msgspec
import torch

from sglang.kernels.jit.benchmark.marker import BenchResult, do_bench

# Plan section 10. A comparison may only be made between records that carry the
# same boundary string.
BOUNDARY_ISOLATED = "isolated"
BOUNDARY_PREPARED_INPUT = "prepared_input"
BOUNDARY_ROUTE_INCLUSIVE = "route_inclusive"
BOUNDARY_COMPLETE_LOCAL_MOE = "complete_local_moe"
BOUNDARIES = (
    BOUNDARY_ISOLATED,
    BOUNDARY_PREPARED_INPUT,
    BOUNDARY_ROUTE_INCLUSIVE,
    BOUNDARY_COMPLETE_LOCAL_MOE,
)

# Cache state is DERIVED from the timing mode, not chosen by the caller.
#
# Methodology-audit finding (2026-07-25): the previous "cold"/"hot" selector
# was a NO-OP for zero-argument thunks — marker's rotation machinery sizes the
# rotation from the nbytes of input_args, and a closure passes none, so
# rotate_count was always 1 and every record labelled "cold" was in fact a
# hot-L2 steady state.  The labels below say what marker actually produces:
#
# "l2_hot_graph": use_cuda_graph=True with a closure.  100 in-graph iterations
# on the same addresses; median is a hot-L2 steady state.  CPU launch cost is
# EXCLUDED — replay does not re-run Python or launches.
#
# "l2_flushed_eager": use_cuda_graph=False.  marker zeroes an L2-sized buffer
# before every timed iteration, so inputs are genuinely cold, and the timing
# INCLUDES per-call CPU launch work between the events on the stream.
#
# A producer-realistic state still requires timing the producer+consumer pair
# at a common boundary (plan section 7.4); no flag here can produce it.
CACHE_L2_HOT_GRAPH = "l2_hot_graph"
CACHE_L2_FLUSHED_EAGER = "l2_flushed_eager"
CACHE_STATES = (CACHE_L2_HOT_GRAPH, CACHE_L2_FLUSHED_EAGER)


class TimingRecord(msgspec.Struct, frozen=True, kw_only=True):
    """One measured candidate at one declared boundary."""

    record_id: str
    candidate: str
    boundary: str
    cache_state: str
    params: dict[str, Any]
    median_s: float
    mean_s: float
    # Quartiles of marker's timing samples (each sample is itself a mean over
    # the in-graph loop in graph mode). The spread is decision input — a
    # winner claim needs disjoint spreads, not just ordered medians.
    p25_s: float
    p75_s: float
    replicate_s: tuple[float, ...]
    memory_footprint_bytes: int | None
    bandwidth_gib_s: float | None
    graph_replay: bool
    device_name: str
    source_revision: str
    # Set by `pair_with_base`; the matched base-only denominator for this
    # measurement, measured in the same session at the same boundary.
    base_record_id: str | None = None
    ratio_to_base: float | None = None


class TimingSuite(msgspec.Struct, kw_only=True):
    """A session's records plus the provenance needed to re-read them."""

    suite: str
    device_name: str
    source_revision: str
    torch_version: str
    host: str
    records: list[TimingRecord] = []

    def add(self, record: TimingRecord) -> TimingRecord:
        self.records.append(record)
        return record


def resolve_source_revision() -> str:
    """Best-effort git description of the tree that produced a measurement."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if head.returncode != 0:
        return "unknown"
    suffix = "-dirty" if dirty.stdout.strip() else ""
    return head.stdout.strip() + suffix


def new_suite(suite: str, *, source_revision: str | None = None) -> TimingSuite:
    return TimingSuite(
        suite=suite,
        device_name=torch.cuda.get_device_name(),
        source_revision=source_revision or resolve_source_revision(),
        torch_version=str(torch.__version__),
        host=platform.node(),
    )


def _record_id(candidate: str, boundary: str, cache_state: str, params: dict) -> str:
    digest_source = msgspec.json.encode(
        {
            "candidate": candidate,
            "boundary": boundary,
            "cache_state": cache_state,
            "params": {key: params[key] for key in sorted(params)},
        }
    )
    return hashlib.sha256(digest_source).hexdigest()[:16]


def measure(
    fn: Callable[[], Any],
    *,
    suite: TimingSuite,
    candidate: str,
    boundary: str,
    params: dict[str, Any],
    graph_replay: bool = True,
    memory_footprint_bytes: int | None = None,
    warmup_iters: int = 50,
    replay_iters: int = 1000,
) -> TimingRecord:
    """Time one zero-argument thunk and append a record to ``suite``.

    ``fn`` takes no arguments and writes into buffers the caller owns, which is
    how every kernel in this stack is shaped.  Because marker computes a memory
    footprint by RE-INVOKING the function and measuring its return value, that
    path is disabled here and the caller declares
    ``memory_footprint_bytes`` explicitly instead — the kernels return ``None``
    and a second invocation would be both wrong and wasteful.
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary {boundary!r}; expected one of {BOUNDARIES}")
    cache_state = CACHE_L2_HOT_GRAPH if graph_replay else CACHE_L2_FLUSHED_EAGER

    result: BenchResult = do_bench(
        fn,
        use_cuda_graph=graph_replay,
        warmup_iters=warmup_iters,
        replay_iters=replay_iters,
        metrics=(0.5, "avg", 0.25, 0.75),
        # Zero-arg closures give marker nothing to rotate, so these flags are
        # inert either way; None documents that no rotation happens.
        graph_clone_args=None,
        graph_clone_kwargs=None,
        disable_log_bandwidth=True,
        memory_output=None,
        memory_args=None,
    )
    median_s, mean_s, p25_s, p75_s = result.times[:4]
    bandwidth = None
    if memory_footprint_bytes is not None and median_s > 0:
        bandwidth = memory_footprint_bytes / (1024**3) / median_s

    return suite.add(
        TimingRecord(
            record_id=_record_id(candidate, boundary, cache_state, params),
            candidate=candidate,
            boundary=boundary,
            cache_state=cache_state,
            params=params,
            median_s=median_s,
            mean_s=mean_s,
            p25_s=p25_s,
            p75_s=p75_s,
            replicate_s=tuple(result.times),
            memory_footprint_bytes=memory_footprint_bytes,
            bandwidth_gib_s=bandwidth,
            graph_replay=graph_replay,
            device_name=suite.device_name,
            source_revision=suite.source_revision,
        )
    )


def pair_with_base(record: TimingRecord, base: TimingRecord) -> TimingRecord:
    """Attach a matched base-only denominator measured in the same session.

    Plan section 14 requires base-only controls measured at the SAME boundary,
    in the same run — a denominator carried over from an earlier session has a
    different clock state and a different build.  Returns a new record; the
    caller replaces the entry it holds.
    """
    if record.boundary != base.boundary:
        raise ValueError(
            "cannot pair measurements taken at different boundaries: "
            f"{record.boundary!r} vs {base.boundary!r}"
        )
    if record.cache_state != base.cache_state:
        raise ValueError(
            "cannot pair measurements taken in different cache states: "
            f"{record.cache_state!r} vs {base.cache_state!r}"
        )
    return msgspec.structs.replace(
        record,
        base_record_id=base.record_id,
        ratio_to_base=record.median_s / base.median_s if base.median_s > 0 else None,
    )


def write_suite(suite: TimingSuite, path: str) -> str:
    """Serialize a suite and return the SHA256 of the bytes written."""
    payload = msgspec.json.format(msgspec.json.encode(suite), indent=2)
    with open(path, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()
