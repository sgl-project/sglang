"""Crash-safe kernel isolation — subprocess per CANDIDATE (not per tactic).

A kernel that hits "no kernel image available" or a C++ ``std::unordered_map::at()``
abort corrupts the CUDA context; you cannot reliably try/except and keep benchmarking
other candidates in the same process (the FlashInfer #2701/#2695 crashes were fixed by
revert, not caught in-process). So each surviving candidate is benchmarked in its own
spawned subprocess. Its job is to protect the *tuning run's completeness* — a narrower
bar than protecting a live server, which never runs measurement at all (bake-only ingest).

Grain: one subprocess per backend candidate, NOT per tactic — the per-tactic loop keeps
FlashInfer's in-process try/except + cudaGetLastError inside each subprocess. Cost is
~3-5 s spawn/CUDA-cold-start per candidate (so ~15-25 s for 5); a warm-worker-pool is the
speed/isolation tradeoff for v2.

Failure taxonomy the parent classifies (never crashes on):
  ImportError            -> wheel not built for this arch  -> backend unavailable
  "no kernel image"      -> prebuilt kernel wrong SM       -> infeasible on this device
  torch.cuda.OutOfMemory -> shape too big at this config   -> skip shape, keep smaller
  timeout / signal       -> hang / pathological config     -> mark broken

Every classified failure becomes a structured note in the emitted config, so the corpus
is self-documenting — the opposite of vLLM's per-scenario skip-lists that "just hid it."
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import queue as _queue
from typing import Dict, List, Optional

from .harness import mock_decode_latency, mock_prefill_latency
from .shapes import AttnProfile, parse_decode_key, parse_prefill_key

# failure kinds
IMPORT_ERROR = "import_error"
NO_KERNEL_IMAGE = "no_kernel_image"
OOM = "oom"
TIMEOUT = "timeout"
CRASH = "crash"


@dataclasses.dataclass
class CandidateResult:
    backend: str
    latencies: Dict[str, float]  # bucket_key -> median_us (may be PARTIAL on failure)
    failure: Optional[str] = None  # one of the failure kinds, else None
    # per-shape skips (e.g. a bucket whose working set OOMs on this GPU) — the shape is
    # dropped and the sweep continues, per the documented taxonomy "skip shape, keep smaller"
    skipped_shapes: Dict[str, str] = dataclasses.field(default_factory=dict)


def classify_exception(exc: BaseException) -> str:
    msg = str(exc).lower()
    if isinstance(exc, ImportError):
        return IMPORT_ERROR
    if "no kernel image" in msg:
        return NO_KERNEL_IMAGE
    if "out of memory" in msg or exc.__class__.__name__ == "OutOfMemoryError":
        return OOM
    return CRASH


def classify_exit(exitcode: Optional[int]) -> str:
    # negative exitcode == killed by signal (SIGSEGV=-11, SIGABRT=-6) -> context-corrupting crash
    if exitcode is None:
        return TIMEOUT
    if exitcode < 0:
        return CRASH
    return CRASH


def _worker(
    backend: str,
    phase: str,
    bucket_keys: List[str],
    profile: AttnProfile,
    bandwidth_divergent: bool,
    mock: bool,
    out: mp.Queue,
) -> None:
    """Runs in the child process. In mock mode, computes synthetic latencies; a backend
    whose name starts with 'crash' simulates an uncatchable abort (SIGABRT)."""
    try:
        if mock and backend.startswith("crash"):
            # Simulate the uncatchable child death with SIGKILL, not os.abort(): the
            # parent-side contract is identical (signal death, no result -> CRASH), but
            # SIGABRT would trip the macOS crash reporter ("Python quit unexpectedly")
            # on every test run. The live validator's __abort__ below keeps a true SIGABRT.
            import os
            import signal

            os.kill(os.getpid(), signal.SIGKILL)
        if mock and backend.startswith("oom"):
            raise RuntimeError("CUDA out of memory")
        if mock and backend.startswith("nokernel"):
            raise RuntimeError(
                "no kernel image is available for execution on the device"
            )
        # --- real-hardware deliberate failures, for the live crash-recovery validation ---
        if not mock and backend == "__oom__":  # pragma: no cover - real GPU only
            from .realbench import force_oom

            force_oom()
        if not mock and backend == "__abort__":  # pragma: no cover - real GPU only
            import os

            os.abort()
        if (
            not mock and backend == "__incompatible__"
        ):  # pragma: no cover - real GPU only
            from .realbench import force_incompatible

            force_incompatible()

        skipped_shapes: Dict[str, str] = {}
        for key in bucket_keys:
            try:
                if phase == "decode":
                    sh = parse_decode_key(key)
                    us = (
                        mock_decode_latency(backend, sh, profile, bandwidth_divergent)
                        if mock
                        else _real_bench_decode(backend, sh, profile)
                    )  # pragma: no cover
                else:
                    sh = parse_prefill_key(key)
                    us = (
                        mock_prefill_latency(backend, sh, profile)
                        if mock
                        else _real_bench_prefill(backend, sh, profile)
                    )  # pragma: no cover
            except BaseException as e:  # noqa: BLE001
                if classify_exception(e) == OOM:
                    # "OOM -> shape too big at this config -> skip shape, keep smaller":
                    # a per-shape skip, NOT a candidate-level failure. Empty the CUDA
                    # cache so the failed allocation doesn't poison the next bucket.
                    skipped_shapes[key] = OOM
                    if not mock:  # pragma: no cover - real GPU only
                        import torch

                        torch.cuda.empty_cache()
                    continue
                raise  # import / no-kernel-image / crash: candidate-level
            out.put(
                ("bucket", key, us)
            )  # stream: a timeout later never loses this bucket
        out.put(("done", skipped_shapes))
    except BaseException as e:  # noqa: BLE001 - we want to report every failure kind
        out.put(("err", classify_exception(e)))


def _real_bench_decode(backend, sh, profile):  # pragma: no cover - real GPU only
    from .realbench import real_decode_latency

    return real_decode_latency(backend, sh, profile)


def _real_bench_prefill(backend, sh, profile):  # pragma: no cover - real GPU only
    from .realbench import real_prefill_latency

    return real_prefill_latency(backend, sh, profile)


class _Fold:
    """Accumulates the worker's streamed messages into a CandidateResult."""

    def __init__(self):
        self.lat: Dict[str, float] = {}
        self.shape_skips: Dict[str, str] = {}
        self.failure: Optional[str] = None
        self.finished = False  # saw "done" or "err" — the stream is complete

    def feed(self, msg) -> None:
        if msg[0] == "bucket":
            self.lat[msg[1]] = msg[2]
        elif msg[0] == "done":
            self.shape_skips = msg[1]
            self.finished = True
        else:  # ("err", kind) — candidate-level failure
            self.failure = msg[1]
            self.finished = True

    def result(self, backend: str) -> CandidateResult:
        return CandidateResult(backend, self.lat, self.failure, self.shape_skips)


def run_candidate_isolated(
    backend: str,
    phase: str,
    bucket_keys: List[str],
    profile: AttnProfile,
    bandwidth_divergent: bool,
    mock: bool = True,
    timeout_s: float = 120.0,
    isolate: bool = True,
) -> CandidateResult:
    """Benchmark one candidate. ``isolate=True`` spawns a subprocess (crash-safe);
    ``isolate=False`` runs in-process (fast path for deterministic mock tests).

    Per-bucket results are STREAMED from the child, so a timeout or crash partway
    through the grid keeps every completed bucket (``latencies`` is partial and
    ``failure`` says why the sweep stopped) instead of discarding the candidate.
    """
    if not isolate:
        q: mp.Queue = _InlineQueue()
        _worker(backend, phase, bucket_keys, profile, bandwidth_divergent, mock, q)
        fold = _Fold()
        for msg in q.drain():
            fold.feed(msg)
        return fold.result(backend)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_worker,
        args=(backend, phase, bucket_keys, profile, bandwidth_divergent, mock, q),
    )
    p.start()

    import time

    fold = _Fold()
    deadline = time.monotonic() + timeout_s
    child_gone = False
    while not fold.finished and time.monotonic() < deadline:
        try:
            fold.feed(q.get(timeout=0.2))
        except _queue.Empty:
            if not p.is_alive():  # child died (possibly aborted) mid-stream
                child_gone = True
                break

    if not fold.finished and child_gone:
        # Final drain: mp.Queue delivery (feeder thread) can lag process exit, so a fast
        # child's tail messages could otherwise be lost and the run misclassified.
        drain_until = time.monotonic() + 1.0
        while not fold.finished and time.monotonic() < drain_until:
            try:
                fold.feed(q.get(timeout=0.2))
            except _queue.Empty:
                break
    p.join(timeout=5)

    if not fold.finished:
        if p.is_alive():  # ran out of budget — keep the partial grid
            p.terminate()
            p.join()
            fold.failure = TIMEOUT
        else:  # died mid-stream (SIGABRT/SIGSEGV => crash)
            fold.failure = classify_exit(p.exitcode)
    return fold.result(backend)


class _InlineQueue:
    """Tiny stand-in so the in-process path shares the worker code path."""

    def __init__(self):
        self._items = []

    def put(self, x):
        self._items.append(x)

    def drain(self):
        items, self._items = self._items, []
        return items
