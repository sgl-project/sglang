"""Env-gated per-cycle wall-clock profiler for the EAGLE3 verify cycle.

Enabled by ``SGLANG_DEBUG_SPEC_CYCLE=1`` (off by default -> zero overhead).
Splits each speculative cycle into four wall-clock phases plus the verify
batch size (to attribute slow cycles to large-bs vs prefill-interleaving):

- ``bs``     : target-verify batch size (requests in the verify forward).
- ``draft``  : draft-model execution (all ``speculative_num_steps`` steps).
- ``gap1``   : draft_end -> verify_start. Host/scheduler gap: tree-mask build,
               draft-result handling, verify-batch setup, scheduler dispatch.
- ``verify`` : the N-token target-verify forward (N = speculative_num_steps+1).
- ``overhead``: verify_end -> next draft_start (accept / sample / schedule).

``cycle = draft + gap1 + verify + overhead``. Host/serialization/graph-replay
gaps are invisible to Ascend ``op_statistic`` (which only sums kernel time);
this measures them directly. ``torch.<device>.synchronize()`` barriers perturb
absolute TPOT slightly but preserve a valid **relative** phase split.

Each cycle's line is emitted at the *next* cycle's draft entry, because
``overhead`` (verify_end -> next draft_start) is only known once the following
draft begins.
"""

import logging
import time
from contextlib import contextmanager
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Single-threaded scheduler path; module-global scratch is intentional.
_pending_draft_ms: Optional[float] = None
_pending_gap1_ms: Optional[float] = None
_pending_verify_ms: Optional[float] = None
_pending_bs: Optional[int] = None
_prev_verify_end: Optional[float] = None
_draft_end_ts: Optional[float] = None
_cycle_ct: int = 0


def enabled() -> bool:
    return bool(envs.SGLANG_DEBUG_SPEC_CYCLE.get())


def _dev_sync() -> None:
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _emit_cycle(now: float) -> None:
    """Emit the just-finished cycle. Called at each draft-phase entry."""
    global _cycle_ct
    if _pending_draft_ms is None or _pending_verify_ms is None:
        return  # incomplete cycle (e.g. draft without a following verify)
    draft_ms = _pending_draft_ms
    gap1_ms = _pending_gap1_ms if _pending_gap1_ms is not None else float("nan")
    verify_ms = _pending_verify_ms
    overhead_ms = (now - _prev_verify_end) * 1000.0 if _prev_verify_end else float("nan")
    cycle_ms = draft_ms + gap1_ms + verify_ms + overhead_ms
    _cycle_ct += 1
    logger.info(
        "[spec-cycle] ct=%d bs=%s draft=%.2fms gap1=%.2fms verify=%.2fms overhead=%.2fms cycle=%.2fms",
        _cycle_ct,
        _pending_bs,
        draft_ms,
        gap1_ms,
        verify_ms,
        overhead_ms,
        cycle_ms,
    )


@contextmanager
def draft_phase():
    """Time the draft-execution block; flush the previous cycle's line first."""
    global _pending_draft_ms, _pending_verify_ms, _pending_gap1_ms, _pending_bs
    now = time.perf_counter()
    _emit_cycle(now)
    _pending_draft_ms = None
    _pending_verify_ms = None
    _pending_gap1_ms = None
    _pending_bs = None
    _dev_sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _dev_sync()
        _pending_draft_ms = (time.perf_counter() - t0) * 1000.0
        _draft_end_ts = time.perf_counter()  # for gap1 = verify_entry - draft_end


@contextmanager
def verify_phase(forward_batch):
    """Time the target-verify forward; record bs; gap1 = entry - prev draft_end."""
    global _pending_verify_ms, _prev_verify_end, _pending_gap1_ms, _pending_bs
    _pending_bs = forward_batch.batch_size
    # gap1 first (host wall-clock draft_end -> verify_entry), before any sync.
    _pending_gap1_ms = (
        (time.perf_counter() - _draft_end_ts) * 1000.0
        if _draft_end_ts is not None
        else float("nan")
    )
    _dev_sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _dev_sync()
        _pending_verify_ms = (time.perf_counter() - t0) * 1000.0
        _prev_verify_end = time.perf_counter()
