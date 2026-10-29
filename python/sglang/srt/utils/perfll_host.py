"""PERF-LL host-loop helpers for small decode batches (all off by default).

SGLANG_PERFLL_HOST_PIN_MAIN: the scheduler's main thread gets two cores of its own at
the end of its NUMA node's CPU set (a pair per GPU on that node); every other thread of
the process moves off the node's reserved cores. Threads started later inherit the
main thread's pair, so the idle loop moves them off again.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_main_cores: frozenset = frozenset()
_other_cores: frozenset = frozenset()
_last_scan = 0.0


def pin_main_thread(gpu_id: int) -> None:
    global _main_cores, _other_cores
    allowed = sorted(os.sched_getaffinity(0))
    # NUMA-bound schedulers share a node two GPUs at a time; an unbound one sees all CPUs.
    slot = gpu_id % 2 if len(allowed) <= 96 else gpu_id
    per_node = 2 if len(allowed) <= 96 else 4
    if len(allowed) < 2 * per_node + 8:
        logger.warning("PERF-LL host: PIN_MAIN skipped, %d CPUs allowed", len(allowed))
        return
    reserved = allowed[-2 * per_node :]
    _main_cores = frozenset(allowed[-2 * slot - 2 : len(allowed) - 2 * slot])
    _other_cores = frozenset(allowed) - frozenset(reserved)
    os.sched_setaffinity(0, _main_cores)
    moved = _move_other_threads()
    logger.info(
        "PERF-LL host: PIN_MAIN gpu %d main thread on CPUs %s, %d other threads on %d CPUs",
        gpu_id,
        sorted(_main_cores),
        moved,
        len(_other_cores),
    )


def _move_other_threads() -> int:
    me = threading.get_native_id()
    moved = 0
    for tid in os.listdir("/proc/self/task"):
        tid = int(tid)
        if tid == me:
            continue
        try:
            if os.sched_getaffinity(tid) != _other_cores:
                os.sched_setaffinity(tid, _other_cores)
                moved += 1
        except OSError:
            pass
    return moved


def maybe_repin() -> None:
    """Call only while the scheduler is idle (at most every 5 s)."""
    global _last_scan
    if not _main_cores:
        return
    now = time.monotonic()
    if now - _last_scan < 5.0:
        return
    _last_scan = now
    _move_other_threads()


WAIT_EVENTS = envs.SGLANG_PERFLL_HOST_WAIT_EVENTS.get()
_join_events: dict = {}
_copy_done_ring: list = []
_copy_done_pos = 0


def wait_stream(dst, src) -> None:
    """dst.wait_stream(src): the same join, through a cached event for the pair (a
    re-record never changes a wait already enqueued)."""
    if not WAIT_EVENTS:
        dst.wait_stream(src)
        return
    key = (dst.cuda_stream, src.cuda_stream)
    event = _join_events.get(key)
    if event is None:
        event = _join_events[key] = torch.cuda.Event()
    event.record(src)
    dst.wait_event(event)


OUT_POOL = envs.SGLANG_PERFLL_HOST_OUT_POOL.get()
# A pooled tensor is handed out again 8 requests of its shape later; the overlap loop's
# readers of a round's index outputs finish within two rounds.
_RING_DEPTH = 8
_rings: dict = {}


def torch_empty(shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def round_empty(shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    """torch.empty for a decode round's index outputs (every element written by the
    producing kernel, read only within the round), from a ring per (shape, dtype, device)."""
    key = (shape, dtype, device)
    ring = _rings.get(key)
    if ring is None:
        ring = _rings[key] = [
            [torch.empty(shape, dtype=dtype, device=device) for _ in range(_RING_DEPTH)],
            0,
        ]
    pos = ring[1]
    ring[1] = pos + 1 if pos + 1 < _RING_DEPTH else 0
    return ring[0][pos]


def copy_done_event(device_module):
    """A fresh event, or with WAIT_EVENTS the next of a ring of 4: a result's event is
    synchronized one pass after its record, long before the ring comes round."""
    global _copy_done_pos
    if not WAIT_EVENTS:
        return device_module.Event()
    if not _copy_done_ring:
        _copy_done_ring.extend(device_module.Event() for _ in range(4))
    event = _copy_done_ring[_copy_done_pos]
    _copy_done_pos = (_copy_done_pos + 1) % 4
    return event
