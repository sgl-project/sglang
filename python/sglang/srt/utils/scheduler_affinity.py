"""Scheduler main-thread CPU affinity (SGLANG_OPT_SCHEDULER_MAIN_THREAD_AFFINITY).

The scheduler's main thread gets two cores of its own at the end of its NUMA node's
CPU set (a pair per GPU on that node); every other thread of the process moves off
the node's reserved cores. Threads started later inherit the main thread's pair, so
the idle loop moves them off again.
"""

from __future__ import annotations

import logging
import os
import threading
import time

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
        logger.warning(
            "Scheduler main-thread affinity skipped, %d CPUs allowed", len(allowed)
        )
        return
    reserved = allowed[-2 * per_node :]
    _main_cores = frozenset(allowed[-2 * slot - 2 : len(allowed) - 2 * slot])
    _other_cores = frozenset(allowed) - frozenset(reserved)
    os.sched_setaffinity(0, _main_cores)
    moved = _move_other_threads()
    logger.info(
        "Scheduler main thread of gpu %d on CPUs %s, %d other threads on %d CPUs",
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
            # The thread exited between the listing and the call.
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
