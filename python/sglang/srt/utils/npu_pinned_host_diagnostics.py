"""Opt-in diagnostics for NPU pinned host allocations.

Set SGLANG_NPU_PINNED_HOST_DEBUG=1 to log allocator and host memory state at
the SGLang call sites that have produced aclrtMallocHostWithCfg failures.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_LOG_INTERVAL_SECONDS = 30
_LARGE_ALLOCATION_BYTES = 1 << 30
_last_log_time: dict[str, float] = {}
_call_counts: dict[str, int] = {}


def npu_pinned_host_debug_enabled() -> bool:
    return os.environ.get("SGLANG_NPU_PINNED_HOST_DEBUG") == "1"


def _read_cgroup_value(name: str) -> str | None:
    try:
        with open(f"/sys/fs/cgroup/{name}") as file:
            return file.read().strip()
    except OSError:
        return None


def _read_proc_value(path: str, key: str) -> str | None:
    try:
        with open(path) as file:
            for line in file:
                if line.startswith(f"{key}:"):
                    return line.partition(":")[2].strip()
    except OSError:
        pass
    return None


def _memory_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "cgroup_memory_current": _read_cgroup_value("memory.current"),
        "cgroup_memory_max": _read_cgroup_value("memory.max"),
        "cgroup_memory_peak": _read_cgroup_value("memory.peak"),
        "mem_available": _read_proc_value("/proc/meminfo", "MemAvailable"),
        "process_rss": _read_proc_value("/proc/self/status", "VmRSS"),
        "process_locked": _read_proc_value("/proc/self/status", "VmLck"),
    }
    try:
        import torch_npu

        stats = torch_npu.npu.host_memory_stats()
        snapshot.update(
            {
                "pinned_active_bytes": stats.get("active_bytes.current"),
                "pinned_reserved_bytes": stats.get("allocated_bytes.current"),
                "pinned_active_peak_bytes": stats.get("active_bytes.peak"),
                "pinned_reserved_peak_bytes": stats.get("allocated_bytes.peak"),
                "pinned_active_requests": stats.get("active_requests.current"),
                "pinned_owned_blocks": stats.get("allocations.current"),
                "pinned_host_alloc_calls": stats.get("num_host_alloc"),
                "pinned_host_free_calls": stats.get("num_host_free"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        # Diagnostics must never turn a successful allocation into a failure.
        snapshot["pinned_stats_error"] = repr(exc)
    return snapshot


@contextmanager
def trace_npu_pinned_host_allocation(
    site: str,
    *,
    requested_bytes: int | None = None,
    enabled: bool = True,
    details: dict[str, object] | None = None,
) -> Iterator[None]:
    """Log a sampled before/after snapshot, and always log allocation failures.

    Sampling keeps per-token D2H traces manageable; allocations >= 1 GiB are
    always logged. No torch_npu API is called unless the opt-in flag is set.
    """
    if not enabled or not npu_pinned_host_debug_enabled():
        yield
        return

    started = time.monotonic()
    count = _call_counts[site] = _call_counts.get(site, 0) + 1
    sampled = (
        requested_bytes is not None and requested_bytes >= _LARGE_ALLOCATION_BYTES
    ) or started - _last_log_time.get(site, float("-inf")) >= _LOG_INTERVAL_SECONDS
    context = {
        "site": site,
        "pid": os.getpid(),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "call_count": count,
        "requested_bytes": requested_bytes,
    }
    if details:
        context.update(details)
    if sampled:
        _last_log_time[site] = started
        logger.warning(
            "NPU pinned host allocation before: %s", context | _memory_snapshot()
        )

    try:
        yield
    except Exception:
        logger.exception(
            "NPU pinned host allocation failed: %s", context | _memory_snapshot()
        )
        raise
    else:
        if sampled:
            logger.warning(
                "NPU pinned host allocation after: %s",
                context
                | {"elapsed_seconds": round(time.monotonic() - started, 3)}
                | _memory_snapshot(),
            )
