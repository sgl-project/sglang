"""Opt-in diagnostics for NPU pinned host allocations.

Set SGLANG_NPU_PINNED_HOST_DEBUG=1 to log allocator and host memory state at
the SGLang call sites that have produced aclrtMallocHostWithCfg failures.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_INTERVAL_SECONDS = 30
_LARGE_ALLOCATION_BYTES = 1 << 30
_MONITOR_INTERVAL_SECONDS = 5
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


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _percent(used: int | None, limit: int | None) -> float | None:
    if used is None or limit is None or limit <= 0:
        return None
    return round(100 * used / limit, 2)


def _meminfo_bytes(key: str) -> int | None:
    value = _read_proc_value("/proc/meminfo", key)
    if value is None:
        return None
    try:
        amount, unit = value.split()
        return int(amount) * 1024 if unit == "kB" else None
    except ValueError:
        return None


def _cgroup_v2_directory(root: Path) -> Path:
    """Locate this process's cgroup when the mount exposes the full tree."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                candidate = root / line[3:].lstrip("/")
                if (candidate / "memory.current").exists():
                    return candidate
    except OSError:
        pass
    # With a private cgroup namespace the mount root is already this process's
    # cgroup, although /proc/self/cgroup may show a host-relative path.
    return root


def _cgroup_v2_memory(
    root: Path,
) -> tuple[int | None, int | None, Path, str | None]:
    directory = _cgroup_v2_directory(root)
    selected = (_read_int(directory / "memory.current"), None, directory, None)
    smallest_headroom: int | None = None
    while True:
        current = _read_int(directory / "memory.current")
        for limit_name in ("memory.max", "memory.high"):
            limit = _read_int(directory / limit_name)
            if current is not None and limit is not None:
                headroom = limit - current
                if smallest_headroom is None or headroom < smallest_headroom:
                    selected = (current, limit, directory, limit_name)
                    smallest_headroom = headroom
        if directory == root:
            break
        directory = directory.parent
    return selected


def _read_keyed_ints(path: Path) -> dict[str, int]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    values = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            try:
                values[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return values


def _numa_node_memory() -> dict[str, dict[str, int]]:
    nodes = {}
    for path in Path("/sys/devices/system/node").glob("node[0-9]*/meminfo"):
        values = {}
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            match = re.search(r"\b(MemTotal|MemFree|MemUsed):\s+(\d+)\s+kB", line)
            if match:
                values[f"{match[1].lower()}_bytes"] = int(match[2]) * 1024
        if values:
            nodes[path.parent.name] = values
    return nodes


@lru_cache(maxsize=1)
def _boot_id_hash() -> str | None:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None
    return hashlib.sha256(boot_id.encode()).hexdigest()[:16] if boot_id else None


@lru_cache(maxsize=1)
def _numa_bind_policies() -> list[str]:
    """Best-effort view of mapped memory policies, not a process-wide guarantee."""
    policies = set()
    try:
        with open("/proc/self/numa_maps") as file:
            for line in file:
                parts = line.split(maxsplit=2)
                if len(parts) > 1 and parts[1].startswith("bind:"):
                    policies.add(parts[1])
    except OSError:
        pass
    return sorted(policies)


def _host_and_cgroup_snapshot() -> dict[str, object]:
    """Keep host-wide and cgroup usage separate; their denominators differ."""
    host_total = _meminfo_bytes("MemTotal")
    host_available = _meminfo_bytes("MemAvailable")
    cgroup = Path("/sys/fs/cgroup")
    current, limit, cgroup, limit_kind = _cgroup_v2_memory(cgroup)
    if current is None and limit is None:
        # Common cgroup v1 mount; omit percentages when the mount is unavailable.
        cgroup = cgroup / "memory"
        current = _read_int(cgroup / "memory.usage_in_bytes")
        limit = _read_int(cgroup / "memory.limit_in_bytes")
        if limit is not None and limit >= 1 << 60:
            limit = None
        limit_kind = "memory.limit_in_bytes" if limit is not None else None
    hard_limit = (
        limit
        if limit_kind == "memory.limit_in_bytes"
        else _read_int(cgroup / "memory.max")
    )

    cgroup_headroom = (
        max(0, limit - current) if current is not None and limit is not None else None
    )
    headrooms = [
        value for value in (host_available, cgroup_headroom) if value is not None
    ]
    effective_headroom = min(headrooms) if headrooms else None
    cgroup_events = _read_keyed_ints(cgroup / "memory.events")
    cgroup_stat = _read_keyed_ints(cgroup / "memory.stat")

    snapshot = {
        "host_memory_total_bytes": host_total,
        "host_memory_available_bytes": host_available,
        "host_memory_used_pct": _percent(
            host_total - host_available
            if host_total is not None and host_available is not None
            else None,
            host_total,
        ),
        "cgroup_memory_current_bytes": current,
        "cgroup_memory_max_bytes": hard_limit,
        "cgroup_memory_used_pct": _percent(current, hard_limit),
        "cgroup_memory_effective_limit_bytes": limit,
        "cgroup_memory_effective_used_pct": _percent(current, limit),
        "cgroup_memory_peak_bytes": _read_int(cgroup / "memory.peak"),
        "cgroup_memory_high_bytes": _read_int(cgroup / "memory.high"),
        "cgroup_memory_limit_path": str(cgroup) if limit is not None else None,
        "cgroup_memory_limit_kind": limit_kind,
        "cgroup_memory_events": {
            key: cgroup_events.get(key) for key in ("high", "max", "oom", "oom_kill")
        },
        "cgroup_memory_anon_bytes": cgroup_stat.get("anon"),
        "cgroup_memory_file_bytes": cgroup_stat.get("file"),
        "cgroup_memory_unevictable_bytes": cgroup_stat.get("unevictable"),
        "effective_host_memory_headroom_bytes": effective_headroom,
        "effective_host_memory_limiter": (
            "cgroup"
            if cgroup_headroom is not None
            and (host_available is None or cgroup_headroom <= host_available)
            else "host"
            if host_available is not None
            else None
        ),
        "numa_node_memory": _numa_node_memory(),
        "numa_mems_allowed_list": _read_proc_value(
            "/proc/self/status", "Mems_allowed_list"
        ),
        "numa_bind_policies": _numa_bind_policies(),
    }
    if cgroup_headroom is not None:
        snapshot["cgroup_memory_headroom_bytes"] = cgroup_headroom
    return snapshot


def _memory_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "cgroup_memory_current": _read_cgroup_value("memory.current"),
        "cgroup_memory_max": _read_cgroup_value("memory.max"),
        "cgroup_memory_peak": _read_cgroup_value("memory.peak"),
        "mem_available": _read_proc_value("/proc/meminfo", "MemAvailable"),
        "process_rss": _read_proc_value("/proc/self/status", "VmRSS"),
        "process_locked": _read_proc_value("/proc/self/status", "VmLck"),
        "ci_run_id": os.environ.get("GITHUB_RUN_ID"),
        "ci_job": os.environ.get("GITHUB_JOB"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "container_hostname": os.environ.get("HOSTNAME"),
        "host_boot_id_hash": _boot_id_hash(),
    }
    try:
        snapshot.update(_host_and_cgroup_snapshot())
    except Exception as exc:  # noqa: BLE001
        snapshot["host_memory_stats_error"] = repr(exc)
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
        snapshot["pinned_reserved_pct_of_cgroup_limit"] = _percent(
            snapshot["pinned_reserved_bytes"], snapshot.get("cgroup_memory_max_bytes")
        )
        snapshot["pinned_reserved_pct_of_host_total"] = _percent(
            snapshot["pinned_reserved_bytes"], snapshot.get("host_memory_total_bytes")
        )
    except Exception as exc:  # noqa: BLE001
        # Diagnostics must never turn a successful allocation into a failure.
        snapshot["pinned_stats_error"] = repr(exc)
    return snapshot


def log_npu_host_baseline(event: str) -> None:
    """Record runner pressure before or after a test without initializing NPU."""
    try:
        snapshot = _host_and_cgroup_snapshot()
    except Exception as exc:  # noqa: BLE001
        snapshot = {"host_memory_stats_error": repr(exc)}
    logger.warning(
        "NPU host baseline: %s",
        {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "ci_run_id": os.environ.get("GITHUB_RUN_ID"),
            "ci_job": os.environ.get("GITHUB_JOB"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "container_hostname": os.environ.get("HOSTNAME"),
            "host_boot_id_hash": _boot_id_hash(),
        }
        | snapshot,
    )


class PinnedHostMemoryMonitor:
    """Sample scheduler Host memory through model loading and request handling.

    torch.npu.memory_stats() describes device allocations. The separate
    torch_npu.npu.host_memory_stats() counters describe pinned Host allocations.
    Its native peak counters also catch spikes between periodic samples.
    """

    def __init__(self, *, enabled: bool):
        self.enabled = enabled and npu_pinned_host_debug_enabled()
        self.phase = "loading"
        self.loading_active_peak_bytes: int | None = None
        self.loading_reserved_peak_bytes: int | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _emit(self, event: str) -> None:
        with self._lock:
            snapshot = _memory_snapshot()
            active_peak = snapshot.get("pinned_active_peak_bytes")
            reserved_peak = snapshot.get("pinned_reserved_peak_bytes")
            if self.phase == "runtime":
                if (
                    isinstance(active_peak, int)
                    and self.loading_active_peak_bytes is not None
                ):
                    snapshot["active_peak_phase"] = (
                        "runtime"
                        if active_peak > self.loading_active_peak_bytes
                        else "loading_or_equal"
                    )
                if (
                    isinstance(reserved_peak, int)
                    and self.loading_reserved_peak_bytes is not None
                ):
                    snapshot["reserved_peak_phase"] = (
                        "runtime"
                        if reserved_peak > self.loading_reserved_peak_bytes
                        else "loading_or_equal"
                    )
            logger.warning(
                "NPU pinned host monitor: %s",
                {
                    "event": event,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "pid": os.getpid(),
                    "rank": os.environ.get("RANK"),
                    "local_rank": os.environ.get("LOCAL_RANK"),
                    "phase": self.phase,
                    "loading_pinned_active_peak_bytes": self.loading_active_peak_bytes,
                    "loading_pinned_reserved_peak_bytes": (
                        self.loading_reserved_peak_bytes
                    ),
                }
                | snapshot,
            )

    def start(self) -> None:
        if not self.enabled:
            return
        self._emit("start")
        self._thread = threading.Thread(
            target=self._run, name="npu-pinned-host-monitor", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(_MONITOR_INTERVAL_SECONDS):
            self._emit("sample")

    def mark_runtime(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            snapshot = _memory_snapshot()
            self.loading_active_peak_bytes = snapshot.get("pinned_active_peak_bytes")
            self.loading_reserved_peak_bytes = snapshot.get(
                "pinned_reserved_peak_bytes"
            )
            self.phase = "runtime"
        self._emit("runtime_start")

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._emit("stop")


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
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
