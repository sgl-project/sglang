"""Runtime device-memory accounting for the serving loop.

sgl-project/sglang#46831 reports a ROCm DCP-PD deployment aborting with
``HSA_STATUS_ERROR_OUT_OF_RESOURCES`` while the PyTorch allocator's *reserved*
memory grew ~15.7 GiB against ~0.8 GiB of live allocations, with the driver's
own free-memory reading falling from 16.36 GiB to 0.92 GiB a couple of seconds
before the abort.  Investigating that needs visibility the engine does not have
today:

* the split between *live* allocations and the *cached* blocks the allocator
  holds but has not handed back to the driver,
* the driver's free-memory reading -- the number the HIP runtime aborts on, and
  the only one that also reflects memory taken outside the Torch allocator,
* the non-Torch footprint, i.e. the memory the driver has handed out that Torch
  does not account for at all (HIP workspaces, other libraries, other
  processes),
* the batch shape that coincided with the growth, and
* what an allocator-cache release costs in wall-clock time, measured before it
  is turned into a serving policy.

This module provides that visibility, plus an opt-in policy that releases the
allocator cache on *every* rank when a device free-memory floor is crossed.

Why per-rank and not rank-0-only: the engine's only periodic allocator-cache
release today is the one carried by ``IdleSleeper``, and the sleeper is created
only where ``pp_rank == 0 and attn_tp_rank == 0 and attn_cp_rank == 0`` because
the sockets it polls exist only there.  The cache release therefore rides along
on rank 0 and every other rank keeps its cached blocks for the lifetime of the
process.  #46831 measured that asymmetry directly: a rank-0-only release
(their control ``S2k``) restored TP0 to 10.7 GiB free while the other decode
ranks stayed near full and aborted.

Note that all-rank release is *not* a free win.  The reporter's own all-rank
hunk was never measured in isolation -- their ``S2n``/``S2m`` runs "also
retained diagnostics and the earlier TP0 hook" -- so this module ships the
mechanism and the measurements, and leaves the policy off by default rather
than carrying an unverified workaround forward.

Everything here is opt-in.  The defaults keep the engine's current behavior:
no periodic logging, no periodic release, no snapshots.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.log_utils import create_log_targets, log_json

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

# The subset of ``torch.cuda.memory_stats()`` that is worth logging every
# iteration. The full stats dict is large and mostly static.
_MEMORY_STAT_KEYS = (
    "allocated_bytes.all.current",
    "allocated_bytes.all.peak",
    "reserved_bytes.all.current",
    "reserved_bytes.all.peak",
    "active_bytes.all.current",
    "active_bytes.all.peak",
    "inactive_split_bytes.all.current",
    "inactive_split_bytes.all.peak",
    "requested_bytes.all.peak",
    "num_alloc_retries",
    "num_ooms",
    "num_device_alloc",
    "num_device_free",
)


@dataclass(frozen=True)
class DeviceMemorySample:
    """One reading of the device's memory state on this rank.

    ``allocated_bytes``/``reserved_bytes`` come from the Torch caching
    allocator; ``free_bytes``/``total_bytes`` come from the driver via
    ``mem_get_info``. The two views answer different questions: Torch's
    ``reserved`` counts what the allocator has taken from the driver, while
    ``free_bytes`` is what the driver would let the next allocation have.
    The HIP runtime aborts on the latter.
    """

    allocated_bytes: int
    reserved_bytes: int
    free_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def cached_bytes(self) -> int:
        """Bytes the allocator holds but is not using."""
        return max(self.reserved_bytes - self.allocated_bytes, 0)

    @property
    def unaccounted_bytes(self) -> Optional[int]:
        """Driver-allocated memory that the Torch allocator does not own.

        ``total - free`` is everything the driver has handed out; subtracting
        Torch's ``reserved`` leaves the footprint of everything *outside* the
        Torch allocator -- HIP/ROCm runtime workspaces, FlashInfer or
        aiter buffers, other libraries, and any other process on the device.

        This is the number that separates "the allocator cached a lot" from
        "something that is not Torch took the memory". #46831 could not
        attribute its 15.7 GiB growth to either, because the engine logged
        neither.
        """
        if self.free_bytes is None or self.total_bytes is None:
            return None
        return max(self.total_bytes - self.free_bytes - self.reserved_bytes, 0)

    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "cached_bytes": self.cached_bytes,
        }
        if self.free_bytes is not None:
            data["free_bytes"] = self.free_bytes
        if self.total_bytes is not None:
            data["total_bytes"] = self.total_bytes
        unaccounted = self.unaccounted_bytes
        if unaccounted is not None:
            data["unaccounted_bytes"] = unaccounted
        data["stats"] = self.stats
        return data


def read_device_memory() -> DeviceMemorySample:
    """Read the current device memory state.

    Never raises: a driver that cannot answer is reported as an absent
    reading rather than taking the serving loop down with it.
    """
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())

    free_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    try:
        free, total = torch.cuda.mem_get_info()
        free_bytes, total_bytes = int(free), int(total)
    except Exception as e:  # pragma: no cover - depends on the driver
        logger.debug("mem_get_info() failed: %s", e)

    stats: Dict[str, Any] = {}
    try:
        raw = torch.cuda.memory_stats()
    except Exception as e:  # pragma: no cover - depends on the build
        logger.debug("memory_stats() failed: %s", e)
        raw = {}
    for key in _MEMORY_STAT_KEYS:
        value = raw.get(key)
        if value is not None:
            stats[key] = value

    return DeviceMemorySample(
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        stats=stats,
    )


def read_free_bytes() -> Optional[int]:
    """Read only the driver's free-memory counter."""
    try:
        free, _ = torch.cuda.mem_get_info()
        return int(free)
    except Exception as e:  # pragma: no cover - depends on the driver
        logger.debug("mem_get_info() failed: %s", e)
        return None


def release_allocator_cache() -> None:
    """Return the allocator's cached blocks to the driver."""
    from sglang.srt.platforms import current_platform

    current_platform.empty_cache()


def _default_rank_info() -> Dict[str, Any]:
    """Best-effort identification of this process inside the parallelism."""
    info: Dict[str, Any] = {}
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            info["global_rank"] = dist.get_rank()
    except Exception:  # pragma: no cover - dist is optional here
        pass

    try:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        for name in ("tp_rank", "pp_rank", "attn_tp_rank", "attn_cp_rank", "dcp_rank"):
            try:
                info[name] = getattr(parallel, name)
            except Exception:
                continue
    except Exception:  # pragma: no cover - parallel context is optional here
        pass

    return info


class DeviceMemoryMonitor:
    """Per-rank device-memory accounting and an opt-in cache-release policy.

    The monitor is driven by decode iterations: every decode forward pass is
    one iteration, which is the unit the reporter's own workaround used
    ("every eight decode iterations").

    It lives in the model runner rather than in the scheduler because the
    scheduler process *is* the TP worker process for its rank (one process per
    (pp_rank, tp_rank)), so a per-process check is inherently per-rank and
    needs no collective to stay consistent across a TP group.
    """

    def __init__(
        self,
        *,
        rank_info: Optional[Dict[str, Any]] = None,
        log_interval: int = 0,
        release_interval: int = 8,
        free_threshold_bytes: Optional[int] = None,
        snapshot_dir: Optional[str] = None,
        memory_reader: Callable[[], DeviceMemorySample] = read_device_memory,
        free_reader: Callable[[], Optional[int]] = read_free_bytes,
        cache_release: Callable[[], None] = release_allocator_cache,
        loggers: Optional[List[logging.Logger]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.rank_info = rank_info if rank_info is not None else _default_rank_info()
        self.log_interval = int(log_interval)
        self.release_interval = max(int(release_interval), 1)
        self.free_threshold_bytes = free_threshold_bytes
        self.snapshot_dir = snapshot_dir
        self.memory_reader = memory_reader
        self.free_reader = free_reader
        self.cache_release = cache_release
        self.loggers = (
            loggers
            if loggers is not None
            else create_log_targets(targets=[], name_prefix=__name__)
        )
        self.clock = clock

        self.decode_iterations = 0
        self.release_count = 0
        self.total_release_ms = 0.0
        self.max_release_ms = 0.0
        self._snapshot_written = False

    @staticmethod
    def maybe_create(
        rank_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[DeviceMemoryMonitor]:
        """Build a monitor from the environment, or ``None`` if it is off.

        Off means: no periodic logging, no periodic release and no snapshots,
        which is the engine's current behavior.
        """
        log_interval = envs.SGLANG_DEVICE_MEMORY_LOG_INTERVAL.get()
        threshold_mb = envs.SGLANG_DEVICE_MEMORY_FREE_THRESHOLD_MB.get()
        release_interval = envs.SGLANG_DEVICE_MEMORY_RELEASE_INTERVAL.get()
        snapshot_dir = envs.SGLANG_DEVICE_MEMORY_SNAPSHOT_DIR.get()
        log_target = envs.SGLANG_DEVICE_MEMORY_LOG_TARGET.get()

        if log_interval <= 0 and threshold_mb <= 0 and not snapshot_dir:
            return None

        if not torch.cuda.is_available():
            logger.warning(
                "Device memory monitoring is requested but no CUDA/ROCm device is "
                "available; the monitor stays disabled."
            )
            return None

        resolved_rank_info = _default_rank_info()
        if rank_info:
            resolved_rank_info.update(rank_info)

        targets = [t.strip() for t in log_target.split(",") if t.strip()]
        return DeviceMemoryMonitor(
            rank_info=resolved_rank_info,
            log_interval=log_interval,
            release_interval=release_interval,
            free_threshold_bytes=int(threshold_mb * MIB) if threshold_mb > 0 else None,
            snapshot_dir=snapshot_dir or None,
            loggers=create_log_targets(targets=targets, name_prefix=__name__),
        )

    def observe(
        self,
        *,
        is_decode: bool,
        batch_size: int = 0,
        num_tokens: int = 0,
        forward_mode: str = "",
        can_run_graph: bool = False,
    ) -> None:
        """Record one forward pass.

        Only decode passes count: the allocator growth in #46831 tracked the
        decode loop, and the release policy must not fire on a prefill that
        legitimately needs its blocks.
        """
        if not is_decode:
            return
        self.decode_iterations += 1
        iteration = self.decode_iterations
        shape = {
            "batch_size": batch_size,
            "num_tokens": num_tokens,
            "forward_mode": forward_mode,
            "can_run_graph": can_run_graph,
        }
        if self._should_check_release(iteration):
            self._maybe_release(iteration, shape)
        if self._should_log(iteration):
            self._log_sample(iteration, shape)

    def _should_check_release(self, iteration: int) -> bool:
        if self.free_threshold_bytes is None:
            return False
        return iteration % self.release_interval == 0

    def _should_log(self, iteration: int) -> bool:
        if self.log_interval <= 0:
            return False
        return iteration % self.log_interval == 0

    def _maybe_release(self, iteration: int, shape: Dict[str, Any]) -> None:
        free_bytes = self.free_reader()
        if free_bytes is None or free_bytes >= self.free_threshold_bytes:
            return
        before = self._read_sample()
        self._write_snapshot("before_release")
        start = self.clock()
        self.cache_release()
        pause_ms = (self.clock() - start) * 1000.0
        self.release_count += 1
        self.total_release_ms += pause_ms
        self.max_release_ms = max(self.max_release_ms, pause_ms)
        after = self._read_sample()
        self._write_snapshot("after_release")
        log_json(
            self.loggers,
            "device_memory.release",
            {
                **self.rank_info,
                "decode_iterations": iteration,
                "free_threshold_bytes": self.free_threshold_bytes,
                "free_bytes_before": free_bytes,
                "pause_ms": round(pause_ms, 3),
                "release_count": self.release_count,
                "total_release_ms": round(self.total_release_ms, 3),
                "max_release_ms": round(self.max_release_ms, 3),
                "before": before.to_json() if before else None,
                "after": after.to_json() if after else None,
                **shape,
            },
        )

    def _log_sample(self, iteration: int, shape: Dict[str, Any]) -> None:
        sample = self._read_sample()
        if sample is None:
            return
        log_json(
            self.loggers,
            "device_memory.sample",
            {
                **self.rank_info,
                "decode_iterations": iteration,
                "release_count": self.release_count,
                "total_release_ms": round(self.total_release_ms, 3),
                **sample.to_json(),
                **shape,
            },
        )

    def _read_sample(self) -> Optional[DeviceMemorySample]:
        try:
            return self.memory_reader()
        except Exception as e:  # pragma: no cover - depends on the driver
            logger.warning("Failed to read device memory: %s", e)
            return None

    def _write_snapshot(self, reason: str) -> Optional[str]:
        """Dump the allocator's segment/blocks table to a JSON file.

        The snapshot is taken together with a ``mem_get_info`` reading so the
        two can be compared: the segment table explains Torch's own usage, the
        free-memory reading exposes everything else on the device.
        """
        if not self.snapshot_dir:
            return None
        if reason == "baseline" and self._snapshot_written:
            return None
        try:
            snapshot = torch.cuda.memory_snapshot()
        except Exception as e:  # pragma: no cover - depends on the build
            logger.warning("Failed to capture an allocator snapshot: %s", e)
            return None
        try:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            stamp = int(time.time() * 1000)
            path = os.path.join(
                self.snapshot_dir, f"rank_snapshot_{reason}_{stamp}.json"
            )
            free_bytes = read_free_bytes()
            payload = {
                "reason": reason,
                "timestamp": time.time(),
                "rank_info": self.rank_info,
                "free_bytes": free_bytes,
                "allocator_snapshot": snapshot,
            }
            with open(path, "w") as f:
                json.dump(payload, f, default=str)
        except Exception as e:  # pragma: no cover - filesystem dependent
            logger.warning("Failed to write the allocator snapshot: %s", e)
            return None
        if reason == "baseline":
            self._snapshot_written = True
        return path

    def write_baseline_snapshot(self) -> Optional[str]:
        """Capture the allocator state before the serving loop starts."""
        return self._write_snapshot("baseline")
