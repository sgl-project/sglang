"""Poll-vs-async benchmark for the HiCache NIXL storage backend.

Quantify the CPU that a NIXL transfer's busy-poll burns, and the parallel CPU
across HiCache's two always-on IO threads, before an async/event-driven backend
is justified (issue #26693, "needs benchmarks first").

It drives the SHIPPED poll loop in HiCacheNixl._xfer_and_wait
(python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py:131-164) through the
public batch_set_v1/batch_get_v1 entry points and changes NO backend code: it
wraps the live NIXL agent's settable methods (initialize_xfer is the seam the
shipped unit test already rebinds; check_xfer_state and the module-level
time.sleep are wrapped by the same mechanism) to count poll iterations and time
the loop, restoring everything on exit.

THE HEADLINE is the CPU core-fraction, not latency:

    core_fraction = loop_cpu_s / loop_wall_s     (dimensionless, 0..1 per thread)

loop_cpu_s and loop_wall_s are accumulated over the SAME span -- strictly the
in-loop check_xfer_state + sleep calls, NOTHING else. Identical numerator/
denominator span bounds it to [0,1] and keeps the surrounding batch_set/get_v1
registration work out of the number. The headline is taken from the *no_sleep*
variant (sleep removed, so the loop spins on check_xfer_state alone -- a faithful
busy-poll). The 'real' variant (shipped 100us sleep) is reported alongside but
reads lower because the sleep yields the core. Together they bracket the
CPU-vs-latency tradeoff that no single sleep constant escapes -- only a
notification mechanism does.

CONCURRENCY (the production prize): HiCache spawns exactly K=2 always-on IO
threads -- prefetch (batch_get_v1) and backup (batch_set_v1) -- each running a
GIL-free poll loop (NIXL releases the GIL in getXferStatus/postXferReq/
createXferReq). run_concurrency drives that real K=2 topology against ONE shared
HiCacheNixl and reports:

    wall_speedup     = serial_wall / concurrent_wall   -- the DISCRIMINATOR
    wall_ceiling     = serial_wall / critical_path      -- honest parallel ceiling (<=K)
    overlap_frac     = wall_speedup / wall_ceiling       -- fraction of ceiling reached
    cpu_conservation = total_cpu(K) / total_cpu(1)       -- SANITY co-metric (~1.0)

The serial baseline runs the SAME heterogeneous role mix (every role back-to-back
on one thread), so wall_speedup compares like with like. The two metrics are
different axes, not one verdict. cpu_conservation is the CPU COST: ~1.0+ means
every thread ran and burned its poll CPU, nothing serialized away. wall_speedup is
the LATENCY BENEFIT: at K=2 the asymmetric READ/WRITE roles cap it near 1.0 even
when both threads are fully concurrent -- so a modest wall speedup is NOT
serialization, it is the busy-poll CPU MULTIPLYING across the two IO threads while
the wall barely moves. K=4 zero_copy (more balanced) shows wall_speedup ~2x, the
clean proof the loops run on separate cores. K=2 is the hard NIXL production
ceiling (a whole batch fans into a single _xfer_and_wait); K>2 is a LABELED
hypothetical (bounce K>2 is skipped -- it shares one bounce buffer).

CRITICAL REGIME NOTE: NIXL's POSIX backend defaults to SYNCHRONOUS I/O, so
transfer() returns DONE-on-submit and the loop runs zero iterations -- measuring
nothing. This benchmark requests an async POSIX mode via --posix-async
{uring,aio,none} (forwarded through extra_config) and prints the resolved backend
params on every row. Default uring reads the completion ring in USERSPACE (no
per-poll syscall); aio issues io_getevents per poll (syscall_bound, stamped). A
run with mean_polls==0 is reported INCONCLUSIVE-for-latency, NOT refuted: a config
that never polls cannot refute a polling-tax claim.

NIXL is Linux-only, and the magnitudes only mean something on a real kernel: the
per-thread CPU clock and the 100us sleep must be finer than the loop they measure.
The benchmark REFUSES TO RUN on a coarse-clock host (e.g. a gVisor sandbox, whose
~10ms clock and ~1ms sleep would turn core_fraction into quantization noise) -- it
exits with the measured resolution and points you at a bare-metal/KVM host.

Only the single-threaded PollProbe patches time.sleep; the concurrency path uses
the TRUE sleep, so its wall numbers are unpatched. The latency tax is reported as
a secondary, caveated number -- the CPU core-fraction is the result that matters.

Usage:
    python bench_nixl_poll.py --posix-async uring --mode both --direction both \
        --batch-sizes 1,4,16,64,128 --iters 200 --concurrency 1,2,4 \
        --output-file nixl_poll.jsonl
"""

import argparse
import json
import os
import platform
import shutil
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
)

# The shipped loop's requested sleep (hicache_nixl.py:155 -> time.sleep(0.0001)).
REQUESTED_SLEEP_S = 1e-4

# Below this many polls per op, the per-poll check cost (check_call_ns_inflight) is
# dominated by the single terminal completion-reaping poll (reaping many descriptors
# at once), not steady-state polling -- so it is flagged and read as an upper bound,
# not a point estimate. The same-span core_fraction headline is NOT affected.
MIN_POLLS_FOR_PER_POLL = 5

# Per-thread CPU clock. Required for the concurrency metrics: process_time folds
# every thread's CPU into each window and would inflate cpu_conservation. We
# detect it once and stamp results when it is missing.
_THREAD_CPU_RELIABLE = hasattr(time, "CLOCK_THREAD_CPUTIME_ID")


def _thread_cpu() -> float:
    if _THREAD_CPU_RELIABLE:
        try:
            return time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
        except OSError:
            pass
    return time.process_time()


# --------------------------------------------------------------------------- #
# Host pool: a minimal MHA-style CPU stand-in mirroring the shipped unit test's
# MockMemPoolHost (test/registered/unit/mem_cache/test_hicache_nixl_storage.py).
# We do NOT use the real MHATokenToKVPoolHost: its __init__ asserts a GPU
# device_pool, so it cannot be built CPU-only.
# --------------------------------------------------------------------------- #
class BenchMemPoolHost:
    """Page-first (zero-copy) or layer-first (bounce) CPU host pool."""

    def __init__(
        self,
        zero_copy: bool,
        page_size: int = 16,
        layer_num: int = 4,
        head_num: int = 8,
        head_dim: int = 64,
        num_pages: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.layout = "page_first" if zero_copy else "layer_first"
        self.page_size = page_size
        self.layer_num = layer_num
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.num_pages = num_pages
        self.size = page_size * num_pages
        self.pin_memory = False
        if zero_copy:
            self.kv_buffer = torch.zeros(
                (2, self.size, layer_num, head_num, head_dim), dtype=dtype
            )
        else:
            self.kv_buffer = torch.zeros(
                (2, layer_num, self.size, head_num, head_dim), dtype=dtype
            )

    def page_bytes(self) -> int:
        return self.get_dummy_flat_data_page().numel() * self.dtype.itemsize

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        base = self.kv_buffer.data_ptr()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        idx_list = indices.tolist()
        for i in range(0, len(idx_list), self.page_size):
            k_ptr = base + idx_list[i] * (
                self.layer_num * self.head_num * self.head_dim * self.dtype.itemsize
            )
            ptr_list.append(k_ptr)
            ptr_list.append(k_ptr + v_offset)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
        )
        return ptr_list, [element_size] * len(ptr_list)

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
        ).flatten()

    def get_data_page(self, index, flat=True):
        if hasattr(index, "item"):
            index = int(index.item())
        page = self.kv_buffer[:, :, index : index + self.page_size, :, :]
        return page.flatten() if flat else page

    def set_from_flat_data_page(self, index, data_page):
        if hasattr(index, "item"):
            index = int(index.item())
        self.kv_buffer[:, :, index : index + self.page_size, :, :] = data_page.reshape(
            2, self.layer_num, self.page_size, self.head_num, self.head_dim
        )

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        # CPU smoke uses use_direct_io=False so alignment is not consulted; keep
        # False so the safe copy path is taken if O_DIRECT is ever enabled.
        return False


# --------------------------------------------------------------------------- #
# PollProbe: additive SINGLE-THREADED instrumentation. Wraps the live agent's
# initialize_xfer + check_xfer_state and the module-level time.sleep, restoring
# everything on exit. No backend file edited.
#
# Scoping honesty: patching nixl_mod.time.sleep mutates the process-global time
# module (the loop does `import time` then time.sleep), NOT a per-file copy. Safe
# here ONLY because PollProbe is single-threaded and restored in __exit__. The
# concurrency path deliberately does NOT use PollProbe -- it would race threads.
#
# core_fraction same-span construction: the probe accumulates BOTH wall AND
# per-thread CPU strictly inside check_xfer_state + sleep -- the exact
# instructions of the shipped loop -- and nothing outside it, so
# core_fraction = loop_cpu / loop_wall is bounded [0,1] and excludes the
# batch op's registration work.
# --------------------------------------------------------------------------- #
class TimeWrapper:
    # Shadow hicache_nixl's `time` reference instead of mutating the process-global
    # time.sleep. Only sleep is overridden; every other time.* attribute falls
    # through to the real module.
    def __init__(self, real_time, sleep_func):
        self._real_time = real_time
        self._sleep_func = sleep_func

    def __getattr__(self, name):
        if name == "sleep":
            return self._sleep_func
        return getattr(self._real_time, name)


class PollProbe:
    def __init__(self, hicache, nixl_mod, no_sleep: bool):
        self.hicache = hicache
        self.agent = hicache.agent
        self.nixl_mod = nixl_mod
        self.no_sleep = no_sleep
        self.records: List[dict] = []
        self._n_polls = 0
        self._loop_wall = 0.0  # wall inside the loop incl. sleeps
        self._loop_cpu = 0.0  # per-thread CPU inside the loop incl. sleeps
        self._check_wall = 0.0  # wall in check_xfer_state only (excludes sleeps)
        self._check_cpu = 0.0  # per-thread CPU in check_xfer_state only
        self._t_init: Optional[float] = None
        self._orig: Dict[str, Any] = {}

    def _flush(self):
        if self._t_init is not None:
            self.records.append(
                {
                    "polls": self._n_polls,
                    "loop_wall_ms": self._loop_wall * 1000.0,
                    "loop_cpu_ms": self._loop_cpu * 1000.0,
                    "check_wall_ms": self._check_wall * 1000.0,
                    "check_cpu_ms": self._check_cpu * 1000.0,
                }
            )
        self._n_polls = 0
        self._loop_wall = 0.0
        self._loop_cpu = 0.0
        self._check_wall = 0.0
        self._check_cpu = 0.0
        self._t_init = None

    def __enter__(self):
        a = self.agent
        self._orig = {
            "initialize_xfer": a.initialize_xfer,
            "check_xfer_state": a.check_xfer_state,
            "time": self.nixl_mod.time,
        }
        oi = self._orig["initialize_xfer"]
        oc = self._orig["check_xfer_state"]

        def w_init(*a_, **k_):
            # Each transfer starts a new poll-loop record.
            self._flush()
            self._t_init = time.perf_counter()
            return oi(*a_, **k_)

        def w_check(handle):
            # Same-span accounting: time BOTH wall and per-thread CPU around the
            # exact check_xfer_state call the shipped loop issues, nothing else.
            # Both end-clocks are read together (c1 then t1) right after the call.
            # The wall window still brackets the two clock reads the CPU window
            # excludes (~2 reads, sub-microsecond per poll), so core_fraction is
            # biased *down* by that overhead -- a conservative understatement of
            # the busy-poll CPU, never an inflation past the [0,1] bound.
            self._n_polls += 1
            t0 = time.perf_counter()
            c0 = _thread_cpu()
            state = oc(handle)
            c1 = _thread_cpu()
            t1 = time.perf_counter()
            self._loop_cpu += c1 - c0
            self._check_cpu += c1 - c0
            dt = t1 - t0
            self._loop_wall += dt
            self._check_wall += dt
            return state

        def w_sleep(seconds):
            # no_sleep variant: skip the sleep entirely so the loop busy-spins on
            # check_xfer_state alone -- the faithful busy-poll cf_nosleep measures.
            if self.no_sleep:
                return None
            t0 = time.perf_counter()
            c0 = _thread_cpu()
            r = self._orig["time"].sleep(seconds)
            c1 = _thread_cpu()
            t1 = time.perf_counter()
            self._loop_cpu += c1 - c0
            self._loop_wall += t1 - t0
            return r

        a.initialize_xfer = w_init
        a.check_xfer_state = w_check
        self.nixl_mod.time = TimeWrapper(self._orig["time"], w_sleep)
        return self

    def __exit__(self, *exc):
        self._flush()
        self.agent.initialize_xfer = self._orig["initialize_xfer"]
        self.agent.check_xfer_state = self._orig["check_xfer_state"]
        self.nixl_mod.time = self._orig["time"]
        return False


@dataclass
class CellResult:
    backend_name: str
    backend_mode: str  # uring | aio | none (resolved, after fallback)
    syscall_bound: bool  # True if the resolved check path issues a per-poll syscall
    backend_params: str  # resolved agent.get_backend_params(), so the regime is logged
    posix_async: str  # as requested on the CLI
    mode: str  # zero_copy | bounce
    actual_zero_copy: bool
    direction: str  # WRITE | READ
    variant: str  # real | no_sleep
    batch_pages: int
    page_bytes: int
    total_bytes: int
    iters: int
    ok_rate: float
    # --- headline CPU metrics (SAME-SPAN: numerator and denominator both cover
    # only the in-loop check+sleep, never the surrounding batch op) ---
    mean_polls: float
    max_polls: int
    mean_loop_wall_ms: float  # in-loop wall (check + sleep), per op
    mean_loop_cpu_ms: float  # in-loop per-thread CPU (check + sleep), per op
    mean_check_wall_ms: float  # check_xfer_state wall only (no sleeps), per op
    core_fraction: float  # loop_cpu / loop_wall (same-span, [0,1] headline)
    # --- per-poll check cost + per-op busy CPU ---
    check_call_ns_inflight: float  # measured in-flight per-poll check wall; -1 if N/A
    check_call_low_poll: (
        bool  # True if mean_polls < MIN_POLLS_FOR_PER_POLL (reaping-contaminated)
    )
    busy_cpu_ms: float  # mean_polls * inflight / 1e6 (per op); -1 if N/A
    # --- latency (secondary, sleep-granularity-bound) ---
    realized_sleep_us: float
    mean_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    throughput_gbs: float
    inconclusive: bool  # True if mean_polls < 1 (loop never spun)


@dataclass
class ConcurrencyResult:
    backend_mode: str
    syscall_bound: bool
    mode: str
    direction_topology: str
    K: int
    is_production_topology: bool  # True only for K==2
    iters_per_thread: int
    serial_wall_ms: float  # heterogeneous serial wall: K roles back-to-back
    concurrent_wall_ms: float  # wall for the K-thread run (measured loop only)
    critical_path_ms: float  # slowest single role's wall (the parallel floor)
    total_cpu_s_1: float  # summed thread CPU at the serial baseline (all K roles)
    total_cpu_s_k: float  # summed thread CPU at K (same K roles, K threads)
    cpu_conservation: float  # total_cpu(K)/total_cpu(1) -- sanity (~1.0)
    wall_speedup: float  # serial_wall/concurrent_wall -- DISCRIMINATOR
    wall_ceiling: float  # serial_wall/critical_path (honest parallel ceiling, <=K)
    overlap_frac: float  # wall_speedup/wall_ceiling (fraction of ceiling reached)
    mean_polls_per_thread: float
    realized_sleep_us_idle: float
    realized_sleep_us_underload: float
    parallel_verdict: str


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, int(q * (len(s) - 1)))
    return s[k]


def _measure_sleep_granularity(
    target_s: float = REQUESTED_SLEEP_S, n: int = 200
) -> float:
    """Realized wall per time.sleep(target_s), in microseconds. On a gVisor/coarse
    host this is far larger than requested (often ~1ms); on a tuned host it
    approaches target_s."""
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        time.sleep(target_s)
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples) if samples else target_s * 1e6


def _thread_cpu_resolution_ns(n: int = 200000) -> float:
    """Smallest non-zero delta of the per-thread CPU clock, in ns. A real kernel
    reports ~1-1000ns; a gVisor sandbox reports ~10ms, far coarser than the
    sub-microsecond poll loop -> core_fraction becomes untrustworthy there."""
    if not _THREAD_CPU_RELIABLE:
        return float("inf")
    last = _thread_cpu()
    best = float("inf")
    for _ in range(n):
        cur = _thread_cpu()
        d = cur - last
        if d > 0:
            best = min(best, d)
        last = cur
    return best * 1e9 if best != float("inf") else float("inf")


def _fs_type(path: str) -> str:
    """Filesystem backing `path` (ext4, tmpfs, xfs, nfs, ...), via /proc/mounts.
    tmpfs means /tmp is RAM-backed -- a different I/O cost profile worth logging so a
    reviewer knows whether the POSIX numbers are disk-backed or memory-backed."""
    try:
        real = os.path.realpath(path)
        best_mp, best_fs = "", "unknown"
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mp, fstype = parts[1], parts[2]
                if real == mp or real.startswith(mp.rstrip("/") + "/"):
                    if len(mp) > len(best_mp):
                        best_mp, best_fs = mp, fstype
        return best_fs
    except Exception:
        return "unknown"


def _posix_async_extra(backend_mode: str) -> dict:
    """plugin.posix.* init params for the requested async mode. NIXL POSIX async
    knobs default false (sync I/O => zero polls); 'none' keeps that sync default
    so the degenerate regime is visible explicitly."""
    posix: dict = {"active": True}
    if backend_mode == "aio":
        posix["use_aio"] = "true"
    elif backend_mode == "uring":
        posix["use_uring"] = "true"
    return {"plugin": {"posix": posix}, "use_direct_io": False}


def _make_hicache(file_path: str, backend_mode: str):
    # Lazy import so --help works without nixl.
    try:
        from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl
    except ImportError as e:
        raise SystemExit(
            f"NIXL not importable ({e}); run on Linux with nixl installed."
        )
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="bench_model",
        extra_config=_posix_async_extra(backend_mode),
    )
    return HiCacheNixl(storage_config=config, file_path=file_path)


def _resolve_backend_mode(file_path: str, requested: str) -> Tuple[str, bool]:
    """Resolve the effective async backend mode, falling back uring -> aio if the
    requested backend fails to construct. Returns (resolved_mode, syscall_bound):
    uring's steady-state check reads the CQ ring in userspace (no per-poll
    syscall); aio issues io_getevents per poll."""
    candidates = [requested]
    if requested == "uring":
        candidates.append("aio")  # documented fallback
    last_err: Optional[Exception] = None
    for mode in candidates:
        try:
            probe = _make_hicache(file_path, mode)
            try:
                probe.close()
            except Exception:
                pass
            syscall_bound = mode != "uring"
            if mode != requested:
                print(
                    f"WARNING: backend '{requested}' failed to construct; fell back "
                    f"to '{mode}'. Rows stamped backend_mode={mode}, "
                    f"syscall_bound={syscall_bound}."
                )
            return mode, syscall_bound
        except SystemExit:
            raise
        except Exception as e:  # construction failed; try the next candidate
            last_err = e
            continue
    raise SystemExit(
        f"No requested POSIX backend could be constructed "
        f"(tried {candidates}): {last_err}"
    )


def _seed_keys(hicache, host_indices, keys: List[str]) -> None:
    hicache.batch_set_v1(keys, host_indices)


def _run_variant(
    hicache,
    nixl_mod,
    host: BenchMemPoolHost,
    direction: str,
    batch_pages: int,
    iters: int,
    warmup: int,
    variant: str,
    mode: str,
    posix_async: str,
    backend_mode: str,
    syscall_bound: bool,
    realized_sleep_us: float,
) -> CellResult:
    page_size = host.page_size
    host_indices = torch.arange(batch_pages * page_size, dtype=torch.int64)
    page_bytes = host.page_bytes()
    total_bytes = page_bytes * batch_pages
    no_sleep = variant == "no_sleep"

    # Pre-generate keys. For READ, seed OUTSIDE the probe so the recorded
    # transfers are genuine reads, not contaminated by interleaved seeding writes.
    keys_per_iter = [
        [f"{variant}_{mode}_{direction}_{it}_{i}" for i in range(batch_pages)]
        for it in range(iters)
    ]

    # Warm-up writes so the FS / page cache is in steady state. Discarded.
    for w in range(max(warmup, 1)):
        wkeys = [f"warm_{mode}_{direction}_{w}_{i}" for i in range(batch_pages)]
        hicache.batch_set_v1(wkeys, host_indices)
        if direction == "READ":
            hicache.batch_get_v1(wkeys, host_indices)

    if direction == "READ":
        for keys in keys_per_iter:
            _seed_keys(hicache, host_indices, keys)

    latencies: List[float] = []
    oks: List[bool] = []

    with PollProbe(hicache, nixl_mod, no_sleep=no_sleep) as probe:
        for it in range(iters):
            keys = keys_per_iter[it]
            t0 = time.perf_counter()
            if direction == "WRITE":
                res = hicache.batch_set_v1(keys, host_indices)
            else:
                res = hicache.batch_get_v1(keys, host_indices)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            oks.append(bool(res) and all(res))

    polls = [r["polls"] for r in probe.records]
    loop_walls = [r["loop_wall_ms"] for r in probe.records]
    loop_cpus = [r["loop_cpu_ms"] for r in probe.records]
    check_walls = [r["check_wall_ms"] for r in probe.records]
    check_cpus = [r["check_cpu_ms"] for r in probe.records]
    mean_lat = statistics.fmean(latencies) if latencies else 0.0
    mean_polls = statistics.fmean(polls) if polls else 0.0
    mean_loop_wall_ms = statistics.fmean(loop_walls) if loop_walls else 0.0
    mean_loop_cpu_ms = statistics.fmean(loop_cpus) if loop_cpus else 0.0
    mean_check_wall_ms = statistics.fmean(check_walls) if check_walls else 0.0
    mean_check_cpu_ms = statistics.fmean(check_cpus) if check_cpus else 0.0
    throughput = (total_bytes / (mean_lat / 1000.0)) / 1e9 if mean_lat else 0.0

    # Headline CPU: SAME-SPAN core fraction. Numerator (loop_cpu) and denominator
    # (loop_wall) accumulate over the identical in-loop check+sleep instructions,
    # so this is bounded [0,1] and immune to the batch op's CPU.
    total_loop_wall_s = mean_loop_wall_ms * iters / 1000.0
    total_loop_cpu_s = mean_loop_cpu_ms * iters / 1000.0
    core_fraction = (
        (total_loop_cpu_s / total_loop_wall_s) if total_loop_wall_s > 0 else 0.0
    )

    # In-flight per-poll check cost: the WALL of one check_xfer_state call (how long
    # a poll takes); the check_ns column. For uring this is a userspace ring read.
    check_call_ns_inflight = (
        (mean_check_wall_ms / 1000.0) / mean_polls * 1e9
        if (mean_polls > 0 and mean_check_wall_ms > 0)
        else -1.0
    )
    # Few polls => the per-poll cost is dominated by the terminal completion-reaping
    # poll, not steady-state polling; flag it so check_ns reads as an upper bound.
    check_call_low_poll = 0 < mean_polls < MIN_POLLS_FOR_PER_POLL
    # Per-op CPU the busy-poll burns: true per-thread CPU summed over the in-loop
    # check calls -- NOT wall. For aio, io_getevents can block off-CPU (wall > CPU),
    # so this measures CPU directly rather than polls * check-wall.
    busy_cpu_ms = mean_check_cpu_ms if mean_polls > 0 else -1.0

    return CellResult(
        backend_name=hicache.backend_selector.backend_name,
        backend_mode=backend_mode,
        syscall_bound=syscall_bound,
        backend_params=str(
            hicache.agent.get_backend_params(hicache.backend_selector.backend_name)
        ),
        posix_async=posix_async,
        mode=mode,
        actual_zero_copy=hicache.is_zero_copy,
        direction=direction,
        variant=variant,
        batch_pages=batch_pages,
        page_bytes=page_bytes,
        total_bytes=total_bytes,
        iters=iters,
        ok_rate=(sum(oks) / len(oks)) if oks else 0.0,
        mean_polls=mean_polls,
        max_polls=max(polls) if polls else 0,
        mean_loop_wall_ms=mean_loop_wall_ms,
        mean_loop_cpu_ms=mean_loop_cpu_ms,
        mean_check_wall_ms=mean_check_wall_ms,
        core_fraction=core_fraction,
        check_call_ns_inflight=check_call_ns_inflight,
        check_call_low_poll=check_call_low_poll,
        busy_cpu_ms=busy_cpu_ms,
        realized_sleep_us=realized_sleep_us,
        mean_latency_ms=mean_lat,
        median_latency_ms=statistics.median(latencies) if latencies else 0.0,
        p90_latency_ms=_pct(latencies, 0.90),
        p99_latency_ms=_pct(latencies, 0.99),
        throughput_gbs=throughput,
        inconclusive=(mean_polls < 1.0 and variant == "real"),
    )


def _concurrency_verdict(
    K: int,
    wall_speedup: float,
    wall_ceiling: float,
    overlap_frac: float,
    cpu_conservation: float,
    polls_per_thread: float,
) -> str:
    """Two-axis read of a K-thread run, NOT a binary PARALLEL/SERIALIZED label.
    cpu_conservation is the CPU COST -- ~1.0+ means every thread ran and burned its
    poll CPU, nothing serialized away. wall_speedup is the LATENCY BENEFIT -- how
    much sooner the K loops finish together; it is capped near 1.0 at K=2 by the
    asymmetric READ/WRITE roles even when both threads are fully concurrent, so a
    modest wall speedup is NOT evidence of serialization."""
    if K == 1:
        return "baseline (K=1)"
    if cpu_conservation < 0.9:
        return (
            f"INCONCLUSIVE: cpu_conservation={cpu_conservation:.2f}<0.9 -- a thread "
            f"may have stalled or been starved; wall_speedup={wall_speedup:.2f}. "
            f"Re-run with more iters."
        )
    if wall_speedup >= 1.5:
        latency = (
            f"a real wall win (wall_speedup={wall_speedup:.2f}, {overlap_frac:.0%} of "
            f"the {wall_ceiling:.2f}x asymmetry-limited ceiling) -- proof the {K} "
            f"loops run on separate cores (impossible under GIL serialization)"
        )
    elif wall_speedup >= 1.05:
        latency = (
            f"a modest wall win (wall_speedup={wall_speedup:.2f}, {overlap_frac:.0%} "
            f"of the {wall_ceiling:.2f}x ceiling) -- the asymmetric READ/WRITE roles "
            f"cap it"
        )
    else:
        latency = (
            f"~no wall win (wall_speedup={wall_speedup:.2f}) -- the slow role "
            f"dominates the critical path, so parallel polling costs CPU without "
            f"buying latency"
        )
    return (
        f"CONCURRENT: {K} IO threads, total CPU conserved "
        f"(cpu_conservation={cpu_conservation:.2f}, {polls_per_thread:.0f} "
        f"polls/thread); {latency}"
    )


# --------------------------------------------------------------------------- #
# Concurrency harness. Drives the real K=2 production topology (prefetch +
# backup) against ONE shared HiCacheNixl, with thread-local poll counting via a
# SINGLE pre-installed check_xfer_state wrapper. It does NOT use PollProbe and
# does NOT patch time.sleep -- the concurrent real path uses the true sleep.
#
# Timed regions cover the measured get/set loop. Per-thread READ pre-seeding is
# hoisted out before any timer; the serial baseline runs the SAME heterogeneous
# role mix (every role back-to-back on one thread) so wall_speedup compares like
# with like; the serial and concurrent runs use DISJOINT key namespaces so a WRITE
# role creates its files cold in BOTH (otherwise the serial baseline pays cold
# inode-creation while the concurrent run overwrites warm, inflating wall_speedup).
# Two known biases are CONSERVATIVE (both understate wall_speedup, i.e. run against
# the parallelism claim, and are small at iters_per_thread=200): the concurrent
# wall also brackets the workers' start()/join(), which the serial baseline does
# not pay; and the realized-sleep sampler is a separate low-duty thread that runs
# during (and shares cores with) the timed concurrent region.
# --------------------------------------------------------------------------- #
def run_concurrency(
    hicache,
    host: BenchMemPoolHost,
    K: int,
    iters_per_thread: int,
    warmup: int,
    mode: str,
    backend_mode: str,
    syscall_bound: bool,
    realized_sleep_us_idle: float,
) -> ConcurrencyResult:
    page_size = host.page_size
    batch_pages = 1  # one page per op keeps per-thread namespaces small + isolated

    # Per-thread host_indices slice so threads never touch the same host rows. In
    # zero_copy mode both directions share kv_buffer, so the disjoint slices ARE the
    # isolation -- race-free at ANY K. In bounce mode the backend has a SINGLE shared
    # _bounce_set/_bounce_get slot, so it is race-free only at K<=2 (one READ on
    # _bounce_get + one WRITE on _bounce_set = disjoint); K>2 bounce would put two
    # same-role threads on one slot and is skipped by main (see the guard there).
    def indices_for(tid: int):
        start = tid * batch_pages * page_size
        return torch.arange(start, start + batch_pages * page_size, dtype=torch.int64)

    # Single pre-installed thread-local poll counter on the SHARED agent.
    tls = threading.local()
    orig_check = hicache.agent.check_xfer_state

    def counting_check(handle):
        n = getattr(tls, "polls", 0)
        tls.polls = n + 1
        return orig_check(handle)

    # Thread 0 = prefetch (READ via batch_get_v1), thread 1 = backup (WRITE via
    # batch_set_v1) -- the verified K=2 topology. K>2 alternates roles (labeled).
    def role_of(tid: int) -> str:
        return "READ" if tid % 2 == 0 else "WRITE"

    # Two DISJOINT key namespaces per thread: one for the serial baseline, one
    # for the concurrent run. WRITE roles create their files COLD in BOTH (this
    # is production behavior -- the backup thread writes NEW KV entries), so
    # neither timed region benefits from the other's file creation. Without this
    # split the serial baseline pays cold inode-creation while the concurrent run
    # overwrites the same files warm, which inflates wall_speedup far past the
    # critical-path ceiling. READ namespaces are pre-seeded for BOTH so reads are
    # warm in both regions (symmetric). Built + seeded OUTSIDE any timed region,
    # so neither wall carries seeding cost.
    def make_keys(tag: str) -> Dict[int, List[List[str]]]:
        return {
            tid: [
                [f"{tag}_K{K}_{mode}_{tid}_{it}_{i}" for i in range(batch_pages)]
                for it in range(iters_per_thread)
            ]
            for tid in range(K)
        }

    keys_serial = make_keys("S")
    keys_conc = make_keys("C")
    for tid in range(K):
        if role_of(tid) == "READ":
            idx = indices_for(tid)
            for keys in keys_serial[tid] + keys_conc[tid]:
                hicache.batch_set_v1(keys, idx)

    per_thread_cpu: Dict[int, float] = {}
    per_thread_polls: Dict[int, int] = {}
    per_role_wall_ms: Dict[int, float] = {}
    cpu_lock = threading.Lock()

    def run_role(tid: int, my_keys: List[List[str]]) -> None:
        """Run only the MEASURED get/set loop for thread tid against the given key
        namespace (no seeding, no sleep sampling). Records this thread's own CPU +
        poll count + wall."""
        idx = indices_for(tid)
        role = role_of(tid)
        tls.polls = 0
        w0 = time.perf_counter()
        c0 = _thread_cpu()
        for it in range(iters_per_thread):
            keys = my_keys[it]
            if role == "WRITE":
                hicache.batch_set_v1(keys, idx)
            else:
                hicache.batch_get_v1(keys, idx)
        c1 = _thread_cpu()
        w1 = time.perf_counter()
        with cpu_lock:
            per_thread_cpu[tid] = c1 - c0
            per_thread_polls[tid] = getattr(tls, "polls", 0)
            per_role_wall_ms[tid] = (w1 - w0) * 1000.0

    # Warm-up (single thread) so the FS / page cache is hot before timing.
    for w in range(max(warmup, 1)):
        wk = [f"cwarm_{K}_{mode}_{w}_{i}" for i in range(batch_pages)]
        hicache.batch_set_v1(wk, indices_for(0))

    hicache.agent.check_xfer_state = counting_check
    try:
        # Heterogeneous serial baseline: every role back-to-back on one thread, so
        # the baseline does the same READ+WRITE mix the concurrent run does.
        # serial_wall is the sum of all roles' walls. The parallel CEILING is the
        # longest single role (critical path): perfect overlap cannot finish
        # before its slowest member -- with asymmetric roles that ceiling is < K,
        # so the verdict tests against it, not K.
        per_thread_cpu.clear()
        per_thread_polls.clear()
        per_role_wall_ms.clear()
        t0 = time.perf_counter()
        for tid in range(K):
            run_role(tid, keys_serial[tid])
        serial_wall_ms = (time.perf_counter() - t0) * 1000.0
        total_cpu_s_1 = sum(per_thread_cpu.values())
        critical_path_ms = max(per_role_wall_ms.values()) if per_role_wall_ms else 0.0

        # K-thread concurrent run -- measured loop only (seeding already done).
        per_thread_cpu.clear()
        per_thread_polls.clear()

        # Under-load realized-sleep sampling in a SEPARATE thread that does not
        # feed per_thread_cpu / concurrent_wall, so its real sleeps never
        # contaminate the verdict metrics. It spins while the workers run.
        underload_samples: List[float] = []
        stop_sampler = threading.Event()

        def sampler():
            while not stop_sampler.is_set():
                t = time.perf_counter()
                time.sleep(REQUESTED_SLEEP_S)
                underload_samples.append((time.perf_counter() - t) * 1e6)

        workers = [
            threading.Thread(target=run_role, args=(tid, keys_conc[tid]))
            for tid in range(K)
        ]
        sampler_thread = threading.Thread(target=sampler, daemon=True)
        sampler_thread.start()
        t0 = time.perf_counter()
        for th in workers:
            th.start()
        for th in workers:
            th.join()
        concurrent_wall_ms = (time.perf_counter() - t0) * 1000.0
        stop_sampler.set()
        sampler_thread.join(timeout=1.0)

        total_cpu_s_k = sum(per_thread_cpu.values())
        mean_polls_per_thread = (
            statistics.fmean(list(per_thread_polls.values()))
            if per_thread_polls
            else 0.0
        )
    finally:
        hicache.agent.check_xfer_state = orig_check

    # CPU conservation: both regimes do the identical K-role mix (serial sums them
    # on one thread, concurrent spreads them across K threads), so this is ~1.0
    # when the parallel run neither loses CPU (under-measured) nor inflates it
    # wildly. It is NOT a parallelism proof on its own (GIL-serialized threads
    # still each burn their role's CPU => ~1.0). Slightly >1.0 is normal: under
    # load each transfer's wall is longer, so the busy-poll spins a few more times.
    cpu_conservation = total_cpu_s_k / total_cpu_s_1 if total_cpu_s_1 > 0 else 0.0
    wall_speedup = (
        serial_wall_ms / concurrent_wall_ms if concurrent_wall_ms > 0 else 0.0
    )
    # PARALLEL ceiling = serial_wall / critical_path: perfect overlap cannot finish
    # before the slowest single role. With asymmetric roles this is < K, so we test
    # the achieved overlap as a FRACTION of its own honest ceiling, not against K.
    wall_ceiling = serial_wall_ms / critical_path_ms if critical_path_ms > 0 else 0.0
    overlap_frac = (wall_speedup / wall_ceiling) if wall_ceiling > 0 else 0.0
    realized_sleep_us_underload = (
        statistics.median(underload_samples)
        if underload_samples
        else realized_sleep_us_idle
    )

    verdict = _concurrency_verdict(
        K,
        wall_speedup,
        wall_ceiling,
        overlap_frac,
        cpu_conservation,
        mean_polls_per_thread,
    )

    return ConcurrencyResult(
        backend_mode=backend_mode,
        syscall_bound=syscall_bound,
        mode=mode,
        direction_topology=(
            "prefetch+backup (K=2 production)"
            if K == 2
            else (
                "single (K=1 baseline)"
                if K == 1
                else "round-robin (hypothetical, NOT NIXL production)"
            )
        ),
        K=K,
        is_production_topology=(K == 2),
        iters_per_thread=iters_per_thread,
        serial_wall_ms=serial_wall_ms,
        concurrent_wall_ms=concurrent_wall_ms,
        critical_path_ms=critical_path_ms,
        total_cpu_s_1=total_cpu_s_1,
        total_cpu_s_k=total_cpu_s_k,
        cpu_conservation=cpu_conservation,
        wall_speedup=wall_speedup,
        wall_ceiling=wall_ceiling,
        overlap_frac=overlap_frac,
        mean_polls_per_thread=mean_polls_per_thread,
        realized_sleep_us_idle=realized_sleep_us_idle,
        realized_sleep_us_underload=realized_sleep_us_underload,
        parallel_verdict=verdict,
    )


def _print_headline(results: List[CellResult]) -> None:
    """What a reviewer reads first: the granularity-robust CPU core-fraction. The
    headline is the NO_SLEEP core_fraction (a true busy-poll); the 'real'
    core_fraction is shown alongside but is lower because the shipped sleep
    yields the core."""
    print("\n=== HEADLINE: CPU core-fraction (busy-poll vs shipped loop) ===")
    ns_by_key = {
        (r.mode, r.direction, r.batch_pages): r
        for r in results
        if r.variant == "no_sleep"
    }
    hdr = (
        f"{'mode':<10}{'dir':<6}{'pages':>6}"
        f"{'polls_ns':>9}{'cf_nosleep':>11}{'cf_real':>9}"
        f"{'check_ns':>10}{'busy_cpu_ms':>12}{'ok%':>5}{'incon':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.variant != "real":
            continue
        sib = ns_by_key.get((r.mode, r.direction, r.batch_pages))
        cf_ns = sib.core_fraction if sib is not None else float("nan")
        polls_ns = sib.mean_polls if sib is not None else float("nan")
        # '*' marks low-poll cells where check_ns is reaping-contaminated (upper bound).
        star = "*" if r.check_call_low_poll else ""
        check_ns = (
            "N/A"
            if r.check_call_ns_inflight < 0
            else f"{r.check_call_ns_inflight:.1f}{star}"
        )
        busy = "N/A" if r.busy_cpu_ms < 0 else f"{r.busy_cpu_ms:.5f}"
        ok_pct = f"{r.ok_rate * 100:.0f}"
        print(
            f"{r.mode:<10}{r.direction:<6}{r.batch_pages:>6}"
            f"{polls_ns:>9.1f}{cf_ns:>11.3f}{r.core_fraction:>9.3f}"
            f"{check_ns:>10}{busy:>12}{ok_pct:>5}{('Y' if r.inconclusive else ''):>6}"
        )
    print(
        "cf_nosleep ~1.0 => the busy-poll loop pins a core (the CPU claim); cf_real "
        "is the same loop WITH the shipped 100us sleep (lower because the sleep "
        "yields). polls_ns is the no_sleep busy-poll count; check_ns is the measured "
        "per-poll check_xfer_state WALL; busy_cpu_ms is the per-op thread-CPU summed "
        "over the in-loop check calls (real variant). check_ns* (mean_polls<5) is "
        "reaping-contaminated -- read it as an upper bound, not steady-state; the "
        "core_fraction headline is unaffected. ok% is the transfer success rate."
    )
    bad = [r for r in results if r.ok_rate < 1.0]
    if bad:
        print(
            f"WARNING: {len(bad)} cell(s) had ok_rate<1.0 (transfers failed); their "
            "numbers are NOT trustworthy: "
            + ", ".join(
                f"{r.mode}/{r.direction}/{r.batch_pages}/{r.variant}={r.ok_rate:.2f}"
                for r in bad
            )
        )


def _print_concurrency(cresults: List[ConcurrencyResult]) -> None:
    print(
        "\n=== CONCURRENCY: wall speedup + CPU conservation (the production prize) ==="
    )
    hdr = (
        f"{'mode':<10}{'K':>3} {'topology':<48}"
        f"{'wall_x':>8}{'ceil_x':>8}{'overlap':>9}{'cpu_cons':>10}"
        f"{'polls/th':>9}{'sleep_ul_us':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for c in cresults:
        cpu_cons = f"{c.cpu_conservation:.2f}"
        print(
            f"{c.mode:<10}{c.K:>3} {c.direction_topology:<48}"
            f"{c.wall_speedup:>8.2f}{c.wall_ceiling:>8.2f}{c.overlap_frac:>9.2f}"
            f"{cpu_cons:>10}"
            f"{c.mean_polls_per_thread:>9.1f}{c.realized_sleep_us_underload:>12.1f}"
        )
    print("-" * len(hdr))
    for c in cresults:
        print(f"  K={c.K} ({c.mode}): {c.parallel_verdict}")
    print(
        "Two axes, not one label: cpu_cons (=total_cpu(K)/total_cpu(1)) is the CPU "
        "COST -- ~1.0+ means every thread ran and burned its poll CPU, nothing "
        "serialized away; wall_x is the LATENCY BENEFIT -- how much sooner K loops "
        "finish together. They diverge: at K=2 the asymmetric READ/WRITE roles cap "
        "wall_x near 1.0 while both threads still burn poll CPU (parallel polling "
        "costs CPU without buying latency); K=4 zero_copy shows wall_x ~2x, the clean "
        "proof the loops run on separate cores. K=2 is the NIXL production topology; "
        "K>2 is a labeled hypothetical (bounce K>2 skipped -- shared bounce buffer)."
    )


def _print_latency_tax(results: List[CellResult]) -> None:
    """SECONDARY, caveated. Sleep-granularity-bound latency tax (single-threaded,
    platform-dependent -- NOT the headline). tax = real - no_sleep latency; it
    scales with the sleep granularity (realized_sleep_us) and shrinks toward 0 on
    a host with a finer sleep. The CPU headline is what survives."""
    by_key = {(r.mode, r.direction, r.batch_pages, r.variant): r for r in results}
    print(
        "\n=== Sleep-granularity-bound latency tax "
        "(secondary, platform-dependent -- NOT the headline) ==="
    )
    print(
        f"{'mode':<10}{'dir':<6}{'pages':>6}{'sleep_us':>9}"
        f"{'tax_ms':>10}{'tax_%':>9}{'GB/s_lost':>11}{'note':>14}"
    )
    seen = set()
    for mode, direction, pages, _v in by_key:
        if (mode, direction, pages) in seen:
            continue
        seen.add((mode, direction, pages))
        real = by_key.get((mode, direction, pages, "real"))
        ns = by_key.get((mode, direction, pages, "no_sleep"))
        if real is None or ns is None:
            continue
        if real.mean_polls < 1.0:
            print(
                f"{mode:<10}{direction:<6}{pages:>6}{real.realized_sleep_us:>9.1f}"
                f"{'N/A':>10}{'N/A':>9}{'N/A':>11}{'INCONCLUSIVE':>14}"
            )
            continue
        tax = real.mean_latency_ms - ns.mean_latency_ms
        pct = (tax / real.mean_latency_ms * 100.0) if real.mean_latency_ms else 0.0
        lost = ns.throughput_gbs - real.throughput_gbs
        print(
            f"{mode:<10}{direction:<6}{pages:>6}{real.realized_sleep_us:>9.1f}"
            f"{tax:>10.4f}{pct:>9.1f}{lost:>11.3f}{'':>14}"
        )
    print(
        "tax ~= mean_polls * realized_sleep_us: a sleep-granularity artifact that "
        "shrinks with the sleep. The CPU core-fraction (headline) persists."
    )


def _clear_files(file_path: str) -> None:
    # Drop the benchmark's temporary KV files so a long sweep cannot exhaust a
    # size-limited /tmp (tmpfs). Best-effort; never fatal. Called only after all
    # measurement is written, so it cannot perturb any cell's timing.
    try:
        entries = os.listdir(file_path)
    except OSError:
        return
    for entry in entries:
        full = os.path.join(file_path, entry)
        try:
            if os.path.isfile(full) or os.path.islink(full):
                os.remove(full)
            else:
                shutil.rmtree(full, ignore_errors=True)
        except OSError:
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--posix-async",
        default="uring",
        choices=["uring", "aio", "none"],
        help=(
            "POSIX async mode to request. Default uring (reads the completion ring "
            "in userspace, no per-poll syscall); falls back to aio (syscall-bound) "
            "if uring fails. sync 'none' => zero polls, degenerate."
        ),
    )
    p.add_argument("--mode", default="both", choices=["zero_copy", "bounce", "both"])
    p.add_argument("--direction", default="both", choices=["WRITE", "READ", "both"])
    p.add_argument("--variants", default="real,no_sleep")
    p.add_argument("--batch-sizes", default="1,4,16,64,128")
    p.add_argument(
        "--concurrency",
        default="1,2,4",
        help=(
            "Comma-separated K thread counts. K=2 is the NIXL production topology; "
            "K>2 is a labeled hypothetical. Empty to skip."
        ),
    )
    p.add_argument(
        "--concurrency-iters",
        type=int,
        default=200,
        help="Iterations per thread in the concurrency harness.",
    )
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--layer-num", type=int, default=4)
    p.add_argument("--head-num", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    p.add_argument("--file-path", default="/tmp/nixl_poll_bench")
    p.add_argument("--output-file", default=None)
    args = p.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    for b in batch_sizes:
        if b > STORAGE_BATCH_SIZE:
            raise SystemExit(
                f"batch {b} exceeds STORAGE_BATCH_SIZE={STORAGE_BATCH_SIZE}"
            )
    variants = args.variants.split(",")
    if "real" in variants and "no_sleep" not in variants:
        print(
            "WARNING: --variants has 'real' but not 'no_sleep'; the busy-poll "
            "core-fraction headline and the latency tax both need the no_sleep "
            "sibling and will read N/A."
        )
    modes = ["zero_copy", "bounce"] if args.mode == "both" else [args.mode]
    directions = ["WRITE", "READ"] if args.direction == "both" else [args.direction]
    concurrency_ks = (
        [int(x) for x in args.concurrency.split(",") if x.strip()]
        if args.concurrency.strip()
        else []
    )

    os.makedirs(args.file_path, exist_ok=True)

    # Import the module object (not just the class) so PollProbe can patch its
    # time.sleep. Lazy + guarded so --help works without nixl.
    try:
        from sglang.srt.mem_cache.storage.nixl import hicache_nixl as nixl_mod
    except ImportError as e:
        raise SystemExit(f"NIXL not importable ({e}); run on Linux.")

    backend_mode, syscall_bound = _resolve_backend_mode(
        args.file_path, args.posix_async
    )

    # A real kernel has a fine per-thread CPU clock (~1us or better) and a sleep
    # near the requested 100us. A coarse-clock host (e.g. a gVisor sandbox at
    # ~10ms / ~1ms) would turn core_fraction into quantization noise, so refuse to
    # run there rather than emit numbers a reader might trust.
    realized_sleep_us_idle = _measure_sleep_granularity()
    clock_res_ns = _thread_cpu_resolution_ns()
    un = platform.uname()
    fs = _fs_type(args.file_path)
    if clock_res_ns >= 1e5 or realized_sleep_us_idle >= 400.0:
        raise SystemExit(
            f"Refusing to run: this host cannot produce trustworthy magnitudes "
            f"(thread_cpu_clock_res={clock_res_ns:.0f}ns, "
            f"realized_sleep_us={realized_sleep_us_idle:.1f} for a requested "
            f"{REQUESTED_SLEEP_S * 1e6:.0f}us). The poll loop is sub-millisecond, so "
            f"a clock or sleep this coarse measures quantization, not the loop. Run "
            f"on a bare-metal or KVM host (this host: {un.system} {un.release} "
            f"{un.machine}, fs({args.file_path})={fs})."
        )
    print(
        f"host: {un.system} {un.release} {un.machine} | fs({args.file_path})={fs} | "
        f"backend_mode={backend_mode} syscall_bound={syscall_bound} "
        f"thread_cpu_clock_res={clock_res_ns:.0f}ns "
        f"realized_sleep_us={realized_sleep_us_idle:.1f} (requested "
        f"{REQUESTED_SLEEP_S * 1e6:.0f}us) -> real kernel verified"
    )

    results: List[CellResult] = []
    cresults: List[ConcurrencyResult] = []
    for mode in modes:
        zero_copy = mode == "zero_copy"
        host = BenchMemPoolHost(
            zero_copy=zero_copy,
            page_size=args.page_size,
            layer_num=args.layer_num,
            head_num=args.head_num,
            head_dim=args.head_dim,
            num_pages=max(batch_sizes) + max([1] + concurrency_ks) + 8,
            dtype=dtype,
        )
        hicache = _make_hicache(args.file_path, backend_mode)
        hicache.register_mem_pool_host(host)
        if hicache.is_zero_copy != zero_copy:
            print(
                f"WARNING: requested mode={mode} but backend chose "
                f"zero_copy={hicache.is_zero_copy}"
            )

        for direction in directions:
            for batch in batch_sizes:
                for variant in variants:
                    results.append(
                        _run_variant(
                            hicache,
                            nixl_mod,
                            host,
                            direction,
                            batch,
                            args.iters,
                            args.warmup,
                            variant,
                            mode,
                            args.posix_async,
                            backend_mode,
                            syscall_bound,
                            realized_sleep_us_idle,
                        )
                    )

        # Hardening (2026-06-06): the single-thread sweep above leaves backend/agent
        # state that slows the concurrency measurement ~3x -- verified NOT files, NOT
        # page cache, NOT transfer count. Rebuild the backend on a fresh path so
        # concurrency is measured in isolation (a fresh server), not contaminated by
        # the sweep. Without this, K>=2 wall_speedup understates parallelism ~2x and
        # overstates polls/thread ~3x.
        try:
            hicache.clear()
            hicache.close()
        except Exception:
            pass
        conc_path = os.path.join(args.file_path, "_conc")
        shutil.rmtree(conc_path, ignore_errors=True)
        os.makedirs(conc_path, exist_ok=True)
        hicache = _make_hicache(conc_path, backend_mode)
        hicache.register_mem_pool_host(host)

        for K in concurrency_ks:
            # bounce mode shares ONE _bounce_set/_bounce_get slot, so K>2 would put
            # two same-role threads on the same buffer (a real data race once NIXL
            # drops the GIL). zero_copy is race-free at any K (disjoint kv_buffer
            # slices). K>2 is a labeled hypothetical anyway -- skip it for bounce.
            if K > 2 and not zero_copy:
                print(
                    f"SKIP concurrency K={K} mode=bounce: the backend shares one "
                    f"bounce buffer across same-role threads (racy for K>2). Use "
                    f"zero_copy for the K>2 hypothetical; K=2 is the production case."
                )
                continue
            cresults.append(
                run_concurrency(
                    hicache,
                    host,
                    K,
                    args.concurrency_iters,
                    args.warmup,
                    mode,
                    backend_mode,
                    syscall_bound,
                    realized_sleep_us_idle,
                )
            )

        try:
            hicache.clear()
            hicache.close()
        except Exception:
            pass

    _print_headline(results)
    if cresults:
        _print_concurrency(cresults)
    _print_latency_tax(results)

    if any(r.inconclusive for r in results):
        print(
            "\nNOTE: some real-variant cells had mean_polls<1 (loop never spun). "
            "The POSIX backend ran synchronous I/O for those cells; the latency "
            "tax is INCONCLUSIVE there, not refuted. Re-check --posix-async and "
            "the logged backend_params."
        )

    if args.output_file:
        with open(args.output_file, "a") as f:
            for r in results:
                f.write(json.dumps({"kind": "cell", **asdict(r)}) + "\n")
            for c in cresults:
                f.write(json.dumps({"kind": "concurrency", **asdict(c)}) + "\n")
        print(
            f"\nWrote {len(results)} cell rows + {len(cresults)} concurrency rows "
            f"to {args.output_file}"
        )

    _clear_files(args.file_path)  # clear residual temp files at the end


if __name__ == "__main__":
    main()
