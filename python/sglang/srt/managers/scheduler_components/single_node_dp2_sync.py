from __future__ import annotations

import atexit
import ctypes
import importlib.util
import os
import time
from collections.abc import Sequence
from pathlib import Path

import torch

_ENABLE_ENV = "SGLANG_DSPARK_DP2_SHM_MLP_SYNC"
_SESSION_ENV = "SGLANG_DSPARK_DP2_SHM_SESSION_ID"
_TIMEOUT_ENV = "SGLANG_DSPARK_DP2_SHM_TIMEOUT_MS"
_METRICS_ENV = "SGLANG_DSPARK_DP2_SHM_METRICS"
_LIBRARY_ENV = "SGLANG_DSPARK_DP2_SHM_LIBRARY"
_SKIP_GATHER_ENV = "SGLANG_SCHEDULER_SKIP_ALL_GATHER"
_NCCL_GATHER_ENV = "SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH"
_LIBRARY_BASENAME = "sglang_dp2_sync.so"
_EXPECTED_ABI_VERSION = 2
_WORLD_SIZE = 2
_PAYLOAD_WIDTH = 7
_MLP_CHANNEL = "mlp"
_VERIFY_TIER_CHANNEL = "verify_tier"
_VERIFY_TIER_PAYLOAD_MAGIC = 0x4453504B54494552
_ERROR_BUFFER_SIZE = 512
_METRIC_PHASES = (
    "exchange_total",
    "peer_wait",
    "arrival_skew",
    "post_latest_arrival",
)
_enabled: bool | None = None


def _strict_bool_env(name: str, default: str) -> bool:
    value = os.environ.get(name, default)
    if value not in {"0", "1"}:
        raise RuntimeError(f"{name} must be exactly 0 or 1, got {value!r}")
    return value == "1"


def single_node_dp2_sync_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = _strict_bool_env(_ENABLE_ENV, "0")
    return _enabled


def _timeout_ns() -> int:
    value = os.environ.get(_TIMEOUT_ENV, "30000")
    try:
        timeout_ms = int(value)
    except ValueError as error:
        raise RuntimeError(
            f"{_TIMEOUT_ENV} must be a positive integer, got {value!r}"
        ) from error
    if timeout_ms <= 0:
        raise RuntimeError(f"{_TIMEOUT_ENV} must be a positive integer, got {value!r}")
    return timeout_ms * 1_000_000


class _NativeStats(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("total_ns", ctypes.c_uint64),
        ("peer_wait_ns", ctypes.c_uint64),
        ("arrival_skew_ns", ctypes.c_uint64),
        ("post_latest_arrival_ns", ctypes.c_uint64),
    ]


def _resolve_library_path() -> Path:
    override = os.environ.get(_LIBRARY_ENV)
    if override:
        return Path(override)

    package = importlib.util.find_spec("sgl_kernel")
    if package is None or package.submodule_search_locations is None:
        raise RuntimeError(
            "strict DP2 shared-memory sync requires the sgl-kernel package"
        )
    locations = tuple(package.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(
            "strict DP2 shared-memory sync expected one sgl-kernel package "
            f"location, got {locations}"
        )
    return Path(locations[0]) / _LIBRARY_BASENAME


def _load_library() -> ctypes.CDLL:
    path = _resolve_library_path()
    if not path.is_file():
        raise RuntimeError(
            f"strict DP2 shared-memory MLP sync library is missing: {path}"
        )
    library = ctypes.CDLL(str(path))
    library.sglang_dp2_sync_abi_version.argtypes = []
    library.sglang_dp2_sync_abi_version.restype = ctypes.c_uint32
    library.sglang_dp2_sync_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.sglang_dp2_sync_open.restype = ctypes.c_int
    library.sglang_dp2_sync_exchange.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(_NativeStats),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.sglang_dp2_sync_exchange.restype = ctypes.c_int
    library.sglang_dp2_sync_exchange_values.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.POINTER(_NativeStats),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.sglang_dp2_sync_exchange_values.restype = ctypes.c_int
    library.sglang_dp2_sync_close.argtypes = [ctypes.c_void_p]
    library.sglang_dp2_sync_close.restype = None
    actual_abi = library.sglang_dp2_sync_abi_version()
    if actual_abi != _EXPECTED_ABI_VERSION:
        raise RuntimeError(
            "strict DP2 shared-memory MLP sync ABI mismatch: "
            f"expected {_EXPECTED_ABI_VERSION}, got {actual_abi}"
        )
    return library


def validate_single_node_dp2_sync_runtime() -> None:
    if not single_node_dp2_sync_enabled():
        raise RuntimeError(f"{_ENABLE_ENV}=1 is required; Gloo fallback is forbidden")
    session_id = os.environ.get(_SESSION_ENV, "")
    if not session_id:
        raise RuntimeError(f"{_SESSION_ENV} is required")
    _timeout_ns()
    if not _strict_bool_env(_METRICS_ENV, "1"):
        raise RuntimeError(f"{_METRICS_ENV}=1 is required for runtime observability")
    for incompatible_env in (_SKIP_GATHER_ENV, _NCCL_GATHER_ENV):
        if os.environ.get(incompatible_env) != "0":
            raise RuntimeError(
                f"{incompatible_env}=0 must be explicit; alternate MLP sync "
                "paths are forbidden"
            )
    _load_library()


class _SyncMetrics:
    def __init__(self, rank: int, channel: str) -> None:
        from prometheus_client import Counter, Gauge

        if channel == _MLP_CHANNEL:
            metric_prefix = "sglang:dp2_mlp_sync"
            subject = "MLP geometry"
        elif channel == _VERIFY_TIER_CHANNEL:
            metric_prefix = "sglang:dp2_verify_tier_sync"
            subject = "DSpark verify-tier"
        else:
            raise RuntimeError(f"unknown strict DP2 sync channel {channel!r}")
        self._rank = str(rank)
        seconds_total = Counter(
            name=f"{metric_prefix}_seconds_total",
            documentation=(
                f"Cumulative native DP2 shared-memory {subject} exchange "
                "time. peer_wait includes rank arrival skew; "
                "post_latest_arrival isolates transport and wakeup cost."
            ),
            labelnames=["dp_rank", "phase"],
        )
        calls_total = Counter(
            name=f"{metric_prefix}_calls_total",
            documentation=(f"Number of native DP2 shared-memory {subject} exchanges."),
            labelnames=["dp_rank"],
        )
        max_seconds = Gauge(
            name=f"{metric_prefix}_max_seconds",
            documentation=(
                f"Maximum native DP2 {subject} exchange phase time in the latest "
                "one-second reporting window."
            ),
            labelnames=["dp_rank", "phase"],
            multiprocess_mode="mostrecent",
        )
        enabled = Gauge(
            name=f"{metric_prefix}_enabled",
            documentation=(
                f"One when the strict native DP2 shared-memory {subject} sync path "
                "is initialized."
            ),
            labelnames=["dp_rank"],
            multiprocess_mode="mostrecent",
        )
        sequence = Gauge(
            name=f"{metric_prefix}_sequence",
            documentation=(
                f"Latest completed strict native DP2 {subject} sync sequence."
            ),
            labelnames=["dp_rank"],
            multiprocess_mode="mostrecent",
        )
        self._seconds = [
            seconds_total.labels(dp_rank=self._rank, phase=phase)
            for phase in _METRIC_PHASES
        ]
        self._max = [
            max_seconds.labels(dp_rank=self._rank, phase=phase)
            for phase in _METRIC_PHASES
        ]
        self._calls = calls_total.labels(dp_rank=self._rank)
        self._sequence = sequence.labels(dp_rank=self._rank)
        enabled.labels(dp_rank=self._rank).set(1)
        self._totals_ns = [0, 0, 0, 0]
        self._max_ns = [0, 0, 0, 0]
        self._calls_pending = 0
        self._flush_deadline_ns = time.monotonic_ns() + 1_000_000_000

    def observe(self, stats: _NativeStats) -> None:
        values = (
            stats.total_ns,
            stats.peer_wait_ns,
            stats.arrival_skew_ns,
            stats.post_latest_arrival_ns,
        )
        for index, value in enumerate(values):
            self._totals_ns[index] += value
            self._max_ns[index] = max(self._max_ns[index], value)
        self._calls_pending += 1
        now_ns = time.monotonic_ns()
        if now_ns < self._flush_deadline_ns:
            return
        for index in range(len(_METRIC_PHASES)):
            self._seconds[index].inc(self._totals_ns[index] / 1_000_000_000)
            self._max[index].set(self._max_ns[index] / 1_000_000_000)
            self._totals_ns[index] = 0
            self._max_ns[index] = 0
        self._calls.inc(self._calls_pending)
        self._sequence.set(stats.sequence)
        self._calls_pending = 0
        self._flush_deadline_ns = now_ns + 1_000_000_000


class _SingleNodeDP2Sync:
    def __init__(self, rank: int, channel: str) -> None:
        validate_single_node_dp2_sync_runtime()
        self._pid = os.getpid()
        self._rank = rank
        self._channel = channel
        self._library = _load_library()
        self._handle = ctypes.c_void_p()
        self._stats = _NativeStats()
        self._global_payload = (ctypes.c_int64 * (_WORLD_SIZE * _PAYLOAD_WIDTH))()
        self._error = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        session_id_raw = os.environ[_SESSION_ENV]
        if channel == _MLP_CHANNEL:
            session_id = session_id_raw.encode()
        elif channel == _VERIFY_TIER_CHANNEL:
            session_id = f"{session_id_raw}:{channel}".encode()
        else:
            raise RuntimeError(f"unknown strict DP2 sync channel {channel!r}")
        result = self._library.sglang_dp2_sync_open(
            session_id,
            rank,
            _timeout_ns(),
            ctypes.byref(self._handle),
            self._error,
            len(self._error),
        )
        if result != 0:
            raise RuntimeError(self._error.value.decode())
        if self._handle.value is None:
            raise RuntimeError("strict DP2 shared-memory sync returned a null handle")
        self._metrics = _SyncMetrics(rank, channel)

    def exchange(
        self,
        num_tokens: int,
        num_tokens_for_logprob: int,
        can_cuda_graph: bool,
        is_extend_in_batch: bool,
        local_can_run_tbo: bool,
        local_forward_mode: int,
        can_run_breakable_cuda_graph: bool,
    ) -> Sequence[int]:
        if os.getpid() != self._pid:
            raise RuntimeError(
                "strict DP2 shared-memory sync handle crossed a process fork"
            )
        self._error[0] = 0
        result = self._library.sglang_dp2_sync_exchange_values(
            self._handle,
            num_tokens,
            num_tokens_for_logprob,
            int(can_cuda_graph),
            int(is_extend_in_batch),
            int(local_can_run_tbo),
            local_forward_mode,
            int(can_run_breakable_cuda_graph),
            self._global_payload,
            ctypes.byref(self._stats),
            self._error,
            len(self._error),
        )
        if result != 0:
            message = self._error.value.decode()
            raise RuntimeError(
                f"strict DP2 shared-memory {self._channel} sync failed: " + message
            )
        self._metrics.observe(self._stats)
        return self._global_payload

    def close(self) -> None:
        if self._handle.value is not None and os.getpid() == self._pid:
            self._library.sglang_dp2_sync_close(self._handle)
            self._handle = ctypes.c_void_p()


_managers: dict[str, _SingleNodeDP2Sync] = {}


def _get_manager(
    *,
    channel: str,
    group: torch.distributed.ProcessGroup,
    dp_size: int,
    tp_size: int,
    cp_size: int,
) -> _SingleNodeDP2Sync:
    manager = _managers.get(channel)
    if manager is not None:
        return manager
    if not single_node_dp2_sync_enabled():
        raise RuntimeError(
            f"strict DP2 shared-memory {channel} sync was called while disabled"
        )
    if (dp_size, tp_size, cp_size) != (2, 1, 1):
        raise RuntimeError(
            f"strict shared-memory {channel} sync only supports exact "
            f"DP2/TP1/CP1 geometry, got DP{dp_size}/TP{tp_size}/CP{cp_size}"
        )
    if torch.distributed.get_world_size(group) != _WORLD_SIZE:
        raise RuntimeError(
            f"strict DP2 shared-memory {channel} sync requires group world size 2"
        )
    rank = torch.distributed.get_rank(group)
    if rank not in (0, 1):
        raise RuntimeError(
            f"strict DP2 shared-memory {channel} sync got invalid group rank {rank}"
        )
    manager = _SingleNodeDP2Sync(rank, channel)
    _managers[channel] = manager
    atexit.register(manager.close)
    return manager


def exchange_single_node_dp2_mlp_info(
    num_tokens: int,
    num_tokens_for_logprob: int,
    can_cuda_graph: bool,
    is_extend_in_batch: bool,
    local_can_run_tbo: bool,
    local_forward_mode: int,
    can_run_breakable_cuda_graph: bool,
    *,
    group: torch.distributed.ProcessGroup,
    dp_size: int,
    tp_size: int,
    cp_size: int,
) -> Sequence[int]:
    manager = _get_manager(
        channel=_MLP_CHANNEL,
        group=group,
        dp_size=dp_size,
        tp_size=tp_size,
        cp_size=cp_size,
    )
    return manager.exchange(
        num_tokens,
        num_tokens_for_logprob,
        can_cuda_graph,
        is_extend_in_batch,
        local_can_run_tbo,
        local_forward_mode,
        can_run_breakable_cuda_graph,
    )


def exchange_single_node_dp2_verify_tier(
    local_tier_num_tokens: int,
    *,
    group: torch.distributed.ProcessGroup,
    dp_size: int,
    tp_size: int,
    cp_size: int,
) -> list[int]:
    if local_tier_num_tokens < -1:
        raise RuntimeError(
            "strict DP2 verify-tier sync requires a tier >= -1, got "
            f"{local_tier_num_tokens}"
        )
    manager = _get_manager(
        channel=_VERIFY_TIER_CHANNEL,
        group=group,
        dp_size=dp_size,
        tp_size=tp_size,
        cp_size=cp_size,
    )
    payload = manager.exchange(
        local_tier_num_tokens,
        _VERIFY_TIER_PAYLOAD_MAGIC,
        False,
        False,
        False,
        0,
        False,
    )
    for rank in range(_WORLD_SIZE):
        magic = int(payload[rank * _PAYLOAD_WIDTH + 1])
        if magic != _VERIFY_TIER_PAYLOAD_MAGIC:
            raise RuntimeError(
                "strict DP2 verify-tier sync channel mismatch: "
                f"rank {rank} published 0x{magic:016x}"
            )
    return [int(payload[rank * _PAYLOAD_WIDTH]) for rank in range(_WORLD_SIZE)]
