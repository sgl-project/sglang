from __future__ import annotations

import abc
import logging
import re
import threading
from dataclasses import dataclass
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Mapping, Optional, Sequence

import psutil
import torch

from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.pool_host.common import (
    _cuda_host_unregister,
    get_allocator_from_storage,
)
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()

# Host RAM to leave free when sizing HiCache pools (OS, other processes).
HICACHE_HOST_MEMORY_RESERVE_BYTES: int = 10 * (1024**3)

_CGROUP_MEMORY_FILES: tuple[tuple[Path, Path, str], ...] = (
    (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory.current"),
        "cgroup v2",
    ),
    (
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        "cgroup v1",
    ),
)
_CGROUP_UNLIMITED_THRESHOLD = 1 << 60

_WRITE_BACK_STAGING_PAGE_CHUNK = 64


@dataclass(frozen=True, slots=True)
class HiCacheMemoryPlan:
    page_num: int
    page_size: int
    budget_bytes: Optional[int]
    component_bytes: tuple[tuple[str, int], ...]
    pool_bytes: int
    local_process_count: int
    total_pool_bytes: int
    reserve_bytes: int
    available_bytes: int
    available_source: str


def _read_cgroup_memory_value(path: Path) -> Optional[int]:
    try:
        value = path.read_text().strip()
    except (FileNotFoundError, OSError):
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid cgroup memory value in %s: %r", path, value)
        return None


def _read_text_lines(path: Path) -> tuple[str, ...]:
    try:
        return tuple(path.read_text().splitlines())
    except (FileNotFoundError, OSError):
        return ()


def _decode_mountinfo_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _cgroup_memory_file_pair(cgroup_dir: Path, source: str) -> tuple[Path, Path, str]:
    if source == "cgroup v2":
        return cgroup_dir / "memory.max", cgroup_dir / "memory.current", source
    return (
        cgroup_dir / "memory.limit_in_bytes",
        cgroup_dir / "memory.usage_in_bytes",
        source,
    )


def _discover_cgroup_memory_files(
    *,
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    proc_mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> tuple[tuple[Path, Path, str], ...]:
    """Resolve this process' cgroup plus every constraining ancestor."""
    memberships: dict[str, PurePosixPath] = {}
    for line in _read_text_lines(proc_cgroup_path):
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy_id, controllers, membership = fields
        if hierarchy_id == "0" and not controllers:
            memberships["cgroup v2"] = PurePosixPath(membership)
        elif "memory" in controllers.split(","):
            memberships["cgroup v1"] = PurePosixPath(membership)

    mount_candidates: dict[str, list[tuple[Path, Path, bool]]] = {}
    for line in _read_text_lines(proc_mountinfo_path):
        try:
            mount_fields, fs_fields = line.split(" - ", 1)
        except ValueError:
            continue
        mount_parts = mount_fields.split()
        fs_parts = fs_fields.split()
        if len(mount_parts) < 5 or len(fs_parts) < 3:
            continue

        fs_type = fs_parts[0]
        source: Optional[str] = None
        if fs_type == "cgroup2":
            source = "cgroup v2"
        elif fs_type == "cgroup":
            controller_options = set(fs_parts[1].split(",")) | set(
                fs_parts[2].split(",")
            )
            if "memory" in controller_options:
                source = "cgroup v1"
        if source is None or source not in memberships:
            continue

        mount_root = PurePosixPath(_decode_mountinfo_path(mount_parts[3]))
        mount_point = Path(_decode_mountinfo_path(mount_parts[4]))
        membership = memberships[source]
        try:
            relative_membership = membership.relative_to(mount_root)
            is_direct_match = True
        except ValueError:
            # In a cgroup namespace, /proc/self/cgroup is relative to the
            # namespace root while mountinfo retains the host cgroup root.
            relative_membership = PurePosixPath(str(membership).lstrip("/"))
            is_direct_match = False

        current = mount_point.joinpath(*relative_membership.parts)
        try:
            current.relative_to(mount_point)
        except ValueError:
            continue
        mount_candidates.setdefault(source, []).append(
            (mount_point, current, is_direct_match)
        )

    memory_files: list[tuple[Path, Path, str]] = []
    seen: set[tuple[Path, Path, str]] = set()
    for source, candidates in mount_candidates.items():
        direct_matches = [candidate for candidate in candidates if candidate[2]]
        if direct_matches:
            selected = direct_matches
        else:
            selected = [
                candidate
                for candidate in candidates
                if all(
                    path.exists()
                    for path in _cgroup_memory_file_pair(candidate[1], source)[:2]
                )
            ]

        for mount_point, current, _ in selected:
            while True:
                entry = _cgroup_memory_file_pair(current, source)
                if entry not in seen:
                    seen.add(entry)
                    memory_files.append(entry)
                if current == mount_point:
                    break
                current = current.parent

    return tuple(memory_files)


def get_hicache_available_memory(
    *,
    host_available_bytes: Optional[int] = None,
    cgroup_memory_files: Optional[Sequence[tuple[Path, Path, str]]] = None,
) -> tuple[int, str]:
    """Return memory currently available to this process.

    psutil reports node-wide memory in many containers, so cap it by the
    remaining cgroup allowance when a finite v1 or v2 limit is present.
    """
    if host_available_bytes is None:
        host_available_bytes = psutil.virtual_memory().available

    candidates = [(host_available_bytes, "host")]
    if cgroup_memory_files is None:
        memory_files = _discover_cgroup_memory_files() or _CGROUP_MEMORY_FILES
    else:
        memory_files = cgroup_memory_files
    for limit_path, usage_path, source in memory_files:
        limit = _read_cgroup_memory_value(limit_path)
        usage = _read_cgroup_memory_value(usage_path)
        if limit is None or usage is None:
            continue
        if limit >= _CGROUP_UNLIMITED_THRESHOLD:
            continue
        candidates.append((max(0, limit - usage), source))

    return min(candidates, key=lambda item: item[0])


def get_effective_hicache_host_layer_num(device_pool: KVCache) -> int:
    """Return the per-rank layer count allocated by a HiCache host pool."""
    layer_num = device_pool.layer_num
    if not device_pool.layer_shard_enabled:
        return layer_num
    shard_size = device_pool.layer_shard_size
    if shard_size <= 0:
        raise ValueError(
            f"HiCache layer shard size must be positive, got {shard_size}."
        )
    return (layer_num + shard_size - 1) // shard_size


def validate_hicache_memory(
    requested_bytes: int,
    *,
    description: str,
    reserve_bytes: int = HICACHE_HOST_MEMORY_RESERVE_BYTES,
) -> None:
    available_bytes, available_source = get_hicache_available_memory()
    usable_bytes = max(0, available_bytes - reserve_bytes)
    if requested_bytes > usable_bytes:
        raise ValueError(
            f"Not enough host memory for {description}. Requesting "
            f"{requested_bytes / 1e9:.2f} GB plus "
            f"{reserve_bytes / 1e9:.2f} GB reserve, but only "
            f"{available_bytes / 1e9:.2f} GB is available from "
            f"{available_source}. Please reduce the size of the hierarchical cache."
        )


def build_hicache_memory_plan(
    *,
    page_size: int,
    component_bytes_per_token: Mapping[str, int],
    host_size_gb: Optional[int] = None,
    page_num: Optional[int] = None,
    available_memory: Optional[tuple[int, str]] = None,
    reserve_bytes: int = HICACHE_HOST_MEMORY_RESERVE_BYTES,
    local_process_count: Optional[int] = None,
) -> HiCacheMemoryPlan:
    """Derive and validate one shared page count for all HiCache pools."""
    if (host_size_gb is None) == (page_num is None):
        raise ValueError(
            "HiCache memory plan requires exactly one of host_size_gb or page_num."
        )
    if page_size <= 0:
        raise ValueError(f"HiCache page size must be positive, got {page_size}.")
    if not component_bytes_per_token:
        raise ValueError("HiCache memory plan requires at least one host pool.")

    components = tuple(component_bytes_per_token.items())
    invalid = [(name, value) for name, value in components if value <= 0]
    if invalid:
        raise ValueError(f"HiCache bytes per token must be positive: {invalid}.")

    combined_bytes_per_page = sum(value for _, value in components) * page_size
    budget_bytes = None
    if host_size_gb is not None:
        if host_size_gb <= 0:
            raise ValueError("A fixed HiCache memory plan requires --hicache-size > 0.")
        budget_bytes = int(host_size_gb * 1e9)
        page_num = budget_bytes // combined_bytes_per_page
        page_num = sync_fixed_hicache_page_num(page_num, host_size_gb)
    assert page_num is not None
    if page_num <= 0:
        if host_size_gb is not None:
            raise ValueError(
                f"--hicache-size={host_size_gb} GB cannot fit one shared page "
                f"({combined_bytes_per_page / 1e9:.2f} GB)."
            )
        raise ValueError(f"HiCache page count must be positive, got {page_num}.")

    component_bytes = tuple(
        (name, page_num * page_size * bytes_per_token)
        for name, bytes_per_token in components
    )
    pool_bytes = sum(value for _, value in component_bytes)
    if local_process_count is None:
        local_process_count = 1
    if local_process_count <= 0:
        raise ValueError(
            f"HiCache local process count must be positive, got {local_process_count}."
        )
    total_pool_bytes = pool_bytes * local_process_count
    available_bytes, available_source = (
        available_memory
        if available_memory is not None
        else get_hicache_available_memory()
    )
    required_bytes = total_pool_bytes + reserve_bytes
    breakdown = ", ".join(
        f"{name}={value / 1e9:.2f} GB" for name, value in component_bytes
    )
    if required_bytes > available_bytes:
        sizing = (
            f"The {host_size_gb} GB total budget resolves to"
            if host_size_gb is not None
            else "The configured HiCache ratio resolves to"
        )
        size_option = (
            "--hicache-size" if host_size_gb is not None else "--hicache-ratio"
        )
        raise ValueError(
            f"Not enough host memory for HiCache. {sizing} "
            f"{pool_bytes / 1e9:.2f} GB per rank of pools "
            f"({breakdown}), {total_pool_bytes / 1e9:.2f} GB across "
            f"{local_process_count} local ranks, plus "
            f"{reserve_bytes / 1e9:.2f} GB reserve, but only "
            f"{available_bytes / 1e9:.2f} GB is available from "
            f"{available_source}. Please reduce {size_option}."
        )

    sizing = (
        f"budget={budget_bytes / 1e9:.2f} GB"
        if budget_bytes is not None
        else "ratio-based"
    )
    logger.info(
        "HiCache total host-memory plan: %s, pages=%d, %s, "
        "pools=%.2f GB/rank, local_ranks=%d, local_pools=%.2f GB, "
        "reserve=%.2f GB, available=%.2f GB (%s).",
        sizing,
        page_num,
        breakdown,
        pool_bytes / 1e9,
        local_process_count,
        total_pool_bytes / 1e9,
        reserve_bytes / 1e9,
        available_bytes / 1e9,
        available_source,
    )
    return HiCacheMemoryPlan(
        page_num=page_num,
        page_size=page_size,
        budget_bytes=budget_bytes,
        component_bytes=component_bytes,
        pool_bytes=pool_bytes,
        local_process_count=local_process_count,
        total_pool_bytes=total_pool_bytes,
        reserve_bytes=reserve_bytes,
        available_bytes=available_bytes,
        available_source=available_source,
    )


def sync_fixed_hicache_size(size: int, host_size: int) -> int:
    """Sync fixed-size HiCache token capacity across PP ranks.

    A fixed --hicache-size is specified in GB, but each PP stage may have a
    different bytes/token because it owns different layers. Use the global
    minimum token capacity within the PP group so all stages expose the same
    host-cache capacity.
    Ratio-based sizing already derives from the synced device pool size.
    """
    if host_size <= 0 or not torch.distributed.is_available():
        return size

    if not torch.distributed.is_initialized():
        return size

    try:
        from sglang.srt.distributed.parallel_state import get_pp_group

        pp_group = get_pp_group()
    except AssertionError:
        return size

    if pp_group.world_size <= 1:
        return size

    tensor = torch.tensor(size, dtype=torch.int64)
    torch.distributed.all_reduce(
        tensor,
        op=torch.distributed.ReduceOp.MIN,
        group=pp_group.cpu_group,
    )
    synced_size = int(tensor.item())

    if synced_size != size:
        logger.info(
            "Sync fixed-size HiCache host token capacity from %d to %d.",
            size,
            synced_size,
        )
    return synced_size


def sync_fixed_hicache_page_num(page_num: int, host_size: int) -> int:
    """Sync a total-budget HiCache page count across PP ranks."""
    return sync_fixed_hicache_size(page_num, host_size)


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class HostKVCache(abc.ABC):

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
        allocator_type: str = "default",
        host_page_num: Optional[int] = None,
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.can_use_write_back_jit = False

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_page_num is not None:
            if host_page_num <= 0:
                raise ValueError(
                    f"HiCache host page count must be positive, got {host_page_num}."
                )
            self.page_num = host_page_num
            self.size = self.page_num * self.page_size
        elif host_size > 0:
            self.size = sync_fixed_hicache_size(
                int(host_size * 1e9 // self.size_per_token), host_size
            )
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        if host_page_num is None:
            # Align up the host memory pool size to the page size
            self.page_num = self.size // self.page_size + 1
            self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        if self.size <= device_pool.size:
            logger.warning(
                "HiCache host KV pool (%d tokens) is smaller than the device pool (%d tokens);"
                "L2 cache effectiveness is reduced."
                "Consider increasing --hicache-ratio (or --hicache-size) for higher L2 cache hit rate.",
                self.size,
                device_pool.size,
            )

        requested_bytes = self.size * self.size_per_token
        validate_hicache_memory(requested_bytes, description="hierarchical KV cache")
        logger.info(
            "Allocating %.2f GB host memory for hierarchical KV cache.",
            requested_bytes / 1e9,
        )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()

    def destroy(self):
        """Unregister pinned host buffers in userspace before process exit.

        Large cudaHostRegister'd buffers are otherwise unpinned by the kernel
        during SIGKILL reclaim, which can stall teardown in uninterruptible
        sleep for tens of seconds. Idempotent. (Only the host_register path
        needs this; npu/musa pin_memory buffers are freed by torch.)
        """
        if getattr(self, "_destroyed", False):
            return
        self._destroyed = True
        buffers = getattr(self, "kv_buffer", None)
        if buffers is not None and self.pin_memory and (_is_cuda or _is_hip):
            if not isinstance(buffers, (list, tuple)):
                buffers = [buffers]
            for buf in buffers:
                if buf is not None:
                    _cuda_host_unregister(buf)
        self.kv_buffer = None

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    def _is_device_layer_sharded(self, device_pool=None) -> bool:
        device_pool = device_pool or self.device_pool
        return bool(device_pool.layer_shard_enabled)

    def _device_owned_layer_range(self, device_pool=None) -> tuple[int, int]:
        """Contiguous ``[start, end)`` local device layers this rank stores.

        ``(0, layer_num)`` when the device pool is not layer-sharded.
        """
        device_pool = device_pool or self.device_pool
        if not self._is_device_layer_sharded(device_pool):
            return 0, device_pool.layer_num
        return device_pool._owned_local_layer_range()

    def _effective_host_layer_num(self, device_pool=None) -> int:
        """Number of layers the host pool allocates for this rank."""
        device_pool = device_pool or self.device_pool
        return get_effective_hicache_host_layer_num(device_pool)

    def _is_device_layer_owned(self, device_pool, layer_id: int) -> bool:
        start, end = self._device_owned_layer_range(device_pool)
        return start <= layer_id < end

    def _host_layer_index(self, layer_id: int, device_pool=None) -> int:
        """Map a full local device layer id to its compacted host-buffer slot."""
        start, _ = self._device_owned_layer_range(device_pool)
        return layer_id - start

    def _owned_device_layer_ids(self, device_pool) -> list[int]:
        start, end = self._device_owned_layer_range(device_pool)
        return list(range(start, end))

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        """
        Load KV data from the host memory pool to the device memory pool for a specific layer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from the device memory pool to the host memory pool for all layers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a flat data page from the host memory pool.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """
        Get a dummy flat data page from the host memory pool.
        This is used for prefetching or initializing empty pages.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to the host memory pool.
        """
        raise NotImplementedError()

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        """Return True if per-page strides are multiples of *page_size_bytes*.

        Subclasses should override this with a layout-specific stride formula.
        This base implementation logs a warning and returns False (safe default).
        """
        logger.warning(
            "%s does not implement is_stride_page_aligned(); assuming not aligned. "
            "O_DIRECT with a file-based NIXL backend will fall back to copy mode for this pool.",
            type(self).__name__,
        )
        return False

    @synchronized
    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)
        # Per-slot flag used to detect double-free.
        # slot_used[k] is true if slot k is allocated.
        self.slot_used = torch.zeros(self.size, dtype=torch.bool)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        assert not self.slot_used[select_index].any(), (
            f"Double-alloc detected: slots already allocated: "
            f"{select_index[self.slot_used[select_index]].tolist()}."
        )
        self.slot_used[select_index] = True

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.cpu()
        assert self.slot_used[indices_cpu].all(), (
            f"Double-free detected: slots not currently allocated: "
            f"{indices_cpu[~self.slot_used[indices_cpu]].tolist()}."
        )
        self.slot_used[indices_cpu] = False
        self.free_slots = torch.cat([self.free_slots, indices_cpu])
        return len(indices)
