from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache

import numpy as np
import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager, StateManager
from sglang_simulator.utils import get_logger

logger = get_logger()


class TransportDirection(Enum):
    H2D = "H2D"
    D2H = "D2H"


class HicacheTransportEstimator(ABC):
    def __init__(
        self,
        memory_read_bandwidth_bytes: float,
        memory_write_bandwidth_bytes: float,
    ):
        self.memory_read_bandwidth_bytes = memory_read_bandwidth_bytes
        self.memory_write_bandwidth_bytes = memory_write_bandwidth_bytes

    @abstractmethod
    def estimate_bandwidth(
        self, size_bytes: np.ndarray, direction: TransportDirection
    ) -> np.ndarray:
        raise NotImplementedError


class HicacheTransportOverheadEstimator(HicacheTransportEstimator):
    """Bandwidth model with a fixed launch overhead and 85% efficiency."""

    def estimate_bandwidth(
        self, size_bytes: np.ndarray, direction: TransportDirection
    ) -> np.ndarray:
        if direction is TransportDirection.H2D:
            overhead_s = 6.67e-6
            bandwidth = self.memory_read_bandwidth_bytes * 0.85
        else:
            overhead_s = 4e-6
            bandwidth = self.memory_write_bandwidth_bytes * 0.85
        return size_bytes * bandwidth / (overhead_s * bandwidth + size_bytes)


def compute_contiguous_index_lengths(
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
) -> np.ndarray:
    if len(host_indices) != len(device_indices):
        raise ValueError("Host and device cache index lists must have the same length.")
    if len(host_indices) == 0:
        return np.empty(0, dtype=np.float64)

    host = np.asarray(host_indices.cpu(), dtype=np.int64)
    device = np.asarray(device_indices.cpu(), dtype=np.int64)
    contiguous = (np.diff(host) == 1) & (np.diff(device) == 1)
    cuts = np.flatnonzero(~contiguous) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(host_indices)]
    return (ends - starts).astype(np.float64)


def allocate_meta_tensor(
    dims,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator=None,
    registration_granularity_bytes=None,
) -> torch.Tensor:
    """Allocate metadata-only host cache payload for simulation."""
    return torch.empty(dims, dtype=dtype, device="meta")


def _install_meta_allocators() -> None:
    modules = []
    try:
        from sglang.srt.mem_cache import memory_pool_host

        modules.append(memory_pool_host)
    except ImportError:
        pass
    try:
        from sglang.srt.mem_cache.pool_host import common

        modules.append(common)
    except ImportError:
        pass

    for module in modules:
        allocators = getattr(module, "ALLOC_MEMORY_FUNCS", None)
        if allocators is None:
            continue
        allocators.default_factory = lambda: allocate_meta_tensor
        for key in list(allocators):
            allocators[key] = allocate_meta_tensor


_SIMULATED_AVAILABLE_HOST_MEMORY_BYTES = 1 << 60


class _PsutilProxy:
    def __init__(self, psutil_module):
        self._psutil_module = psutil_module

    def virtual_memory(self):
        snapshot = self._psutil_module.virtual_memory()
        return snapshot._replace(
            available=max(
                snapshot.available,
                _SIMULATED_AVAILABLE_HOST_MEMORY_BYTES,
            )
        )

    def __getattr__(self, name):
        return getattr(self._psutil_module, name)


def _call_with_meta_host_memory(original_init, self, *args, **kwargs):
    """Bypass physical host-payload checks while meta allocation is active."""
    init_globals = getattr(original_init, "__globals__", None)
    psutil_module = init_globals.get("psutil") if init_globals is not None else None
    if psutil_module is None:
        return original_init(self, *args, **kwargs)

    proxy = _PsutilProxy(psutil_module)
    init_globals["psutil"] = proxy
    try:
        return original_init(self, *args, **kwargs)
    finally:
        if init_globals.get("psutil") is proxy:
            init_globals["psutil"] = psutil_module


@lru_cache(maxsize=256)
def get_refined_cache_size_per_token(host_pool) -> float:
    internal_size = float(host_pool.get_size_per_token())
    scheduler_config = ConfigManager.get_scheduler_config()
    if scheduler_config is None or scheduler_config.kv_cache_data_type is None:
        logger.warning(
            "Scheduler KV-cache dtype is unavailable; using %s's native "
            "size-per-token value.",
            host_pool.__class__.__name__,
        )
        return internal_size

    internal_dtype = host_pool.dtype
    dtype_factor = scheduler_config.kv_cache_data_type.bytes / internal_dtype.itemsize
    return internal_size * dtype_factor


_DSV4_TRANSFER_SIZE_MULTIPLIERS = {
    "swa": 130,
    "deepseek_v4_c4": 65,
    "deepseek_v4_c4_indexer": 132,
    "deepseek_v4_c128": 3,
    "deepseek_v4_c4_state": 256,
    "deepseek_v4_c128_state": 256,
    "deepseek_v4_indexer_state": 128,
    "deepseek_v4_c4_indexer_state": 128,
}

_DSV4_PAGED_POOL_NAMES = {
    "swa",
    "deepseek_v4_c4",
    "deepseek_v4_c4_indexer",
    "deepseek_v4_c128",
}


def _dsv4_transfer_size_multiplier(host_pool) -> int | None:
    return _DSV4_TRANSFER_SIZE_MULTIPLIERS.get(str(getattr(host_pool, "pool_name", "")))


def get_transfer_size_per_unit(host_pool, *, all_layers: bool) -> float:
    """Return calibrated bytes moved for one transfer unit.

    DSv4's paged and state pools expose physical page-row geometry through Unified
    HiCache. Preserve the 0714 estimator's calibrated logical-byte multipliers while
    keeping transfer dispatch on the current Unified pool interfaces.
    """
    size = get_refined_cache_size_per_token(host_pool)
    dsv4_multiplier = _dsv4_transfer_size_multiplier(host_pool)
    if dsv4_multiplier is not None:
        return size * dsv4_multiplier

    layer_num = max(int(getattr(host_pool, "layer_num", 1)), 1)
    per_layer_size = size / layer_num
    return per_layer_size * layer_num if all_layers else per_layer_size


def _transport_estimator() -> HicacheTransportEstimator:
    platform = ConfigManager.get_platform_config()
    return HicacheTransportOverheadEstimator(
        memory_read_bandwidth_bytes=platform.memory_read_bandwidth,
        memory_write_bandwidth_bytes=platform.memory_write_bandwidth,
    )


def _normalize_transfer_indices(self, host_indices, device_indices):
    if host_indices is None or device_indices is None:
        return None, None
    if hasattr(self, "_to_page_indices"):
        host_indices = self._to_page_indices(host_indices)
        device_indices = self._to_page_indices(device_indices)
    return host_indices, device_indices


def _transfer_segment_lengths(
    self, host_indices, device_indices, *, count_logical_tokens: bool = False
) -> np.ndarray:
    if host_indices is None or device_indices is None:
        return np.empty(0, dtype=np.float64)

    original_unit_count = len(host_indices)
    host_indices, device_indices = _normalize_transfer_indices(
        self, host_indices, device_indices
    )
    lengths = compute_contiguous_index_lengths(host_indices, device_indices)
    if (
        len(lengths)
        and count_logical_tokens
        and str(getattr(self, "pool_name", "")) in _DSV4_PAGED_POOL_NAMES
    ):
        # The 0714 DSv4 paged-pool H2D estimator counted logical token slots
        # while using page-row contiguity to determine transfer segments.
        lengths[-1] += original_unit_count - len(host_indices)
    return lengths


def _sim_load_to_device_per_layer(
    self,
    device_pool,
    host_indices,
    device_indices,
    layer_id,
    io_backend,
    *,
    is_draft: bool = False,
) -> None:
    segment_lengths = _transfer_segment_lengths(
        self, host_indices, device_indices, count_logical_tokens=True
    )
    if not len(segment_lengths):
        return

    size_bytes = segment_lengths * get_transfer_size_per_unit(self, all_layers=False)
    StateManager.inc_hicache_l2_load_stats(
        call_count=1,
        segment_count=len(size_bytes),
        units=int(np.sum(segment_lengths)),
        bytes_=float(np.sum(size_bytes)),
    )
    bandwidth = _transport_estimator().estimate_bandwidth(
        size_bytes, TransportDirection.H2D
    )
    StateManager.inc_hicache_l2_load_dur(float(np.sum(size_bytes / bandwidth)))


def _sim_backup_from_device_all_layer(
    self, device_pool, host_indices, device_indices, io_backend
) -> None:
    segment_lengths = _transfer_segment_lengths(self, host_indices, device_indices)
    if not len(segment_lengths):
        return

    size_bytes = segment_lengths * get_transfer_size_per_unit(self, all_layers=True)
    bandwidth = _transport_estimator().estimate_bandwidth(
        size_bytes, TransportDirection.D2H
    )
    StateManager.inc_hicache_l2_backup_dur(float(np.sum(size_bytes / bandwidth)))


def _sim_get_data_page(self, index, flat: bool = True) -> torch.Tensor:
    return torch.ones(size=(1, 1)) * index


def _sim_set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
    return None


def _install_transport_methods(target) -> None:
    original_init = target.__init__

    def wrapped_init(self, *args, **kwargs):
        _install_meta_allocators()
        if "pin_memory" in kwargs:
            kwargs["pin_memory"] = False
        return _call_with_meta_host_memory(original_init, self, *args, **kwargs)

    target.__init__ = wrapped_init
    target.load_to_device_per_layer = _sim_load_to_device_per_layer
    target.backup_from_device_all_layer = _sim_backup_from_device_all_layer
    target.get_data_page = _sim_get_data_page
    target.set_from_flat_data_page = _sim_set_from_flat_data_page


class C_MHATokenToKVPoolHostHook(BaseHook):
    HOOK_CLASS_NAME = "MHATokenToKVPoolHost"
    HOOK_MODULE_NAME = r"^sglang\.srt\.mem_cache\.(memory_pool_host|pool_host\.mha)$"
    REGEX = True
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        _install_transport_methods(target)


class C_HostKVCacheHook(BaseHook):
    HOOK_CLASS_NAME = "HostKVCache"
    HOOK_MODULE_NAME = r"^sglang\.srt\.mem_cache\.(memory_pool_host|pool_host\.base)$"
    REGEX = True
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, *args, **kwargs):
            _install_meta_allocators()
            if "pin_memory" in kwargs:
                kwargs["pin_memory"] = False
            elif len(args) > 5:
                args = list(args)
                args[5] = False
            return _call_with_meta_host_memory(original_init, self, *args, **kwargs)

        target.__init__ = wrapped_init


class C_PackedSingleKVPoolHook(BaseHook):
    """Allocate byte-packed single KV pools from their runtime geometry."""

    HOOK_CLASS_NAME = r".*SingleKVPool$"
    HOOK_MODULE_NAME = r"^sglang\.srt\.mem_cache\..+$"
    REGEX = True
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original_create_buffer = target.create_buffer

        def wrapped_create_buffer(self, *, num_pages: int):
            if self.store_dtype != torch.uint8 or not hasattr(
                self, "get_bytes_per_token"
            ):
                return original_create_buffer(self, num_pages=num_pages)

            try:
                return original_create_buffer(self, num_pages=num_pages)
            except AssertionError:
                # Some packed pools validate production-only geometry. Simulation
                # still needs a correctly sized byte buffer for dummy model configs.
                pass

            bytes_per_token = self.get_bytes_per_token()
            self.kv_cache_total_dim = bytes_per_token
            bytes_per_page = self.page_size * bytes_per_token
            self.bytes_per_page_padded = (bytes_per_page + 575) // 576 * 576
            return torch.zeros(
                num_pages,
                self.bytes_per_page_padded,
                dtype=self.store_dtype,
                device=self.device,
            )

        target.create_buffer = wrapped_create_buffer


class C_GenericHostKVCacheSubclassHook(BaseHook):
    HOOK_CLASS_NAME = r".*(?:PoolHost|HostPool)$"
    HOOK_MODULE_NAME = r"^sglang\.srt\.mem_cache\.(memory_pool_host|pool_host\..+)$"
    REGEX = True
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        if any(base.__name__ == "HostKVCache" for base in target.__mro__[1:]):
            _install_transport_methods(target)
