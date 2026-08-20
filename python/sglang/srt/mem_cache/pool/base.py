"""Device-side KV cache abstractions: the write-location bundle, the byte-span
descriptor, and the ``KVCache`` / ``BaseSWAKVPool`` base classes every device pool
under ``mem_cache/pool/`` derives from."""

from __future__ import annotations

import abc
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.cache_controller import LayerDoneCounter

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


@dataclass
class KVWriteLoc:
    """Write target(s) for ``KVCache.set_kv_buffer``.

    All location info lives here (in the attention metadata), NOT in the pool:
    - ``loc``: the generic per-token write location (the allocated
      ``out_cache_loc``). VIRTUAL under the unified memory pool (it indexes the
      virtual slot space); already physical for a non-unified memory pool.
    - ``swa_loc``: the pre-translated SWA-sub-pool PHYSICAL location for hybrid
      SWA pools (``None`` otherwise).
    - ``full_loc``: the pre-translated full-attention-sub-pool PHYSICAL location
      for the unified memory pool (``None`` otherwise), computed once per forward in
      attention metadata (``ForwardMetadata.out_cache_loc_full_physical``). The
      shared full pool writes it directly; the pool never translates (replacing
      the former per-layer v2p gather / ``set_full_loc`` pin).

    ``swa_loc`` and ``full_loc`` are the parallel pair (each a pre-resolved
    PHYSICAL loc into its sub-pool, mirroring ``swa_kv_pool`` / ``full_kv_pool``);
    ``loc`` is the generic, possibly-virtual fallback. Bundling them lets a
    backend issue one ``set_kv_buffer`` call regardless of pool type.
    """

    loc: torch.Tensor
    swa_loc: Optional[torch.Tensor] = None
    full_loc: Optional[torch.Tensor] = None

    def __post_init__(self):
        # swa_loc / full_loc are resolved once at metadata-init from the full
        # (padded) out_cache_loc; piecewise/DP-padded paths later narrow loc per
        # layer, so slice these pre-resolved locs to match (same per-token order).
        if self.swa_loc is not None and self.swa_loc.shape[0] != self.loc.shape[0]:
            self.swa_loc = self.swa_loc[: self.loc.shape[0]]
        if self.full_loc is not None and self.full_loc.shape[0] != self.loc.shape[0]:
            self.full_loc = self.full_loc[: self.loc.shape[0]]


def unwrap_write_loc(loc_info):
    """Return ``(loc, swa_loc, full_loc)`` from a ``KVWriteLoc`` or a bare loc."""
    if isinstance(loc_info, KVWriteLoc):
        return loc_info.loc, loc_info.swa_loc, loc_info.full_loc
    return loc_info, None, None


class KvBufferDesc:
    """Byte-span math for one KV buffer laid out as rows of ``row_bytes`` holding
    ``tokens_per_row`` tokens each (a row = one token slot, or one whole page)."""

    __slots__ = ("name", "shape", "row_bytes", "tokens_per_row")

    def __init__(self, name: str, shape: tuple, *, row_bytes: int, tokens_per_row: int):
        self.name = name
        self.shape = tuple(shape)
        self.row_bytes = int(row_bytes)
        self.tokens_per_row = int(tokens_per_row)

    def _rows(self, num_tokens: int) -> int:
        n = max(int(num_tokens), 0)
        return (n + self.tokens_per_row - 1) // self.tokens_per_row

    def reserved_span_bytes(self, itemsize: int) -> int:
        """Full upper-bound byte size of the buffer (its whole tensor)."""
        return math.prod(self.shape) * itemsize

    def prefix_span_bytes(self, num_tokens: int, page_size: int) -> int:
        """Bytes to back to make the first ``num_tokens`` tokens usable."""
        return self._rows(num_tokens) * self.row_bytes

    def final_span_bytes(self, num_tokens: int, page_size: int) -> int:
        """Bytes of the final advertised span (adds the padded page). CEIL, not floor:
        an unaligned count must still cover its partial last page (e.g. n=17, page=16
        -> 3 pages, not 2)."""
        return self._rows(max(int(num_tokens), 0) + page_size) * self.row_bytes

    def item_len_bytes(self, page_size: int) -> int:
        """Per-page transfer chunk (one page's worth of this buffer)."""
        return (page_size // self.tokens_per_row) * self.row_bytes


class KVCache(abc.ABC):
    layer_shard_enabled: bool = False
    post_capture_active: bool = False

    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        allocation_label: Optional[str] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.allocation_label = allocation_label
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

    def _finalize_allocation_log(self, num_tokens: int):
        """Common logging and mem_usage computation for KV cache allocation.
        Supports both tuple (K, V) size returns and single KV size returns.
        """
        cache_name = (
            f"{self.allocation_label} KV Cache"
            if self.allocation_label is not None
            else "KV Cache"
        )
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"{cache_name} {'VA upper bound' if self.post_capture_active else 'is allocated'}. dtype: {self.dtype}, "
                f"#tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, "
                f"V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"{cache_name} {'VA upper bound' if self.post_capture_active else 'is allocated'}. dtype: {self.dtype}, "
                f"#tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    def get_kv_buffer_shape(self) -> Tuple[torch.Size, torch.Size]:
        k_buffer, v_buffer = self.get_kv_buffer(self.start_layer)
        return k_buffer.shape, v_buffer.shape

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError()

    def get_kv_cache_quant_method(self) -> Any:
        """Return the concrete KV quant method, unwrapping composite KV pools."""
        fallback = None
        for pool in (
            self,
            getattr(self, "full_kv_pool", None),
            getattr(self, "swa_kv_pool", None),
        ):
            if pool is None:
                continue
            quant_method = getattr(pool, "quant_method", None)
            if quant_method is None:
                continue
            if getattr(quant_method, "name", None) != "unquantized":
                return quant_method
            fallback = quant_method
        return fallback

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool


class BaseSWAKVPool(KVCache):
    """ABC for SWA-like KV pools.

    Subclasses expose a `swa_kv_pool` sub-pool plus a full -> swa index
    mapping. Used by `SWATokenToKVPoolAllocator` and the disagg paths to
    handle SWA state separately from the full KV state.
    """

    swa_kv_pool: KVCache

    @abc.abstractmethod
    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        raise NotImplementedError()
