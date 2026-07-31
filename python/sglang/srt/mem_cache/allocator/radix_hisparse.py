"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

"""Allocator facade for Radix-managed HiSparse KV cache.

Radix owns stable L1 indices. Each L1 index addresses one CPU full-KV row and
one GPU indexer-K row. HiSparse separately owns the bounded GPU full-KV L0 and
may recycle an L0 index without changing the corresponding L1 lifetime.

The L1-to-L0 mapping in this module routes new full-KV writes only. HiSparse's
per-request, per-layer HotBuffer metadata remains the residency source of truth.
The public facade exposes ordinary allocator semantics for L1 only; the
coordinator reaches L0 and the CPU L1 backing through explicit facade methods.
"""

import weakref
from typing import TYPE_CHECKING, Optional, TypeGuard

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost


class RadixHiSparseL1Allocator(PagedTokenToKVPoolAllocator):
    """Inner allocator for stable Radix-owned HiSparse L1 slots.

    This object is kept behind ``RadixHiSparseTokenToKVPoolAllocator`` so only
    the facade is handed to the scheduler and RadixCache.
    """

    def __init__(
        self,
        l0_capacity: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kvcache: HiSparseDSATokenToKVPool,
        need_sort: bool,
        host_to_device_ratio: int = 2,
    ):
        if (
            not isinstance(host_to_device_ratio, int)
            or isinstance(host_to_device_ratio, bool)
            or host_to_device_ratio < 1
        ):
            raise ValueError("host_to_device_ratio must be a positive integer")

        self.l0_capacity = l0_capacity
        self.host_to_device_ratio = host_to_device_ratio
        self.compress_ratio = 1
        super().__init__(
            size=l0_capacity * host_to_device_ratio,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )

    @property
    def l1_capacity(self) -> int:
        return self.size

    def l1_available_size(self) -> int:
        """Return only composite-L1 capacity; L0 pressure is independent."""
        return self.available_size()

    def full_kv_host_locs(self, l1_indices: torch.Tensor) -> torch.Tensor:
        """Map L1 slots to CPU full-KV rows (identity in the MVP)."""
        return l1_indices

    def index_k_device_locs(self, l1_indices: torch.Tensor) -> torch.Tensor:
        """Map L1 slots to GPU indexer-K rows (identity in the MVP)."""
        return l1_indices

    def get_last_loc_compressed(self, last_locs: torch.Tensor) -> torch.Tensor:
        return last_locs

    def alloc_l1_only(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate an extend range in L1 without reserving any L0 slots."""
        return self.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    def resize(self, config) -> None:
        """Resize from the L0 pool budget while preserving the L1 ratio."""
        self.l0_capacity = config.max_total_num_tokens
        self.size = self.l0_capacity * self.host_to_device_ratio
        self.num_pages = self.size // self.page_size
        self.clear()


class HiSparseL0SlotAllocator:
    """Inner allocator for bounded GPU full-KV L0 slots.

    This object is intentionally not a ``BaseTokenToKVPoolAllocator`` and must
    never be handed to RadixCache. It cannot allocate or free L1 indices.
    """

    def __init__(
        self,
        l0_capacity: int,
        l1_capacity: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kvcache: HiSparseDSATokenToKVPool,
        need_sort: bool,
    ):
        if l1_capacity < l0_capacity:
            raise ValueError("l1_capacity must cover every L0 HotBuffer slot")

        self.l1_capacity = l1_capacity
        self.page_size = page_size
        self.device = device
        self._kvcache = kvcache
        self.l0_slot_allocator = PagedTokenToKVPoolAllocator(
            size=l0_capacity,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )

        # Note: append -1 so last_loc=-1 maps to an invalid L0 slot. The page
        # padding covers the final padded L1 page allocated by the paged pool.
        self.l1_to_l0_write_mapping = torch.cat(
            [
                torch.zeros(
                    l1_capacity + page_size,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )
        self._kvcache.register_mapping(weakref.proxy(self.l1_to_l0_write_mapping))

    @property
    def size(self) -> int:
        return self.l0_slot_allocator.size

    def available_size(self) -> int:
        """Return only L0 capacity; it is not an L1 admission budget."""
        return self.l0_slot_allocator.available_size()

    def get_kvcache(self) -> HiSparseDSATokenToKVPool:
        return self._kvcache

    def alloc(self, need_size: int):
        return self.l0_slot_allocator.alloc(need_size)

    def free(self, l0_indices: torch.Tensor) -> None:
        """Release L0 rows without changing any L1 ownership."""
        if l0_indices.numel() == 0:
            return
        l0_indices = l0_indices[l0_indices > 0]
        if l0_indices.numel() > 0:
            self.l0_slot_allocator.free(l0_indices)

    def bind_write_locs(
        self,
        l1_indices: torch.Tensor,
        l0_indices: torch.Tensor,
    ) -> None:
        """Route new full-KV writes from L1 indices to temporary L0 slots."""
        if l1_indices.numel() != l0_indices.numel():
            raise ValueError("L1 and L0 index counts must match")
        self.l1_to_l0_write_mapping[l1_indices] = l0_indices

    def lookup_write_locs(self, l1_indices: torch.Tensor) -> torch.Tensor:
        return self.l1_to_l0_write_mapping[l1_indices]

    def acquire_request_l0_buffer(self, l1_indices: torch.Tensor, need_size: int):
        """Detach initial write slots and return a page-aligned L0 buffer.

        Existing mapped rows are reused first. Any surplus rows are released to
        L0, and any shortfall is allocated from L0. On failure, the L1-to-L0
        write mapping remains unchanged.
        """
        if need_size % self.page_size != 0:
            raise ValueError("HotBuffer allocation must be page-aligned")

        mapped_l0_indices = self.lookup_write_locs(l1_indices)
        mapped_l0_indices = mapped_l0_indices[mapped_l0_indices > 0]

        if len(mapped_l0_indices) >= need_size:
            buffer_l0_indices = mapped_l0_indices[:need_size]
            surplus_l0_indices = mapped_l0_indices[need_size:]
        else:
            page_residual = len(mapped_l0_indices) % self.page_size
            if page_residual:
                page_tail = torch.arange(
                    1,
                    self.page_size - page_residual + 1,
                    dtype=torch.int64,
                    device=self.device,
                )
                mapped_l0_indices = torch.cat(
                    [mapped_l0_indices, mapped_l0_indices[-1] + page_tail]
                )

            extra_size = need_size - len(mapped_l0_indices)
            if extra_size > self.available_size():
                return None
            extra_l0_indices = self.alloc(extra_size)
            if extra_l0_indices is None:
                return None
            buffer_l0_indices = torch.cat([mapped_l0_indices, extra_l0_indices])
            surplus_l0_indices = mapped_l0_indices[:0]

        self.l1_to_l0_write_mapping[l1_indices] = 0
        self.free(surplus_l0_indices)
        return buffer_l0_indices

    def release_write_locs(self, l1_indices: torch.Tensor) -> None:
        """Detach L1 write routes and free only their mapped L0 slots."""
        l0_indices = self.lookup_write_locs(l1_indices)
        self.l1_to_l0_write_mapping[l1_indices] = 0
        self.free(l0_indices)

    def clear(self) -> None:
        self.l0_slot_allocator.clear()
        self.l1_to_l0_write_mapping[:-1].fill_(0)

    def resize(self, config) -> None:
        """Shrink L0 with the post-capture device-pool budget."""
        self.l0_slot_allocator.resize(config)


class RadixHiSparseL1HostPool:
    """L1-lifetime view over the physical CPU full-KV storage.

    The wrapped host pool supplies buffers and copy kernels, while L1 remains
    the only allocator. Exposing the ordinary ``size``/``available_size``
    contract lets the shared coordinator treat legacy and Radix host pools
    uniformly without creating a second host-slot lifetime.
    """

    def __init__(self, storage_pool, l1_allocator: RadixHiSparseL1Allocator):
        self._storage_pool = storage_pool
        self.l1_allocator = l1_allocator

    # Keep the view deliberately narrow: byte movement and transfer
    # registration are valid here, but storage allocation/lifetime remains
    # exclusively owned by the composite L1 allocator and facade.
    _DELEGATED_STORAGE_MEMBERS = frozenset(
        {
            "backup_from_device_all_layer",
            "can_use_jit",
            "data_ptrs",
            "device_pool",
            "dtype",
            "end_layer",
            "get_contiguous_buf_infos",
            "item_size_bytes",
            "kv_buffer",
            "kv_cache_dim",
            "layer_num",
            "layout",
            "load_to_device_per_layer",
            "page_size",
            "start_layer",
            "token_stride_size",
        }
    )

    @property
    def size(self) -> int:
        return self.l1_allocator.size

    @property
    def storage_size(self) -> int:
        return self._storage_pool.size

    def available_size(self) -> int:
        return self.l1_allocator.available_size()

    def alloc(self, need_size: int):
        raise RuntimeError("CPU L1 rows must be allocated through the L1 allocator")

    def alloc_paged_token_slots(self, *args, **kwargs):
        raise RuntimeError("CPU L1 rows use identity L1 indices, not host allocation")

    def free(self, indices: torch.Tensor) -> int:
        raise RuntimeError("CPU L1 rows must be freed through the L1 allocator")

    def __getattr__(self, name):
        if name not in self._DELEGATED_STORAGE_MEMBERS:
            raise AttributeError(
                f"{type(self).__name__!s} does not expose storage member {name!r}"
            )
        return getattr(self._storage_pool, name)


class RadixHiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Public allocator facade for Radix + HiSparse.

    Ordinary allocator operations are L1-only. The CPU full-KV pool is sized
    for the same identity-indexed L1 namespace and owned by this facade. L0 is
    a private cache resource used only by the HiSparse coordinator.
    """

    def __init__(
        self,
        l0_capacity: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kvcache: HiSparseDSATokenToKVPool,
        need_sort: bool,
        host_to_device_ratio: int = 2,
        host_pool: Optional[MLATokenToKVPoolHost] = None,
    ):
        self.l1_allocator = RadixHiSparseL1Allocator(
            l0_capacity=l0_capacity,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
            host_to_device_ratio=host_to_device_ratio,
        )
        self.l0_allocator = HiSparseL0SlotAllocator(
            l0_capacity=l0_capacity,
            l1_capacity=self.l1_allocator.l1_capacity,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )

        # Compatibility views for the existing coordinator/decode admission
        # code. The facade's public alloc/free API still delegates to L1 only.
        self.logical_attn_allocator = self.l1_allocator
        self.hisparse_attn_allocator = self.l0_allocator.l0_slot_allocator

        # Mirror the ordinary allocator attributes used by scheduler and Radix.
        # Their mutable state remains authoritative in l1_allocator.
        self.size = self.l1_allocator.size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort
        self.host_to_device_ratio = host_to_device_ratio
        self.compress_ratio = 1

        self._owns_l1_host_pool = host_pool is None
        self._l1_host_pool_destroyed = False
        if host_pool is None:
            from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

            host_pool = MLATokenToKVPoolHost(
                device_pool=kvcache,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=page_size,
                layout="layer_first",
                override_kv_cache_dim=kvcache.kv_cache_dim,
            )
        self.l1_host_pool = RadixHiSparseL1HostPool(
            storage_pool=host_pool,
            l1_allocator=self.l1_allocator,
        )

        # Paged L1 indices include a leading padding page. HostKVCache also
        # allocates an extra page, so identity addressing is valid end-to-end.
        required_host_rows = self.l1_capacity + self.page_size
        if self.l1_host_pool.storage_size < required_host_rows:
            raise ValueError(
                "CPU L1 pool is too small for identity-addressed L1 slots: "
                f"need {required_host_rows}, got {self.l1_host_pool.storage_size}"
            )

    @property
    def l1_capacity(self) -> int:
        return self.l1_allocator.l1_capacity

    @property
    def l0_capacity(self) -> int:
        return self.l0_allocator.size

    @property
    def size_full(self) -> int:
        return self.l1_capacity

    @property
    def num_pages(self) -> int:
        return self.l1_allocator.num_pages

    @property
    def free_pages(self) -> torch.Tensor:
        return self.l1_allocator.free_pages

    @property
    def release_pages(self) -> torch.Tensor:
        return self.l1_allocator.release_pages

    def available_size(self) -> int:
        """Return L1 availability only; L0 never gates Radix admission."""
        return self.l1_allocator.available_size()

    def l1_available_size(self) -> int:
        return self.l1_allocator.available_size()

    def l0_available_size(self) -> int:
        return self.l0_allocator.available_size()

    def get_kvcache(self) -> HiSparseDSATokenToKVPool:
        return self._kvcache

    def get_l1_host_pool(self) -> RadixHiSparseL1HostPool:
        return self.l1_host_pool

    def alloc(self, need_size: int):
        return self.l1_allocator.alloc(need_size)

    def alloc_extend(self, *args, **kwargs):
        return self.l1_allocator.alloc_extend(*args, **kwargs)

    def alloc_decode(self, *args, **kwargs):
        return self.l1_allocator.alloc_decode(*args, **kwargs)

    def alloc_l1_only(self, *args, **kwargs):
        return self.l1_allocator.alloc_l1_only(*args, **kwargs)

    def alloc_logical_only(self, *args, **kwargs):
        return self.l1_allocator.alloc_l1_only(*args, **kwargs)

    def free(self, l1_indices: torch.Tensor) -> None:
        self.l1_allocator.free(l1_indices)

    def free_segment(self, l1_indices: torch.Tensor, *, start_pos: int) -> None:
        self.l1_allocator.free_segment(l1_indices, start_pos=start_pos)

    def free_group_begin(self) -> None:
        self.l1_allocator.free_group_begin()

    def free_group_end(self) -> None:
        self.l1_allocator.free_group_end()

    def merge_and_sort_free(self) -> None:
        self.l1_allocator.merge_and_sort_free()

    def full_kv_host_locs(self, l1_indices: torch.Tensor) -> torch.Tensor:
        return self.l1_allocator.full_kv_host_locs(l1_indices)

    def index_k_device_locs(self, l1_indices: torch.Tensor) -> torch.Tensor:
        return self.l1_allocator.index_k_device_locs(l1_indices)

    def get_last_loc_compressed(self, last_locs: torch.Tensor) -> torch.Tensor:
        return self.l1_allocator.get_last_loc_compressed(last_locs)

    def alloc_device_buffer(self, l1_indices: torch.Tensor, need_size: int):
        """Compatibility entry used by the shared coordinator hot path."""
        return self.l0_allocator.acquire_request_l0_buffer(l1_indices, need_size)

    def free_hisparse_indices(self, l0_indices: torch.Tensor) -> None:
        """Release L0 rows without changing L1 ownership."""
        self.l0_allocator.free(l0_indices)

    def free_hisparse(self, l1_indices: torch.Tensor) -> None:
        """Release only transient L0 write routes for the supplied L1 rows."""
        self.l0_allocator.release_write_locs(l1_indices)

    def release_l1_write_locs(self, l1_indices: torch.Tensor) -> None:
        self.l0_allocator.release_write_locs(l1_indices)

    def clear(self) -> None:
        self.l1_allocator.clear()
        self.l0_allocator.clear()

    def resize(self, config) -> None:
        required_host_rows = (
            config.max_total_num_tokens * self.host_to_device_ratio + self.page_size
        )
        if self.l1_host_pool.storage_size < required_host_rows:
            raise ValueError(
                "CPU L1 pool cannot cover resized identity-addressed L1 slots: "
                f"need {required_host_rows}, got {self.l1_host_pool.storage_size}"
            )
        self.l1_allocator.resize(config)
        self.l0_allocator.resize(config)
        self.size = self.l1_allocator.size

    def destroy(self) -> None:
        if self._owns_l1_host_pool and not self._l1_host_pool_destroyed:
            self.l1_host_pool._storage_pool.destroy()
            self._l1_host_pool_destroyed = True


def is_radix_hisparse_allocator(
    allocator: BaseTokenToKVPoolAllocator,
) -> TypeGuard[RadixHiSparseTokenToKVPoolAllocator]:
    return isinstance(allocator, RadixHiSparseTokenToKVPoolAllocator)
