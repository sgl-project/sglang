"""HiCache host pool for logical-page KV sharding (mem_cache/page_interleave.py).

The host pool is striped exactly like its device pool. Host slots live in the
same widened logical index space and follow the same placement: logical host
page ``h`` is owned by rank ``h % N`` and is that rank's local host page
``h // N``. Host slots come from the device's own owner-classed allocator, and
every backup / load-back pairs device and host pages of the same owner, so a
rank's share of any transfer is exactly its own pages. Each shard rank
therefore moves only its stripe between its own HBM and host memory, all ranks
in parallel, with no cross-rank traffic.

``size`` stays the per-rank row count (the ratio applied to the per-rank
device pool), so per-rank host memory matches the unsharded case while the
logical host capacity grows with the shard group, like the device's.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

import torch

from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.page_interleave import PageInterleavePlacement
from sglang.srt.mem_cache.pool_host.base import HostKVCache, synchronized
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost


class PageInterleaveHostPool:
    """Compose owner-classed slots with a native host pool's buffers/transfers.

    The native pool is fully constructed before this allocation facade is
    installed; its constructor and clear() never dispatch into sharding code.
    """

    def __init__(
        self,
        host_pool: HostKVCache,
        placement: PageInterleavePlacement,
        *,
        kv_dtype: torch.dtype,
    ):
        self.host_pool = host_pool
        self.placement = placement
        # The device pool's KV dtype; the host stores every fp8 format as uint8.
        self.kv_dtype = kv_dtype
        self.shard_rank = placement.spec.shard_rank
        self.shard_size = placement.spec.shard_size
        assert self.page_size == self.placement.spec.page_size, (
            "the sharded host pool pages must match the device's physical page"
        )
        self.clear()

    def __getattr__(self, name: str):
        # Layout, buffers, transfer kernels and their lifecycle remain owned by
        # the concrete native pool. Only slot ownership is managed here.
        return getattr(object.__getattribute__(self, "host_pool"), name)

    @property
    def logical_size(self) -> int:
        return self.size * self.shard_size

    @property
    def storage_format_tag(self) -> str:
        # Storage can be read by a different owner, or a different shard group.
        # Name the page bytes, excluding rank and pool capacity. Shape, dtype
        # and layer range must agree even when two formats have equal sizes.
        if isinstance(self.host_pool, MLATokenToKVPoolHost):
            head_geometry = (
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self.kv_cache_dim,
            )
        else:
            head_geometry = (self.head_num, self.head_dim, self.v_head_dim)
        geometry = (
            str(self.kv_dtype),
            str(self.dtype),
            self.page_size,
            self.start_layer,
            self.end_layer,
            self.layer_num,
            *head_geometry,
        )
        fingerprint = hashlib.sha256(repr(geometry).encode()).hexdigest()[:16]
        return f"kvshard-v1-{self.layout}-{fingerprint}"

    @synchronized
    def clear(self):
        # The device's classed allocator over this rank's host rows; like the
        # device, it leaves local page 0 unused.
        self._slots = PageInterleavePoolAllocator(
            size=self.size - self.page_size,
            physical_page_size=self.page_size,
            shard_size=self.shard_size,
            dtype=torch.int64,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )

    def available_size(self) -> int:
        return self._slots.available_size()

    @synchronized
    def alloc(
        self, need_size: int, start_owner: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        assert start_owner is not None, (
            "sharded host slots must follow the owners of the device pages "
            "they back up; pass the first page's owner class"
        )
        return self._slots.alloc_cyclic(start_class=start_owner, need_size=need_size)

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self._slots.free(indices.cpu())
        return len(indices)

    def owned_rows(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Narrow a logical (host, device) slot pairing to this rank's pages,
        as local (host, device) rows. Pairs are owner-matched, so ownership
        is read off the CPU host slots and the device side is a plain gather:
        no device-to-host sync."""
        owned = torch.nonzero(
            self.placement.local_mask(host_indices, self.shard_rank)
        ).squeeze(1)
        device_owned = device_indices[
            owned.to(device_indices.device, non_blocking=True)
        ]
        return (
            self.placement.local_index(host_indices[owned]),
            self.placement.local_index(device_owned),
        )

    def owned_storage_pages(
        self, host_indices: torch.Tensor
    ) -> Tuple[List[int], torch.Tensor]:
        """This rank's pages of a storage batch: their positions in the batch
        and their local host rows (whole pages, in batch order)."""
        mask = self.placement.local_mask(host_indices, self.shard_rank)
        positions = torch.nonzero(mask[:: self.page_size]).squeeze(1).tolist()
        return positions, self.placement.local_index(host_indices[mask])
