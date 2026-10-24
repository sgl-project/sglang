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

import functools
from typing import Optional, Tuple

import torch

from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.page_interleave import PageInterleavePlacement
from sglang.srt.mem_cache.pool_host.base import synchronized


class PageInterleaveHostPoolMixin:
    """Owner-classed host slots + the per-rank narrowing of a transfer."""

    def __init__(
        self, buffer_info, *args, placement: PageInterleavePlacement, **kwargs
    ):
        # Set before super().__init__: the base constructor calls clear().
        self.placement = placement
        self.shard_rank = placement.spec.shard_rank
        self.shard_size = placement.spec.shard_size
        super().__init__(buffer_info, *args, **kwargs)
        assert self.page_size == self.placement.spec.page_size, (
            "the sharded host pool pages must match the device's physical page"
        )

    @property
    def logical_size(self) -> int:
        return self.size * self.shard_size

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
        self, need_size: int, start_class: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        assert start_class is not None, (
            "sharded host slots must follow the owners of the device pages "
            "they back up; pass the first page's owner class"
        )
        return self._slots.alloc_cyclic(start_class=start_class, need_size=need_size)

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


@functools.cache
def page_interleave_host_pool_cls(host_pool_cls: type) -> type:
    """``host_pool_cls`` striped like a page-interleave device pool."""
    return type(
        f"PageInterleave{host_pool_cls.__name__}",
        (PageInterleaveHostPoolMixin, host_pool_cls),
        {},
    )
