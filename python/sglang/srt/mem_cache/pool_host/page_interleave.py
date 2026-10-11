"""HiCache host pool for logical-page KV sharding (mem_cache/page_interleave.py).

The host pool is striped exactly like its device pool. Host slots live in the
same widened logical index space and follow the same placement: logical host
page ``h`` is owned by rank ``h % N`` and is that rank's local host page
``h // N``. The free lists are owner-classed like
``PageInterleavePoolAllocator``'s, and every backup / load-back pairs device and
host pages of the same class, so a rank's share of any transfer is exactly its
own pages. Each shard rank therefore moves only its stripe between its own HBM
and host memory, all ranks in parallel, with no cross-rank traffic.

``size`` stays the per-rank row count (the ratio applied to the per-rank
device pool), so per-rank host memory matches the unsharded case while the
logical host capacity grows with the shard group, like the device's.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from sglang.srt.mem_cache.allocator.page_interleave import (
    class_counts,
    pop_cyclic_pages,
)
from sglang.srt.mem_cache.pool_host.base import synchronized


class PageInterleaveHostPoolMixin:
    """Owner-classed host slots + the per-rank narrowing of a transfer."""

    def __init__(self, device_pool, *args, **kwargs):
        # Read before super().__init__: the base constructor calls clear().
        self.placement = device_pool.placement
        self.shard_rank = device_pool.shard_rank
        self.shard_size = device_pool.shard_size
        super().__init__(device_pool, *args, **kwargs)
        assert self.page_size == self.placement.spec.page_size, (
            "the sharded host pool pages must match the device's physical page"
        )

    @property
    def logical_size(self) -> int:
        return self.size * self.shard_size

    @synchronized
    def clear(self):
        # Class r holds the logical host pages owned by rank r: local host
        # page p is logical page p * N + r. No reserved page on the host.
        local_pages = torch.arange(self.size // self.page_size, dtype=torch.int64)
        self.class_free_pages = [
            local_pages * self.shard_size + r for r in range(self.shard_size)
        ]

    def available_size(self) -> int:
        # Min-class floor, as on the device: any cyclic run of this many
        # tokens fits whatever its start class.
        return (
            self.shard_size
            * self.page_size
            * min(len(free) for free in self.class_free_pages)
        )

    @synchronized
    def alloc(
        self, need_size: int, start_class: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Host slots for a backup whose device pages' owners cycle from
        ``start_class`` (taken mod N)."""
        assert start_class is not None, (
            "sharded host slots must follow the owners of the device pages "
            "they back up; pass the first page's owner class"
        )
        assert need_size % self.page_size == 0
        num_pages = need_size // self.page_size
        counts = class_counts(
            start_class=start_class, num_pages=num_pages, shard_size=self.shard_size
        )
        if any(n > len(free) for n, free in zip(counts, self.class_free_pages)):
            return None
        pages = pop_cyclic_pages(
            class_free_pages=self.class_free_pages,
            start_class=start_class,
            num_pages=num_pages,
        )
        offsets = torch.arange(self.page_size, dtype=torch.int64)
        return (pages[:, None] * self.page_size + offsets).reshape(-1)

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        pages = torch.unique(indices.cpu() // self.page_size)
        owners = pages % self.shard_size
        for r in range(self.shard_size):
            self.class_free_pages[r] = torch.cat(
                (self.class_free_pages[r], pages[owners == r])
            )
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
