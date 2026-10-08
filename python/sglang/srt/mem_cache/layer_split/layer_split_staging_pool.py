"""Fixed full-page component slabs exposed to the L3 storage backend.

KV and indexer are independent objects. Window ownership and safe buffer reuse
belong to the transfer engine; this module has no leases or free-list state.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from sglang.srt.mem_cache.layer_split.layer_split_plan import EXCHANGE_COMPONENTS
from sglang.srt.mem_cache.layer_split.layer_split_utils import STAGE_NAMES


class StagingComponent:
    """One component's slab, exposing the host-pool contract storage expects.

    ``get_page_buffer_meta`` mirrors the method every real host pool already
    implements, so a storage backend can address this slab through the same
    zero-copy path once the transfer descriptor can name a physical buffer
    separately from its logical pool.
    """

    def __init__(
        self,
        *,
        name: str,
        num_pages: int,
        layer_count: int,
        page_size: int,
        row_bytes: int,
        pin_memory: bool,
    ) -> None:
        for label, value in (
            ("num_pages", num_pages),
            ("layer_count", layer_count),
            ("page_size", page_size),
            ("row_bytes", row_bytes),
        ):
            if value <= 0:
                raise ValueError(f"{name}: {label} must be positive, got {value}")

        self.name = name
        self.layer_count = layer_count
        self.page_size = page_size
        self.row_bytes = row_bytes
        # uint8 throughout: these bytes are moved and stored verbatim, never
        # interpreted, so the element type would only invite dtype conversions.
        self.buffer = torch.empty(
            (num_pages, layer_count, page_size, row_bytes),
            dtype=torch.uint8,
            pin_memory=pin_memory,
        )

    @property
    def object_bytes(self) -> int:
        """Bytes of one page's L3 object, which is the whole slot.

        Also the stride between slots: there is no padding, so the two cannot
        drift apart and no caller has to pick the right one.
        """
        return self.layer_count * self.page_size * self.row_bytes

    def get_page_buffer_meta(
        self, indices: Sequence[int]
    ) -> Tuple[List[int], List[int]]:
        """``(pointers, sizes)`` for the given slots, one entry per slot.

        Mirrors the method every real host pool implements, so a storage backend
        can address this slab through the same zero-copy path.
        """
        pointers: List[int] = []
        sizes: List[int] = []
        size = self.object_bytes
        for index in indices:
            slot = int(index)
            if not 0 <= slot < self.buffer.shape[0]:
                raise ValueError(f"{self.name}: slot {slot} out of range")
            pointers.append(self.buffer[slot].data_ptr())
            sizes.append(size)
        return pointers, sizes

    def get_hybrid_pool_buffer(self) -> List[torch.Tensor]:
        """The registrable backing tensors for the storage backend.

        ``_iter_host_pool_buffers`` looks for this method first and falls back to
        a ``kv_buffer`` attribute; a slab is one contiguous tensor, so it returns
        exactly that for registration by the backend.
        """
        return [self.buffer]


class FixedStageBuffer:
    """One direction's component slabs and per-round local-shard scratch.

    Full-page position = round.index. No allocator, leases, banks or ordinal-to-
    slot dictionary. A short window simply uses fewer positions. Both directions
    reuse this single window only after all of its I/O and L2 copies finish.
    """

    def __init__(self, stage, config, layer_counts, row_bytes, rank, pin):
        self.stage = stage
        self.max_rounds = config.pages_per_rank_per_window
        self.names = {
            c: f"l3_shared_{STAGE_NAMES[stage]}_{c}" for c in EXCHANGE_COMPONENTS
        }
        self.components = {
            c: StagingComponent(
                name=self.names[c],
                num_pages=self.max_rounds,
                layer_count=sum(layer_counts[c]),
                page_size=config.page_size,
                row_bytes=row_bytes[c],
                pin_memory=pin,
            )
            for c in EXCHANGE_COMPONENTS
        }
        self.shard_bytes = {
            c: tuple(n * config.page_size * row_bytes[c] for n in layer_counts[c])
            for c in EXCHANGE_COMPONENTS
        }
        self.scratch = {
            c: torch.empty(
                self.max_rounds,
                config.shard_size,
                self.shard_bytes[c][rank],
                dtype=torch.uint8,
            )
            for c in EXCHANGE_COMPONENTS
        }
