"""Overwriting freed KV bytes with zeros.

Returning a slot to the allocator is pure index bookkeeping: no ``free`` on any
allocator writes to a KV tensor, so a departed request's KV stays readable at
the freed slot until some later forward happens to overwrite it. This module
builds the byte-range plan that erases it.

Device-tier KV only, and only the pool families whose entire KV byte footprint
is covered by a flat list of row-major buffers. Anything else raises
``KvZeroizeUnsupported`` at build time rather than silently zeroizing part of a
page: a partial wipe is worse than none, because it reads as a guarantee.
"""

from __future__ import annotations

import math

import msgspec
import torch

from sglang.kernels.ops.kvcache.zero_kv_rows import (  # noqa: F401
    warmup_zero_kv_rows,
    zero_kv_rows,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


class KvZeroizeUnsupported(ValueError):
    """The configured pool family has KV bytes this module cannot enumerate."""


class PendingZeroize(msgspec.Struct):
    """One node's slots, held back from the allocator until they are cleared.

    ``free_indices`` are always FULL-pool slot ids (what the component action
    frees); ``zero_indices`` are the slots whose bytes to clear, already
    resolved -- for SWA those are the translated SWA slots, which have to be
    read before any free can rebind the mapping (see ``KvZeroizer.prepare``).
    """

    component_type: ComponentType
    free_indices: torch.Tensor
    zero_indices: torch.Tensor
    # True when zero_indices is a page-aligned run, so page ids come from a
    # stride slice instead of a sync-forcing torch.unique.
    page_aligned_run: bool


class KvZeroizePlan(msgspec.Struct, frozen=True, kw_only=True):
    """The pointer table for one pool: every buffer holding its KV bytes."""

    base_ptrs: torch.Tensor  # int64 [B]
    row_words: torch.Tensor  # int64 [B], int64-words per row
    max_row_words: int
    # 1 when a row is one token slot, page_size when a row is a whole page.
    tokens_per_row: int
    bytes_per_token: int

    def rows_for(
        self, indices: torch.Tensor, *, page_aligned_run: bool
    ) -> torch.Tensor:
        indices = indices.to(torch.int64)
        if self.tokens_per_row == 1:
            # Slot ids are row ids; a duplicate only re-stores its own row.
            return indices
        if page_aligned_run:
            # Page representatives by stride slice. The same trick as
            # PagedTokenToKVPoolAllocator.free_segment, and for the same
            # reason: torch.unique has a data-dependent output shape, so it
            # forces a device sync -- measured at ~48us on B200, four times
            # the kernel itself.
            return indices[:: self.tokens_per_row] // self.tokens_per_row
        # Arbitrary slot set (the translated SWA side): dedup, and pay the
        # sync, because duplicate rows would re-store a whole page each.
        return torch.unique(indices // self.tokens_per_row)

    def zeroize(self, indices: torch.Tensor, *, page_aligned_run: bool) -> None:
        """Clear the rows holding ``indices``.

        ``page_aligned_run`` is the caller's promise that, for a page-row pool,
        ``indices`` is a page-aligned run of whole pages; it buys the stride
        slice above instead of a synchronizing ``torch.unique``.
        """
        if indices.numel() == 0:
            return
        zero_kv_rows(
            base_ptrs=self.base_ptrs,
            row_words=self.row_words,
            row_ids=self.rows_for(indices, page_aligned_run=page_aligned_run),
            max_row_words=self.max_row_words,
        )


def _plan_from_buffers(
    buffers: list[torch.Tensor], *, tokens_per_row: int, device: torch.device
) -> KvZeroizePlan:
    if not buffers:
        raise KvZeroizeUnsupported("pool exposes no KV buffers to zeroize")
    base_ptrs, row_words = [], []
    for buf in buffers:
        if not buf.is_contiguous():
            raise KvZeroizeUnsupported(
                f"KV buffer {tuple(buf.shape)} is not contiguous; its rows are "
                "not a single byte span"
            )
        if buf.ndim < 1 or buf.shape[0] == 0:
            raise KvZeroizeUnsupported(
                f"KV buffer {tuple(buf.shape)} has no leading row dimension"
            )
        row_bytes = math.prod(buf.shape[1:]) * buf.element_size()
        if row_bytes % 8 or buf.data_ptr() % 8:
            raise KvZeroizeUnsupported(
                f"KV buffer row of {row_bytes} B at {buf.data_ptr():#x} is not "
                "int64-aligned"
            )
        base_ptrs.append(buf.data_ptr())
        row_words.append(row_bytes // 8)
    # An all-layers-alias layout would make the same bytes the target of L
    # different row offsets, so the table must address distinct allocations.
    if len(set(base_ptrs)) != len(base_ptrs):
        raise KvZeroizeUnsupported(
            "KV buffers alias one storage; per-buffer row offsets would clear "
            "the wrong bytes"
        )
    return KvZeroizePlan(
        base_ptrs=torch.tensor(base_ptrs, dtype=torch.int64, device=device),
        row_words=torch.tensor(row_words, dtype=torch.int64, device=device),
        max_row_words=max(row_words),
        tokens_per_row=tokens_per_row,
        bytes_per_token=sum(row_words) * 8 // tokens_per_row,
    )


def build_kv_zeroize_plan(pool) -> KvZeroizePlan:
    """The byte ranges of one (non-composite) KV pool, or raise."""
    device = torch.device(pool.device)

    # Exact type, not isinstance: every constructible subclass upstream either
    # holds KV bytes outside the buffers enumerated here
    # (MLATokenToKVPoolFP4.kv_scale_buffer, DSATokenToKVPool.index_key_cache)
    # or hands out per-layer views that overlap inside one shared page envelope
    # (UnifiedMHATokenToKVPool / UnifiedMLATokenToKVPool -- contiguous and
    # distinctly addressed, so none of the structural checks above catch them).
    if type(pool) is MLATokenToKVPool:
        # One buffer per layer, rows are token slots. V is a prefix slice of
        # the same latent row (get_value_buffer returns
        # kv_buffer[..., :kv_lora_rank]), so clearing K clears V.
        return _plan_from_buffers(list(pool.kv_buffer), tokens_per_row=1, device=device)

    if type(pool) is MHATokenToKVPool:
        if pool.k_scale_buffer is not None or pool.v_scale_buffer is not None:
            raise KvZeroizeUnsupported(
                "MHATokenToKVPool under this quantized KV recipe keeps "
                "per-block scales in buffers outside k_buffer/v_buffer; "
                "clearing only the data would leave the scales readable"
            )
        # A row is a whole page when the leading dim counts pages (hnd,
        # vectorized_5d) and a token slot for plain NHD -- the same test
        # _build_kv_buffer_descs uses.
        num_slots = pool.size + pool.page_size
        tokens_per_row = (
            pool.page_size
            if pool.k_buffer and pool.k_buffer[0].shape[0] * pool.page_size == num_slots
            else 1
        )
        return _plan_from_buffers(
            [*pool.k_buffer, *pool.v_buffer],
            tokens_per_row=tokens_per_row,
            device=device,
        )

    raise KvZeroizeUnsupported(f"{type(pool).__name__} is not supported")


class KvZeroizer:
    """Clears the KV bytes behind a set of freed slot indices."""

    def __init__(self, allocator, page_size: int):
        self._allocator = allocator
        self._page_size = page_size
        pool = allocator.get_kvcache()
        if isinstance(pool, SWAKVPool):
            if not isinstance(allocator, SWATokenToKVPoolAllocator):
                raise KvZeroizeUnsupported(
                    f"SWA pool served by {type(allocator).__name__}, which does "
                    "not expose the full->swa slot mapping"
                )
            self._plans = {
                ComponentType.FULL: build_kv_zeroize_plan(pool.full_kv_pool),
                ComponentType.SWA: build_kv_zeroize_plan(pool.swa_kv_pool),
            }
        else:
            self._plans = {ComponentType.FULL: build_kv_zeroize_plan(pool)}

    def supports(self, component_type: ComponentType) -> bool:
        return component_type in self._plans

    def bytes_per_token(self, component_type: ComponentType) -> int:
        return self._plans[component_type].bytes_per_token

    def prepare(
        self, component_type: ComponentType, indices: torch.Tensor
    ) -> torch.Tensor:
        """Resolve the slots to clear, while the mapping to read them is valid.

        A separate step from ``zeroize`` because the two must straddle the
        free: every component frees the node's FULL slot ids and the allocator
        translates, so the SWA side must be resolved BEFORE anything is freed.
        A freed FULL slot can be re-allocated, and ``alloc_extend`` rewrites
        ``full_to_swa_index_mapping`` for it, so a later translation would
        point at a live tenant's SWA slot.
        """
        if component_type is ComponentType.SWA:
            return self._to_swa_indices(indices)
        return indices

    def zeroize(
        self,
        component_type: ComponentType,
        indices: torch.Tensor,
        *,
        page_aligned_run: bool,
    ) -> None:
        self._plans[component_type].zeroize(indices, page_aligned_run=page_aligned_run)

    def _to_swa_indices(self, full_indices: torch.Tensor) -> torch.Tensor:
        """Mirror of ``SWATokenToKVPoolAllocator.free_swa``'s translation, which
        must run before that free clears the mapping."""
        alloc = self._allocator
        if self._page_size == 1:
            mapping_indices = full_indices
        else:
            mapping_indices = alloc._expand_to_full_pages(full_indices)
        swa_indices = alloc.full_to_swa_index_mapping[mapping_indices]
        return swa_indices[swa_indices > 0]


def build_kv_zeroizer(allocator, page_size: int, components) -> KvZeroizer:
    """Fails loud when any live component's KV cannot be enumerated."""
    zeroizer = KvZeroizer(allocator, page_size)
    unsupported = [ct.name for ct in components if not zeroizer.supports(ct)]
    if unsupported:
        raise KvZeroizeUnsupported(
            "KV zeroization cannot clear the KV of these tree components: "
            f"{', '.join(unsupported)}"
        )
    return zeroizer
