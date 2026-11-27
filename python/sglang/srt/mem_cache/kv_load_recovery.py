"""Backend-independent ownership and commit contract for partial KV loads."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)


@dataclass
class KVLoadRecovery:
    """A private healthy suffix awaiting ordinary prefill of a missing span.

    Providers must finish all destination writes before transferring ownership.
    Commit is permitted only at the end of the missing span, after its forward.
    Abort releases only still-private suffix slots, never the caller's prefix.
    """

    span_start: int
    span_end: int
    restored_extent: int
    suffix_indices: torch.Tensor
    source_tier: Literal["storage", "host"] = "storage"

    def commit(self, req: Req, cache: BasePrefixCache) -> None:
        """Adopt the untouched suffix into a completed repair's request row.

        Raises RuntimeError without publication unless the request computed
        exactly through span_end. On success extend_end advances to the certified
        extent and the suffix belongs to the row, not this object.
        """
        if req.extend_end != self.span_end or req.kv.kv_committed_len != self.span_end:
            raise RuntimeError("KV hole repair must commit exactly at its span end")
        cache.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(self.span_end, self.restored_extent)),
            self.suffix_indices,
        )
        if self.source_tier == "storage":
            req.cached_tokens_storage += len(self.suffix_indices)
        else:
            req.cached_tokens_host += len(self.suffix_indices)
        req.kv.set_row_len(self.restored_extent)
        req.extend_end = self.restored_extent
        self.suffix_indices = self.suffix_indices[:0]
        req.kv_load_recovery = None
        logger.info(
            "KV load repair committed req=%s span=[%d,%d) restored_extent=%d",
            req.rid,
            self.span_start,
            self.span_end,
            self.restored_extent,
        )

    def abort(self, allocator: BaseTokenToKVPoolAllocator) -> None:
        """Free the private suffix exactly once; repeated aborts are no-ops."""
        allocator.free(self.suffix_indices)
        self.suffix_indices = self.suffix_indices[:0]


def prepare_partial_kv_load(
    *,
    loaded_start: int,
    loaded_indices: torch.Tensor,
    missing_span: tuple[int, int],
    page_size: int,
    allocator: BaseTokenToKVPoolAllocator,
    policy: Literal["repair", "recompute"],
) -> tuple[torch.Tensor, KVLoadRecovery | None]:
    """Partition a terminal, exclusive KV load into safe prefix and recovery.

    The provider certifies all loaded positions except missing_span and drains
    GPU delivery first. Validate bounds/alignment before freeing anything.
    Recompute frees everything from the hole onward. Repair frees only the
    missing allocation and transfers the healthy suffix to the returned owner.
    The caller continues to own the returned valid-prefix indices.
    Raises ValueError for invalid spans, alignment, or policy without freeing.
    """
    start, end = missing_span
    extent = loaded_start + len(loaded_indices)
    if (
        policy not in ("repair", "recompute")
        or page_size <= 0
        or not 0 <= loaded_start <= start < end <= extent
        or any(p % page_size for p in (loaded_start, start, end, extent))
    ):
        raise ValueError("Invalid partial KV load span, alignment, or policy")
    lo, hi = start - loaded_start, end - loaded_start
    prefix = loaded_indices[:lo]
    if policy == "recompute":
        allocator.free(loaded_indices[lo:])
        return prefix, None
    allocator.free(loaded_indices[lo:hi])
    return prefix, KVLoadRecovery(start, end, extent, loaded_indices[hi:])


def abort_kv_load_recovery(req: Req, cache: BasePrefixCache) -> None:
    """Release private recovery allocations on any request retirement path."""
    if req.kv_load_recovery is not None:
        req.kv_load_recovery.abort(cache.token_to_kv_pool_allocator)
        req.kv_load_recovery = None
