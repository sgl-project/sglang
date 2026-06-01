from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


class FullAttentionKvReservation:
    """A block of full-attention KV slots popped out of the allocator to
    simulate memory pressure. ``release`` returns them to the free list."""

    def __init__(self, allocator, indices: torch.Tensor) -> None:
        self._allocator = allocator
        self._indices: Optional[torch.Tensor] = indices

    def release(self) -> None:
        assert (
            self._indices is not None
        ), "full-attention KV reservation already released"
        self._allocator.free(self._indices)
        self._indices = None


def _full_attn_allocator(ctx: "ScriptedContext"):
    allocator = ctx._scheduler.token_to_kv_pool_allocator
    assert isinstance(allocator, SWATokenToKVPoolAllocator), (
        "reserving full-attention KV requires a hybrid-SWA allocator "
        f"(got {type(allocator).__name__})"
    )
    return allocator.full_attn_allocator


def full_attention_available_size(ctx: "ScriptedContext") -> int:
    return _full_attn_allocator(ctx).available_size()


def reserve_full_attention_kv(
    ctx: "ScriptedContext", num_tokens: int
) -> FullAttentionKvReservation:
    allocator = _full_attn_allocator(ctx)
    indices = allocator.alloc(num_tokens)
    assert indices is not None, (
        f"cannot reserve {num_tokens} full-attention KV tokens; only "
        f"{allocator.available_size()} available"
    )
    return FullAttentionKvReservation(allocator, indices)
