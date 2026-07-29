from __future__ import annotations

from typing import Tuple

import torch

from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool


class DraftMambaStatePlane:
    """Per-request draft states living in the target's mamba slot index space.

    A speculative-draft worker with linear-attention layers keeps its recurrent
    states in its own MambaPool but owns no slot allocator: states are addressed
    by the mamba slot ids the target's HybridReqToTokenPool hands out, so slot
    lifecycle (alloc, radix reuse, free) stays entirely on the target side and a
    slot's draft states survive radix resurrection without copies. The plane
    only tracks whether its copy of each slot is current against the target's
    fresh-assignment generation; the caller must rebuild stale slots before
    reading them.
    """

    def __init__(
        self,
        target_req_to_token_pool: HybridReqToTokenPool,
        mamba_pool: MambaPool,
    ):
        target_size = target_req_to_token_pool.mamba_pool.size
        if mamba_pool.size != target_size:
            raise ValueError(
                "DraftMambaStatePlane must cover every target mamba slot. "
                f"draft pool size={mamba_pool.size}, target pool size={target_size}."
            )
        self.target_req_to_token_pool = target_req_to_token_pool
        self.mamba_pool = mamba_pool
        # -1 = never built, distinct from every target generation (>= 0).
        self.built_generation = torch.full(
            (target_size,),
            -1,
            dtype=torch.int64,
            device=target_req_to_token_pool.mamba_slot_generation.device,
        )

    def lookup(self, req_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mamba_indices, current_mask) for a batch of req_pool_indices.

        current_mask[i] is False when the target allocator recycled the slot
        after the draft last built states there.
        """
        mamba_indices = self.target_req_to_token_pool.get_mamba_indices(req_indices)
        target_generation = self.target_req_to_token_pool.mamba_slot_generation
        current_mask = (
            self.built_generation[mamba_indices] == target_generation[mamba_indices]
        )
        return mamba_indices, current_mask

    def mark_built(self, mamba_indices: torch.Tensor) -> None:
        """Record that the draft states at these slots match the live requests."""
        target_generation = self.target_req_to_token_pool.mamba_slot_generation
        self.built_generation[mamba_indices] = target_generation[mamba_indices]
