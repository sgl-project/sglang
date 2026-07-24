from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.kv_canary.verify import VerifyPlan
from sglang.srt.kv_canary.radix_cache_walker import walk_radix_cache_for_canary

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


def build_verify_plan_radix_sweep(
    *,
    radix_cache: BasePrefixCache,
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    unlocked_only: bool = False,
) -> VerifyPlan:
    """Build a sweep VerifyPlan directly from the radix-cache walker.

    The walker covers every slot held by the radix tree. There is no exclusion for slots also owned
    by running requests because the overlap with per-forward HEAD/TAIL coverage is harmless
    redundancy. The helper applies the SWA LUT before writing the plan.
    """
    device = radix_cache.req_to_token_pool.req_to_token.device

    walk_result = walk_radix_cache_for_canary(
        radix_cache=radix_cache,
        unlocked_only=unlocked_only,
    )
    slot_indices = walk_result.slot_indices.to(device)
    positions = walk_result.positions.to(device)
    prev_slot_indices = walk_result.prev_slot_indices.to(device)

    if swa_window_size > 0:
        assert (
            full_to_swa_index_mapping is not None
        ), "full_to_swa_index_mapping is required when SWA is enabled"
        slot_indices = _swa_translate(
            indices=slot_indices, lut=full_to_swa_index_mapping
        )
        prev_slot_indices = _swa_translate(
            indices=prev_slot_indices, lut=full_to_swa_index_mapping
        )

    num_valid = int(slot_indices.shape[0])
    verify_plan = VerifyPlan.allocate(verify_capacity=max(1, num_valid), device=device)

    verify_plan.verify_slot_indices[:num_valid].copy_(slot_indices)
    verify_plan.verify_expected_positions[:num_valid].copy_(positions)
    verify_plan.verify_prev_slot_indices[:num_valid].copy_(prev_slot_indices)
    verify_plan.verify_num_valid.fill_(num_valid)
    verify_plan.enable.fill_(1)

    return verify_plan


def _swa_translate(
    *,
    indices: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    # 0 is both SWAKVPool's evicted-sentinel and the kernel's kTokenToKvSlotPadding,
    # so evicted indices propagate as the canonical "no real slot" value and the
    # verify kernel handles them.
    if indices.numel() == 0:
        return indices
    lut_dev = lut.to(indices.device).to(torch.int64)
    anchor_mask = indices < 0
    safe = torch.where(anchor_mask, torch.zeros_like(indices), indices).to(torch.int64)
    looked_up = lut_dev[safe]
    return torch.where(anchor_mask, indices.to(torch.int64), looked_up)
