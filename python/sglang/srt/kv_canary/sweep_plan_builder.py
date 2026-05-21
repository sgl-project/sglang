from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.srt.kv_canary.radix_cache_walker import walk_radix_cache_for_canary

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


def fill_verify_plan_radix_sweep(
    *,
    radix_cache: "BasePrefixCache",
    verify_plan_out: VerifyPlan,
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
) -> int:
    """Fill the sweep VerifyPlan directly from the radix-cache walker.

    The walker covers every slot held by the radix tree. There is no exclusion for slots also owned
    by running requests because the overlap with per-forward HEAD/TAIL coverage is harmless
    redundancy. The helper applies the SWA LUT before writing the plan.

    Returns the number of active verify entries.
    """
    device = radix_cache.req_to_token_pool.req_to_token.device

    walk_result = walk_radix_cache_for_canary(
        radix_cache=radix_cache,
    )
    slot_indices = walk_result.slot_indices.to(device)
    positions = walk_result.positions.to(device)
    prev_slot_indices = walk_result.prev_slot_indices.to(device)

    if swa_window_size > 0:
        assert (
            full_to_swa_index_mapping is not None
        ), "full_to_swa_index_mapping is required when SWA is enabled"
        slot_indices = _swa_translate(
            indices=slot_indices,
            lut=full_to_swa_index_mapping,
        )
        prev_slot_indices = _swa_translate(
            indices=prev_slot_indices,
            lut=full_to_swa_index_mapping,
        )

    num_valid = int(slot_indices.shape[0])
    capacity = int(verify_plan_out.verify_slot_indices.shape[0])
    if num_valid > capacity:
        raise RuntimeError(
            f"kv-canary: radix-walker emitted {num_valid} sweep verify entries, exceeding "
            f"pre-allocated sweep_verify_capacity={capacity}; raise the sweep capacity in "
            f"CanaryLaunchCapacities.from_args"
        )

    verify_plan_out.verify_slot_indices[:num_valid].copy_(slot_indices)
    verify_plan_out.verify_positions[:num_valid].copy_(positions)
    verify_plan_out.verify_prev_slot_indices[:num_valid].copy_(prev_slot_indices)
    verify_plan_out.verify_num_valid.fill_(num_valid)
    verify_plan_out.enable.fill_(1)

    return num_valid


def _swa_translate(
    *,
    indices: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    if indices.numel() == 0:
        return indices
    lut_dev = lut.to(indices.device).to(torch.int64)
    sentinel_mask = indices < 0
    safe = torch.where(sentinel_mask, torch.zeros_like(indices), indices).to(
        torch.int64
    )
    translated = lut_dev[safe]
    return torch.where(
        sentinel_mask, indices.to(torch.int64), translated.to(torch.int64)
    )
