from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.srt.kv_canary.radix_cache_walker import walk_radix_cache_for_canary

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


def build_verify_plan_radix_sweep(
    *,
    radix_cache: "BasePrefixCache",
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
        slot_indices, prev_slot_indices, positions = _swa_translate_sweep_plan(
            slot_indices=slot_indices,
            prev_slot_indices=prev_slot_indices,
            positions=positions,
            lut=full_to_swa_index_mapping,
        )

    num_valid = int(slot_indices.shape[0])
    verify_plan = VerifyPlan.allocate(verify_capacity=max(1, num_valid), device=device)

    verify_plan.verify_slot_indices[:num_valid].copy_(slot_indices)
    verify_plan.verify_expected_positions[:num_valid].copy_(positions)
    verify_plan.verify_prev_slot_indices[:num_valid].copy_(prev_slot_indices)
    verify_plan.verify_num_valid.fill_(num_valid)
    verify_plan.enable.fill_(1)

    return verify_plan


def _swa_translate_sweep_plan(
    *,
    slot_indices: torch.Tensor,
    prev_slot_indices: torch.Tensor,
    positions: torch.Tensor,
    lut: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # full_to_swa_index_mapping uses 0 as the "SWA-evicted / unmapped" sentinel
    # (see SWAKVPool.free_swa: writes 0 on free). SWA-pool slot 0 is the
    # padding/null slot and is never allocated, so its canary fields stay all
    # zero -- if we let an evicted FULL index translate to SWA slot 0, the
    # verify kernel would read those zeros and flag every chain-link to an
    # evicted ancestor as a verify_chain_hash mismatch (expected_aux=0 flood).
    #
    # Drop rows where either the current slot or the prev slot lands on the
    # 0 sentinel after translation. The walker's own -1 anchor on root edges
    # is preserved (chain hash anchor); only LUT-introduced 0s are filtered.
    if slot_indices.numel() == 0:
        return slot_indices, prev_slot_indices, positions
    lut_dev = lut.to(slot_indices.device).to(torch.int64)

    def translate(x: torch.Tensor) -> torch.Tensor:
        anchor_mask = x < 0
        safe = torch.where(anchor_mask, torch.zeros_like(x), x).to(torch.int64)
        looked_up = lut_dev[safe]
        return torch.where(anchor_mask, x.to(torch.int64), looked_up)

    slot_swa = translate(slot_indices)
    prev_swa = translate(prev_slot_indices)

    valid_slot = slot_swa != 0
    valid_prev = (prev_slot_indices < 0) | (prev_swa != 0)
    keep = valid_slot & valid_prev

    return slot_swa[keep], prev_swa[keep], positions[keep]
