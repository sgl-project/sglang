"""Slot pickers used by perturb hooks to choose a target slot to corrupt.

Two picking modes:

- :func:`pick_active_slot` (via :func:`collect_active_slots`): random pick from
  slots currently held by an active req, modeling "corrupt KV that production
  is about to read." Used by perturb_req_to_token and perturb_real_kv_used.
- :func:`pick_orphan_slot`: random pick from slots cached in the radix tree
  but not locked by any active req, modeling "corrupt KV that will only be
  read much later via prefix reuse." Used by perturb_real_kv_unused_cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_canary.plan_input import walk_radix_cache_for_canary

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveSlotTarget:
    req_pool_idx: int
    position: int
    slot: int


def collect_active_slots(
    *,
    forward_batch: "ForwardBatch",
    req_to_token_pool: "ReqToTokenPool",
    exclude_out_cache_loc: bool = True,
) -> list[ActiveSlotTarget]:
    """Collect every (req_pool_idx, position, slot) triple for currently-active reqs.

    Excludes slots in ``forward_batch.out_cache_loc`` when ``exclude_out_cache_loc=True``
    so a slot the current forward is about to write isn't picked (write race).
    """
    req_pool_indices = forward_batch.req_pool_indices
    seq_lens = forward_batch.seq_lens
    if req_pool_indices is None or seq_lens is None:
        return []

    table = req_to_token_pool.req_to_token
    if not isinstance(table, torch.Tensor) or table.numel() == 0:
        return []

    excluded: set[int] = set()
    if exclude_out_cache_loc:
        out_cache_loc = forward_batch.out_cache_loc
        if out_cache_loc is not None:
            excluded = set(int(x) for x in out_cache_loc.detach().to("cpu").tolist())

    req_pool_indices_list = req_pool_indices.detach().to("cpu").tolist()
    seq_lens_list = seq_lens.detach().to("cpu").tolist()
    rows, cols = int(table.shape[0]), int(table.shape[1])

    candidates: list[ActiveSlotTarget] = []
    for req_pool_idx, seq_len in zip(req_pool_indices_list, seq_lens_list):
        req_pool_idx_int = int(req_pool_idx)
        seq_len_int = int(seq_len)
        if req_pool_idx_int < 0 or req_pool_idx_int >= rows:
            continue
        upper = min(seq_len_int, cols)
        if upper <= 0:
            continue
        row_slots = table[req_pool_idx_int, :upper].detach().to("cpu").tolist()
        for pos, raw_slot in enumerate(row_slots):
            slot = int(raw_slot)
            if slot < 0:
                continue
            if slot in excluded:
                continue
            candidates.append(
                ActiveSlotTarget(
                    req_pool_idx=req_pool_idx_int,
                    position=pos,
                    slot=slot,
                )
            )
    return candidates


def pick_active_slot(
    *,
    forward_batch: "ForwardBatch",
    req_to_token_pool: "ReqToTokenPool",
    exclude_out_cache_loc: bool = True,
) -> Optional[ActiveSlotTarget]:
    """Random pick from ``collect_active_slots`` output. Returns None if no candidate."""
    candidates = collect_active_slots(
        forward_batch=forward_batch,
        req_to_token_pool=req_to_token_pool,
        exclude_out_cache_loc=exclude_out_cache_loc,
    )
    if not candidates:
        return None
    pick = int(torch.randint(0, len(candidates), (1,)).item())
    return candidates[pick]


def pick_orphan_slot(*, radix_cache: Optional["BasePrefixCache"]) -> Optional[int]:
    """Pick one random orphan slot (radix-cached but not currently locked by any active req).
    Returns None if radix_cache is None or no orphan slots exist."""
    if radix_cache is None:
        return None
    slot_tensor, _, _ = walk_radix_cache_for_canary(
        radix_cache=radix_cache,
        unlocked_only=True,
    )
    if slot_tensor.numel() == 0:
        return None
    valid: list[int] = []
    for raw_slot in slot_tensor.tolist():
        slot = int(raw_slot)
        if slot < 0:
            continue
        valid.append(slot)
    if not valid:
        return None
    pick = int(torch.randint(0, len(valid), (1,)).item())
    return valid[pick]
