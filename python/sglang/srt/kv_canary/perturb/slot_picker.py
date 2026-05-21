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

from sglang.srt.kv_canary.radix_cache_walker import walk_radix_cache_for_canary

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class ReqToTokenEntry:
    req_pool_idx: int
    position: int
    value: int


def collect_active_slots(
    *,
    forward_batch: "ForwardBatch",
    req_to_token_pool: "ReqToTokenPool",
    exclude_out_cache_loc: bool = True,
) -> list[ReqToTokenEntry]:
    """Collect every (req_pool_idx, position, value) triple for currently-active reqs.

    Excludes slots in ``forward_batch.out_cache_loc`` when ``exclude_out_cache_loc=True``
    so a slot the current forward is about to write isn't picked (write race).
    """
    req_pool_indices = forward_batch.req_pool_indices
    seq_lens = forward_batch.seq_lens
    if req_pool_indices is None or seq_lens is None:
        return []

    req_to_token = req_to_token_pool.req_to_token
    if not isinstance(req_to_token, torch.Tensor) or req_to_token.numel() == 0:
        return []

    excluded: set[int] = set()
    if exclude_out_cache_loc:
        out_cache_loc = forward_batch.out_cache_loc
        if out_cache_loc is not None:
            valid_num_tokens = forward_batch.num_token_non_padded_cpu
            if valid_num_tokens is None:
                valid_num_tokens = int(out_cache_loc.shape[0])
            excluded = set(
                int(x)
                for x in out_cache_loc[:valid_num_tokens].detach().to("cpu").tolist()
            )

    req_pool_indices_list = req_pool_indices.detach().to("cpu").tolist()
    seq_lens_list = seq_lens.detach().to("cpu").tolist()
    rows, cols = int(req_to_token.shape[0]), int(req_to_token.shape[1])

    candidates: list[ReqToTokenEntry] = []
    for req_pool_idx, seq_len in zip(req_pool_indices_list, seq_lens_list):
        req_pool_idx_int = int(req_pool_idx)
        seq_len_int = int(seq_len)
        if req_pool_idx_int < 0 or req_pool_idx_int >= rows:
            continue
        upper = min(seq_len_int, cols)
        if upper <= 0:
            continue
        row_values = req_to_token[req_pool_idx_int, :upper].detach().to("cpu").tolist()
        candidates.extend(
            ReqToTokenEntry(
                req_pool_idx=req_pool_idx_int,
                position=pos,
                value=value,
            )
            for pos, raw_value in enumerate(row_values)
            if (value := int(raw_value)) >= 0 and value not in excluded
        )
    return candidates


def pick_active_slot(
    *,
    forward_batch: "ForwardBatch",
    req_to_token_pool: "ReqToTokenPool",
    exclude_out_cache_loc: bool = True,
) -> Optional[ReqToTokenEntry]:
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
    walk_result = walk_radix_cache_for_canary(
        radix_cache=radix_cache,
        unlocked_only=True,
    )
    slot_tensor = walk_result.slot_indices
    if slot_tensor.numel() == 0:
        return None
    valid: list[int] = [
        slot for raw_slot in slot_tensor.tolist() if (slot := int(raw_slot)) >= 0
    ]
    if not valid:
        return None
    pick = int(torch.randint(0, len(valid), (1,)).item())
    return valid[pick]
