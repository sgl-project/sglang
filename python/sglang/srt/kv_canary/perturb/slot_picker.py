from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class ReqToTokenEntry:
    req_pool_idx: int
    position: int
    value: int
    seq_len: int = 0


def collect_active_slots(
    *,
    maybe_inaccurate_forward_batch: "ForwardBatch",
    req_to_token_pool: "ReqToTokenPool",
    exclude_out_cache_loc: bool = True,
) -> list[ReqToTokenEntry]:
    """Collect every (req_pool_idx, position, value) triple for currently-active reqs.

    Excludes slots in ``maybe_inaccurate_forward_batch.out_cache_loc`` when ``exclude_out_cache_loc=True``
    so a slot the current forward is about to write isn't picked (write race).
    """
    req_pool_indices = maybe_inaccurate_forward_batch.req_pool_indices
    seq_lens = maybe_inaccurate_forward_batch.seq_lens
    if req_pool_indices is None or seq_lens is None:
        return []

    req_to_token = req_to_token_pool.req_to_token
    if not isinstance(req_to_token, torch.Tensor) or req_to_token.numel() == 0:
        return []

    excluded: set[int] = set()
    if exclude_out_cache_loc:
        out_cache_loc = maybe_inaccurate_forward_batch.out_cache_loc
        if out_cache_loc is not None:
            valid_num_tokens = maybe_inaccurate_forward_batch.num_token_non_padded_cpu
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
                seq_len=seq_len_int,
            )
            for pos, raw_value in enumerate(row_values)
            if (value := int(raw_value)) >= 0 and value not in excluded
        )
    return candidates


def pick_out_cache_loc_slot(
    *, maybe_inaccurate_forward_batch: "ForwardBatch"
) -> Optional[int]:
    out_cache_loc = maybe_inaccurate_forward_batch.out_cache_loc
    if out_cache_loc is None:
        return None
    total = int(out_cache_loc.shape[0])
    if total <= 0:
        return None
    valid_num_tokens = maybe_inaccurate_forward_batch.num_token_non_padded_cpu
    if valid_num_tokens is None:
        valid_num_tokens = total
    valid_num_tokens = int(valid_num_tokens)
    if valid_num_tokens <= 0:
        return None
    pick = random.randrange(valid_num_tokens)
    slot = int(out_cache_loc[pick].item())
    if slot < 0:
        return None
    return slot
