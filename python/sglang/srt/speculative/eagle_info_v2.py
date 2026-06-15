from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_alloc_reserve_per_decode,
    get_last_loc,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    assign_extend_cache_locs_func as _assign_extend_cache_locs_func_triton,
)
from sglang.srt.speculative.triton_ops.eagle import (
    fill_bonus_tokens as fill_bonus_tokens,
)
from sglang.srt.utils.common import is_cpu

_is_cpu = is_cpu()

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_info import EagleDraftInput

if _is_cpu:
    from sgl_kernel import (
        assign_extend_cache_locs_cpu,
        fill_bonus_tokens_cpu,
    )


def fill_bonus_tokens_func(accept_tokens, accept_lens, bonus_tokens, accept_stride, bs):
    if _is_cpu:
        fill_bonus_tokens_cpu(accept_tokens, accept_lens, bonus_tokens, accept_stride)
    else:
        fill_bonus_tokens[(bs,)](
            accept_tokens, accept_lens, bonus_tokens, accept_stride
        )


def assign_extend_cache_locs_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
) -> torch.Tensor:
    if _is_cpu:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int64,
            device=device,
        )
        assign_extend_cache_locs_cpu(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
        )
        return out_cache_loc
    else:
        return _assign_extend_cache_locs_func_triton(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            batch_size,
            draft_token_num,
            device,
        )


@dataclass
class EagleDraftInputV2Mixin:
    def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):
        batch.maybe_evict_swa()

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()

        # Accumulate penalty
        # This is a relaxed version of penalties for speculative decoding.
        if batch.sampling_info.penalizer_orchestrator.is_required:
            output_ids = torch.tensor(
                [
                    (
                        req.output_ids[-1]
                        if len(req.output_ids)
                        else req.origin_input_ids[-1]
                    )
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                output_ids
            )

        page_size = batch.token_to_kv_pool_allocator.page_size
        double_alloc = get_alloc_reserve_per_decode()

        cur_kv_lens = [0] * bs
        nxt_kv_lens = [0] * bs
        num_needed_tokens = 0
        for i, r in enumerate(batch.reqs):
            cur = r.kv_allocated_len
            # max(cur, ...) clamps so adaptive downswitch (smaller alloc_len_per_decode)
            # cannot make nxt < cur and corrupt allocator state. kv_committed_len lags
            # batch.seq_lens by ~1 verify in overlap mode, so we react to adaptive
            # switches one batch later than a seq_lens-based baseline; the 2*alloc
            # over-allocation buffer absorbs that lag.
            nxt = max(cur, r.kv_committed_len + double_alloc)
            cur_kv_lens[i] = cur
            nxt_kv_lens[i] = nxt
            num_needed_tokens += nxt - cur
            r.kv_allocated_len = nxt
            r.decode_batch_idx += 1
            # Pre-claim bonus slot here (like normal decode); resolve subtracts 1.
            r.kv_committed_len += 1

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens, dtype=torch.int32, device="cpu")

        # Fail fast if the page>1 + topk>1 draft over-allocation
        # (get_alloc_reserve_per_decode) outgrows the req_to_token row: the write below
        # would OOB and free would leak KV. The row is widened to hold it in _init_pools
        # (PR #26972); fail here with a clear error, not on a later cryptic CUDA assert.
        from sglang.srt.server_args import get_global_server_args

        if page_size > 1 and (get_global_server_args().speculative_eagle_topk or 1) > 1:
            max_alloc_len = int(nxt_kv_lens_cpu.max())
            row_width = batch.req_to_token_pool.req_to_token.shape[1]
            assert max_alloc_len <= row_width, (
                f"spec v2 page>1 topk>1 draft over-allocation ({max_alloc_len}) exceeds "
                f"req_to_token row width ({row_width}); page_size={page_size}. Widen the "
                f"row to hold committed + get_alloc_reserve_per_decode (PR #26972)."
            )

        # non_blocking H2D: a blocking .to() syncs the schedule stream, which the WAR
        # barrier has chained to the prev forward -> host stalls a full forward.
        cur_kv_lens_device = cur_kv_lens_cpu.to(device=batch.device, non_blocking=True)
        nxt_kv_lens_device = nxt_kv_lens_cpu.to(device=batch.device, non_blocking=True)
        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens_device,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                cur_kv_lens_device,
                cur_kv_lens_cpu,
                nxt_kv_lens_device,
                nxt_kv_lens_cpu,
                last_loc,
                num_needed_tokens,
            )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_device,
            nxt_kv_lens_device,
            out_cache_loc,
            bs,
        )
