from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dspark_components.dspark_info import RaggedVerifyWindow
from sglang.srt.speculative.dspark_components.kernels.compact_layout import (
    compact_row_index,
    compact_row_index_triton,
    compact_verify_ids,
    compact_verify_ids_triton,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_RAGGED_WINDOW.get()


class BuildRaggedVerifyWindow:
    @classmethod
    def execute(cls, *args, **kwargs) -> RaggedVerifyWindow:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        verify_num_draft_tokens: int,
        model_runner,
    ) -> RaggedVerifyWindow:
        return build_ragged_verify_window(
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=verify_num_draft_tokens,
            model_runner=model_runner,
        )

    @classmethod
    def triton(
        cls,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        verify_num_draft_tokens: int,
        model_runner,
    ) -> RaggedVerifyWindow:
        return build_ragged_verify_window_triton(
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=verify_num_draft_tokens,
            model_runner=model_runner,
        )


def build_ragged_verify_window(
    *,
    batch: ScheduleBatch,
    layout: RaggedVerifyLayout,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    model_runner,
) -> RaggedVerifyWindow:
    prefix_lens = batch.seq_lens
    verify_lens = layout.verify_lens.to(device=device, dtype=torch.int32)
    padded_total = layout.graph_num_tokens

    req_id, within, valid = compact_row_index(
        verify_lens=verify_lens, padded_total=padded_total, device=device
    )
    safe_req = req_id.clamp(max=bs - 1)
    positions = torch.where(
        valid,
        prefix_lens.to(torch.int64)[safe_req] + within,
        torch.zeros_like(within),
    )
    real_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + verify_lens.to(prefix_lens.dtype),
        batch_size=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    verify_cache_loc = torch.nn.functional.pad(
        real_cache_loc, (0, padded_total - real_cache_loc.shape[0])
    )
    verify_cache_loc = torch.where(
        valid, verify_cache_loc, torch.zeros_like(verify_cache_loc)
    )

    verify_ids = compact_verify_ids(
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        layout=layout,
        device=device,
    )

    return RaggedVerifyWindow(
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        verify_ids=verify_ids,
    )


@triton.jit
def _ragged_finalize_kernel(
    req_ptr,
    within_ptr,
    prefix_ptr,
    cache_ptr,
    pos_out_ptr,
    cache_out_ptr,
    bs,
    n,
    real_len,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    req = tl.load(req_ptr + offs, mask=mask, other=0)
    within = tl.load(within_ptr + offs, mask=mask, other=0)
    valid = req < bs
    safe_req = tl.minimum(req, bs - 1)
    prefix = tl.load(prefix_ptr + safe_req, mask=mask, other=0)
    pos = tl.where(valid, prefix + within, 0)
    lmask = mask & (offs < real_len)
    cl = tl.load(cache_ptr + offs, mask=lmask, other=0)
    cl = tl.where(valid, cl, 0)
    tl.store(pos_out_ptr + offs, pos, mask=mask)
    tl.store(cache_out_ptr + offs, cl, mask=mask)


def build_ragged_verify_window_triton(
    *,
    batch: ScheduleBatch,
    layout: RaggedVerifyLayout,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    model_runner,
) -> RaggedVerifyWindow:
    prefix_lens = batch.seq_lens
    verify_lens = layout.verify_lens.to(device=device, dtype=torch.int32)
    padded_total = layout.graph_num_tokens

    req_id, within, _valid = compact_row_index_triton(
        verify_lens=verify_lens, padded_total=padded_total, device=device
    )
    real_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + verify_lens.to(prefix_lens.dtype),
        batch_size=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    prefix_i64 = prefix_lens.to(device=device, dtype=torch.int64).contiguous()
    positions = torch.empty(padded_total, dtype=torch.int64, device=device)
    verify_cache_loc = torch.empty(
        padded_total, dtype=real_cache_loc.dtype, device=device
    )
    BLOCK = 256
    grid = (triton.cdiv(padded_total, BLOCK),)
    _ragged_finalize_kernel[grid](
        req_id,
        within,
        prefix_i64,
        real_cache_loc,
        positions,
        verify_cache_loc,
        bs,
        padded_total,
        real_cache_loc.shape[0],
        BLOCK=BLOCK,
    )

    verify_ids = compact_verify_ids_triton(
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        layout=layout,
        device=device,
    )
    return RaggedVerifyWindow(
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        verify_ids=verify_ids,
    )
