from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_COMMIT_INJECT_LAYOUT.get()


class CommitInjectLayoutResult(msgspec.Struct):
    swa_loc: torch.Tensor
    positions: torch.Tensor


class BuildCommitInjectLayout:
    @classmethod
    def execute(cls, *args, **kwargs) -> CommitInjectLayoutResult:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_pos_offsets: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        commit_lens: torch.Tensor,
        stride: int,
    ) -> CommitInjectLayoutResult:
        return build_commit_inject_layout(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            prefix_lens=prefix_lens,
            block_pos_offsets=block_pos_offsets,
            full_to_swa_mapping=full_to_swa_mapping,
            commit_lens=commit_lens,
            stride=stride,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_pos_offsets: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        commit_lens: torch.Tensor,
        stride: int,
    ) -> CommitInjectLayoutResult:
        return build_commit_inject_layout_triton(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            prefix_lens=prefix_lens,
            block_pos_offsets=block_pos_offsets,
            full_to_swa_mapping=full_to_swa_mapping,
            commit_lens=commit_lens,
            stride=stride,
        )


def build_commit_inject_layout(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_pos_offsets: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
) -> CommitInjectLayoutResult:
    from sglang.kernels.ops.speculative.cache_locs import (
        assign_extend_cache_locs_func,
    )

    bs = req_pool_indices.shape[0]
    device = req_pool_indices.device

    positions_2d = prefix_lens.unsqueeze(1) + block_pos_offsets[:stride]
    positions = positions_2d.reshape(-1).to(dtype=torch.int64)

    cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + stride,
        batch_size=bs,
        draft_token_num=stride,
        device=device,
    ).to(dtype=torch.int64)
    swa_loc = full_to_swa_mapping[cache_loc].to(torch.int32)

    col = torch.arange(stride, device=device).view(1, -1)
    committed = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
    swa_loc = torch.where(committed, swa_loc, torch.full_like(swa_loc, -1))

    return CommitInjectLayoutResult(swa_loc=swa_loc, positions=positions)


@triton.jit
def _commit_inject_layout_kernel(
    req_pool_ptr,
    req_to_token_ptr,
    prefix_lens_ptr,
    block_pos_offsets_ptr,
    full_to_swa_ptr,
    commit_lens_ptr,
    swa_loc_ptr,
    positions_ptr,
    rt_stride,
    stride,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    r = offs // stride
    c = offs % stride

    prefix = tl.load(prefix_lens_ptr + r, mask=mask, other=0).to(tl.int64)
    pos_off = tl.load(block_pos_offsets_ptr + c, mask=mask, other=0).to(tl.int64)
    rp = tl.load(req_pool_ptr + r, mask=mask, other=0).to(tl.int64)
    full_loc = tl.load(
        req_to_token_ptr + rp * rt_stride + prefix + pos_off, mask=mask, other=0
    ).to(tl.int64)
    swa = tl.load(full_to_swa_ptr + full_loc, mask=mask, other=-1).to(tl.int32)

    commit_len = tl.load(commit_lens_ptr + r, mask=mask, other=0).to(tl.int64)
    swa = tl.where(c.to(tl.int64) < commit_len, swa, -1)

    tl.store(swa_loc_ptr + offs, swa, mask=mask)
    tl.store(positions_ptr + offs, prefix + pos_off, mask=mask)


def build_commit_inject_layout_triton(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_pos_offsets: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
) -> CommitInjectLayoutResult:
    bs = req_pool_indices.shape[0]
    n = bs * stride
    device = req_pool_indices.device

    swa_loc = torch.empty(n, dtype=torch.int32, device=device)
    positions = torch.empty(n, dtype=torch.int64, device=device)
    BLOCK = 256
    _commit_inject_layout_kernel[(triton.cdiv(n, BLOCK),)](
        req_pool_indices,
        req_to_token,
        prefix_lens,
        block_pos_offsets,
        full_to_swa_mapping,
        commit_lens,
        swa_loc,
        positions,
        req_to_token.stride(0),
        stride,
        n,
        BLOCK=BLOCK,
    )
    return CommitInjectLayoutResult(swa_loc=swa_loc, positions=positions)
