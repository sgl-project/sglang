from __future__ import annotations

from typing import Tuple

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda
from sglang.srt.utils import ceil_align


class DsparkWindowGather(msgspec.Struct, frozen=True):
    num_q: int
    bs: int
    context_lens: torch.Tensor
    req_pool_indices_per_request: torch.Tensor
    offsets: torch.Tensor
    invalid: torch.Tensor


class ComputeDsparkWindowGather:
    @classmethod
    def execute(cls, *args, **kwargs) -> DsparkWindowGather:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        block_size: int,
        swa_window: int,
    ) -> DsparkWindowGather:
        return compute_dspark_window_gather(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
            block_size=block_size,
            swa_window=swa_window,
        )

    @classmethod
    def triton(
        cls,
        *,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        block_size: int,
        swa_window: int,
    ) -> DsparkWindowGather:
        return compute_dspark_window_gather_triton(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
            block_size=block_size,
            swa_window=swa_window,
        )


class BuildDsparkSwaPageIndices:
    @classmethod
    def execute(cls, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_per_request: torch.Tensor,
        offsets: torch.Tensor,
        invalid: torch.Tensor,
        out_loc: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        swa_window: int,
        page_index_aligned_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_dspark_swa_page_indices(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_per_request=req_pool_indices_per_request,
            offsets=offsets,
            invalid=invalid,
            out_loc=out_loc,
            context_lens=context_lens,
            block_size=block_size,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_per_request: torch.Tensor,
        offsets: torch.Tensor,
        invalid: torch.Tensor,
        out_loc: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        swa_window: int,
        page_index_aligned_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_dspark_swa_page_indices_triton(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_per_request=req_pool_indices_per_request,
            offsets=offsets,
            out_loc=out_loc,
            context_lens=context_lens,
            block_size=block_size,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
        )


def compute_dspark_window_gather(
    *,
    seq_lens_casual: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    block_size: int,
    swa_window: int,
) -> DsparkWindowGather:
    seq_lens_casual = seq_lens_casual.to(torch.int32)
    num_q = seq_lens_casual.size(0)
    assert num_q % block_size == 0, (
        f"DSpark draft block forward must be uniform-gamma: num_q={num_q} not "
        f"divisible by block_size={block_size}."
    )
    bs = num_q // block_size
    device = seq_lens_casual.device

    first_token = torch.arange(bs, device=device, dtype=torch.int64) * block_size
    prefix_lens = (seq_lens_casual[first_token] - 1).to(torch.int32)
    context_lens = torch.clamp(prefix_lens, max=swa_window).to(torch.int32)
    req_pool_indices_per_request = req_pool_indices_repeated[first_token]

    offsets = (
        prefix_lens.to(torch.int64).unsqueeze(1)
        - swa_window
        + torch.arange(swa_window, device=device, dtype=torch.int64).unsqueeze(0)
    )
    invalid = offsets < 0
    offsets = offsets.clamp(min=0)

    return DsparkWindowGather(
        num_q=num_q,
        bs=bs,
        context_lens=context_lens,
        req_pool_indices_per_request=req_pool_indices_per_request,
        offsets=offsets,
        invalid=invalid,
    )


@triton.jit
def _window_gather_kernel(
    seq_lens_casual_ptr,
    req_pool_rep_ptr,
    context_lens_ptr,
    req_pool_out_ptr,
    offsets_ptr,
    invalid_ptr,
    block_size,
    swa_window,
    W_BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    ft = i * block_size
    prefix = tl.load(seq_lens_casual_ptr + ft).to(tl.int64) - 1
    tl.store(context_lens_ptr + i, tl.minimum(prefix, swa_window).to(tl.int32))
    tl.store(req_pool_out_ptr + i, tl.load(req_pool_rep_ptr + ft))
    col = tl.arange(0, W_BLOCK)
    cmask = col < swa_window
    off = prefix - swa_window + col
    tl.store(invalid_ptr + i * swa_window + col, off < 0, mask=cmask)
    tl.store(offsets_ptr + i * swa_window + col, tl.maximum(off, 0), mask=cmask)


def compute_dspark_window_gather_triton(
    *,
    seq_lens_casual: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    block_size: int,
    swa_window: int,
) -> DsparkWindowGather:
    seq_lens_casual = seq_lens_casual.to(torch.int32).contiguous()
    num_q = seq_lens_casual.size(0)
    assert num_q % block_size == 0, (
        f"DSpark draft block forward must be uniform-gamma: num_q={num_q} not "
        f"divisible by block_size={block_size}."
    )
    bs = num_q // block_size
    device = seq_lens_casual.device
    req_pool_indices_repeated = req_pool_indices_repeated.to(device=device).contiguous()
    context_lens = torch.empty(bs, dtype=torch.int32, device=device)
    req_pool_out = torch.empty(bs, dtype=req_pool_indices_repeated.dtype, device=device)
    offsets = torch.empty((bs, swa_window), dtype=torch.int64, device=device)
    invalid = torch.empty((bs, swa_window), dtype=torch.bool, device=device)
    W_BLOCK = triton.next_power_of_2(swa_window)
    _window_gather_kernel[(bs,)](
        seq_lens_casual,
        req_pool_indices_repeated,
        context_lens,
        req_pool_out,
        offsets,
        invalid,
        block_size,
        swa_window,
        W_BLOCK=W_BLOCK,
    )
    return DsparkWindowGather(
        num_q=num_q,
        bs=bs,
        context_lens=context_lens,
        req_pool_indices_per_request=req_pool_out,
        offsets=offsets,
        invalid=invalid,
    )


def build_dspark_swa_page_indices(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_per_request: torch.Tensor,
    offsets: torch.Tensor,
    invalid: torch.Tensor,
    out_loc: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    swa_window: int,
    page_index_aligned_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if offsets.ndim != 2 or offsets.shape[1] != swa_window:
        raise ValueError(
            "offsets must be [bs, swa_window]; "
            f"got shape={tuple(offsets.shape)} (swa_window={swa_window})."
        )
    bs = offsets.shape[0]
    device = offsets.device
    context_lens = context_lens.to(device=device, dtype=torch.int32)

    window_full_locs = req_to_token[
        req_pool_indices_per_request[:, None].to(torch.int64), offsets
    ]
    window_full_locs = window_full_locs.masked_fill(invalid, 0)
    window_swa_locs = full_to_swa_mapping[window_full_locs].to(torch.int32)
    window_swa_locs = window_swa_locs.masked_fill(invalid, -1)

    block_full_locs = out_loc[: bs * block_size].view(bs, block_size)
    block_swa_locs = full_to_swa_mapping[block_full_locs].to(torch.int32)

    target_width = ceil_align(swa_window + block_size, page_index_aligned_size)

    swa_page_indices = _compact_dspark_window_then_block(
        window_swa_locs=window_swa_locs,
        block_swa_locs=block_swa_locs,
        context_lens=context_lens,
        target_width=target_width,
        block_size=block_size,
        swa_window=swa_window,
    )

    swa_page_indices = (
        swa_page_indices.view(bs, 1, target_width)
        .expand(bs, block_size, target_width)
        .reshape(bs * block_size, target_width)
        .contiguous()
    )
    swa_topk_lengths = (
        (context_lens + block_size)
        .view(bs, 1)
        .expand(bs, block_size)
        .reshape(bs * block_size)
        .contiguous()
        .to(torch.int32)
    )
    return swa_page_indices, swa_topk_lengths


def _compact_dspark_window_then_block(
    *,
    window_swa_locs: torch.Tensor,
    block_swa_locs: torch.Tensor,
    context_lens: torch.Tensor,
    target_width: int,
    block_size: int,
    swa_window: int,
) -> torch.Tensor:
    bs = window_swa_locs.shape[0]
    device = window_swa_locs.device
    out = torch.full((bs, target_width), -1, dtype=torch.int32, device=device)

    j = torch.arange(swa_window, device=device, dtype=torch.int32).view(1, -1)
    shift = (swa_window - context_lens.view(-1, 1)).to(torch.int32)
    src_col = (shift + j).clamp_(min=0, max=swa_window - 1).to(torch.int64)
    gathered = torch.gather(window_swa_locs, dim=1, index=src_col)
    valid = j < context_lens.view(-1, 1)
    out[:, :swa_window] = torch.where(valid, gathered, -1)

    block_col = context_lens.view(-1, 1) + torch.arange(
        block_size, device=device, dtype=torch.int32
    ).view(1, -1)
    block_rows = torch.arange(bs, device=device).view(-1, 1).expand(-1, block_size)
    out[block_rows, block_col] = block_swa_locs
    return out


@triton.jit
def _swa_page_indices_kernel(
    req_to_token_ptr,
    full_to_swa_ptr,
    req_pool_ptr,
    offsets_ptr,
    out_loc_ptr,
    context_lens_ptr,
    out_ptr,
    topk_ptr,
    rt_stride,
    swa_window,
    block_size,
    target_width,
    TW_BLOCK: tl.constexpr,
):
    q = tl.program_id(0)
    i = q // block_size
    cl = tl.load(context_lens_ptr + i)
    rp = tl.load(req_pool_ptr + i).to(tl.int64)
    k = tl.arange(0, TW_BLOCK)
    kmask = k < target_width
    in_window = k < cl
    src_col = tl.minimum(tl.maximum((swa_window - cl) + k, 0), swa_window - 1)
    wmask = kmask & in_window
    off = tl.load(offsets_ptr + i * swa_window + src_col, mask=wmask, other=0).to(
        tl.int64
    )
    win_full = tl.load(req_to_token_ptr + rp * rt_stride + off, mask=wmask, other=0).to(
        tl.int64
    )
    win_swa = tl.load(full_to_swa_ptr + win_full, mask=wmask, other=-1).to(tl.int32)

    in_block = (k >= cl) & (k < cl + block_size)
    bmask = kmask & in_block
    bcol = tl.maximum(k - cl, 0)
    blk_full = tl.load(out_loc_ptr + i * block_size + bcol, mask=bmask, other=0).to(
        tl.int64
    )
    blk_swa = tl.load(full_to_swa_ptr + blk_full, mask=bmask, other=-1).to(tl.int32)

    val = tl.where(in_window, win_swa, tl.where(in_block, blk_swa, -1))
    tl.store(out_ptr + q * target_width + k, val.to(tl.int32), mask=kmask)
    tl.store(topk_ptr + q, (cl + block_size).to(tl.int32))


def build_dspark_swa_page_indices_triton(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_per_request: torch.Tensor,
    offsets: torch.Tensor,
    out_loc: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    swa_window: int,
    page_index_aligned_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if offsets.ndim != 2 or offsets.shape[1] != swa_window:
        raise ValueError(
            "offsets must be [bs, swa_window]; "
            f"got shape={tuple(offsets.shape)} (swa_window={swa_window})."
        )
    bs = offsets.shape[0]
    device = offsets.device
    req_pool = req_pool_indices_per_request.to(device=device).contiguous()
    offsets = offsets.to(torch.int64).contiguous()
    out_loc = out_loc[: bs * block_size].contiguous()
    context_lens = context_lens.to(device=device, dtype=torch.int32).contiguous()
    rt_stride = req_to_token.stride(0)
    target_width = ceil_align(swa_window + block_size, page_index_aligned_size)
    n_q = bs * block_size
    swa_page_indices = torch.empty(
        (n_q, target_width), dtype=torch.int32, device=device
    )
    swa_topk_lengths = torch.empty(n_q, dtype=torch.int32, device=device)
    TW_BLOCK = triton.next_power_of_2(target_width)
    _swa_page_indices_kernel[(n_q,)](
        req_to_token,
        full_to_swa_mapping,
        req_pool,
        offsets,
        out_loc,
        context_lens,
        swa_page_indices,
        swa_topk_lengths,
        rt_stride,
        swa_window,
        block_size,
        target_width,
        TW_BLOCK=TW_BLOCK,
    )
    return swa_page_indices, swa_topk_lengths


class BuildBlockSeqLensCausal:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        seq_lens: torch.Tensor,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return build_block_seq_lens_causal(
            seq_lens=seq_lens,
            block_size=block_size,
            device=device,
        )

    @classmethod
    def triton(
        cls,
        *,
        seq_lens: torch.Tensor,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return build_block_seq_lens_causal_triton(
            seq_lens=seq_lens,
            block_size=block_size,
            device=device,
        )


def build_block_seq_lens_causal(
    *,
    seq_lens: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    prefix = seq_lens.to(torch.int32)
    steps = torch.arange(1, block_size + 1, device=device, dtype=torch.int32)
    return (prefix[:, None] + steps[None, :]).reshape(-1)


@triton.jit
def _block_seq_lens_casual_kernel(
    seq_lens_ptr,
    out_ptr,
    block_size,
    n_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    row = offs // block_size
    col = offs % block_size
    prefix = tl.load(seq_lens_ptr + row, mask=mask, other=0)
    tl.store(out_ptr + offs, (prefix + col + 1).to(tl.int32), mask=mask)


def build_block_seq_lens_causal_triton(
    *,
    seq_lens: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    seq_lens = seq_lens.to(device=device, dtype=torch.int64).contiguous()
    n_rows = seq_lens.shape[0]
    n_out = n_rows * block_size
    out = torch.empty(n_out, dtype=torch.int32, device=device)
    BLOCK = 256
    grid = (triton.cdiv(n_out, BLOCK),)
    _block_seq_lens_casual_kernel[grid](seq_lens, out, block_size, n_out, BLOCK=BLOCK)
    return out
