from __future__ import annotations

from typing import Optional, Tuple

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dspark_components.kernels.dispatch import inputs_on_cuda
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


class ExpandPrefillCausallyResult(msgspec.Struct):
    seq_lens_casual: torch.Tensor
    req_pool_indices_repeated: torch.Tensor


class ExpandPrefillCausally:
    @classmethod
    def execute(cls, *args, **kwargs) -> ExpandPrefillCausallyResult:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_start_loc: Optional[torch.Tensor],
        seq_lens_cpu: Optional[list[int]],
        extend_seq_lens_cpu: Optional[list[int]],
        num_tokens: int,
        padded_num_tokens: Optional[int],
    ) -> ExpandPrefillCausallyResult:
        return expand_prefill_causally(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            extend_start_loc=extend_start_loc,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            num_tokens=num_tokens,
            padded_num_tokens=padded_num_tokens,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_start_loc: Optional[torch.Tensor],
        seq_lens_cpu: Optional[list[int]],
        extend_seq_lens_cpu: Optional[list[int]],
        num_tokens: int,
        padded_num_tokens: Optional[int],
    ) -> ExpandPrefillCausallyResult:
        return expand_prefill_causally_triton(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            num_tokens=num_tokens,
            padded_num_tokens=padded_num_tokens,
        )


def expand_prefill_causally(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: Optional[torch.Tensor],
    seq_lens_cpu: Optional[list[int]],
    extend_seq_lens_cpu: Optional[list[int]],
    num_tokens: int,
    padded_num_tokens: Optional[int],
) -> ExpandPrefillCausallyResult:
    device = req_pool_indices.device
    cuda_int32_kwargs = {"dtype": torch.int32, "device": device}

    if extend_start_loc is not None:
        repeats = extend_seq_lens.to(torch.int64)
        req_pool_indices_repeated = torch.repeat_interleave(
            req_pool_indices, repeats, output_size=num_tokens
        )
        start_positions = seq_lens.to(torch.int32) - extend_seq_lens.to(torch.int32) + 1
        start_positions_repeated = torch.repeat_interleave(
            start_positions, repeats, output_size=num_tokens
        )
        start_locs_repeated = torch.repeat_interleave(
            extend_start_loc.to(torch.int32), repeats, output_size=num_tokens
        )
        token_offsets = (
            torch.arange(num_tokens, **cuda_int32_kwargs) - start_locs_repeated
        )
        seq_lens_casual = start_positions_repeated + token_offsets

        if padded_num_tokens is not None and padded_num_tokens > num_tokens:
            pad_size = padded_num_tokens - num_tokens
            seq_lens_casual = torch.nn.functional.pad(
                seq_lens_casual, (0, pad_size), value=1
            )
            req_pool_indices_repeated = torch.cat(
                (
                    req_pool_indices_repeated,
                    req_pool_indices_repeated[-1:].expand(pad_size),
                )
            )
        return ExpandPrefillCausallyResult(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )

    assert seq_lens_cpu is not None and extend_seq_lens_cpu is not None
    seq_lens_casual = torch.empty(num_tokens, **cuda_int32_kwargs)
    idx_to_req_repeated = torch.empty(num_tokens, **cuda_int32_kwargs)
    offset = 0
    for i, (kv_len, qo_len) in enumerate(zip(seq_lens_cpu, extend_seq_lens_cpu)):
        out = seq_lens_casual[offset : offset + qo_len]
        offset += qo_len
        torch.arange(kv_len - qo_len + 1, kv_len + 1, out=out)
        idx_to_req_repeated[offset - qo_len : offset].fill_(i)

    assert offset == num_tokens
    req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]

    if padded_num_tokens is not None and padded_num_tokens > num_tokens:
        pad_size = padded_num_tokens - num_tokens
        seq_lens_casual = torch.nn.functional.pad(
            seq_lens_casual, (0, pad_size), value=1
        )
        req_pool_indices_repeated = torch.nn.functional.pad(
            req_pool_indices_repeated,
            (0, pad_size),
            value=req_pool_indices_repeated[-1].item(),
        )
    return ExpandPrefillCausallyResult(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
    )


@triton.jit
def _expand_prefill_causally_kernel(
    req_pool_ptr,
    seq_lens_ptr,
    extend_seq_lens_ptr,
    seq_lens_casual_ptr,
    req_pool_repeated_ptr,
    bs,
    num_tokens,
    total_tokens,
    BLOCK: tl.constexpr,
    BS_P2: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_tokens

    b = tl.arange(0, BS_P2)
    bmask = b < bs
    extend = tl.load(extend_seq_lens_ptr + b, mask=bmask, other=0).to(tl.int32)
    start_locs = tl.cumsum(extend, axis=0) - extend

    is_real = offs < num_tokens
    t = tl.where(is_real, offs, 0).to(tl.int32)
    started = (start_locs[None, :] <= t[:, None]) & bmask[None, :]
    r = tl.sum(started.to(tl.int32), axis=1) - 1
    r = tl.where(is_real, r, bs - 1).to(tl.int64)

    seq_len = tl.load(seq_lens_ptr + r, mask=mask, other=0).to(tl.int32)
    ext = tl.load(extend_seq_lens_ptr + r, mask=mask, other=0).to(tl.int32)
    start_loc = tl.sum(tl.where(started, extend[None, :], 0).to(tl.int32), axis=1) - ext
    causal = (seq_len - ext + 1) + (t - start_loc)
    causal = tl.where(is_real, causal, 1)

    rp = tl.load(req_pool_ptr + r, mask=mask, other=0)
    tl.store(seq_lens_casual_ptr + offs, causal, mask=mask)
    tl.store(req_pool_repeated_ptr + offs, rp, mask=mask)


def expand_prefill_causally_triton(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    num_tokens: int,
    padded_num_tokens: Optional[int],
) -> ExpandPrefillCausallyResult:
    bs = req_pool_indices.shape[0]
    device = req_pool_indices.device
    total_tokens = (
        padded_num_tokens
        if padded_num_tokens is not None and padded_num_tokens > num_tokens
        else num_tokens
    )

    seq_lens_casual = torch.empty(total_tokens, dtype=torch.int32, device=device)
    req_pool_indices_repeated = torch.empty(
        total_tokens, dtype=req_pool_indices.dtype, device=device
    )
    BLOCK = 256
    _expand_prefill_causally_kernel[(triton.cdiv(total_tokens, BLOCK),)](
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        seq_lens_casual,
        req_pool_indices_repeated,
        bs,
        num_tokens,
        total_tokens,
        BLOCK=BLOCK,
        BS_P2=triton.next_power_of_2(max(bs, 1)),
    )
    return ExpandPrefillCausallyResult(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
    )


class PageTablePositionsResult(msgspec.Struct):
    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor
    page_table: torch.Tensor
    swa_topk_lengths: torch.Tensor


class BuildPageTablePositions:
    @classmethod
    def execute(cls, *args, **kwargs) -> PageTablePositionsResult:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        page_size: int,
        swa_window: int,
    ) -> PageTablePositionsResult:
        return build_page_table_positions(
            req_to_token=req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            page_size=page_size,
            swa_window=swa_window,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        page_size: int,
        swa_window: int,
    ) -> PageTablePositionsResult:
        return build_page_table_positions_triton(
            req_to_token=req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            page_size=page_size,
            swa_window=swa_window,
        )


def build_page_table_positions(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    swa_window: int,
) -> PageTablePositionsResult:
    seq_lens_casual = seq_lens_casual.to(torch.int32)
    positions_casual = seq_lens_casual - 1
    page_table = req_to_token[
        req_pool_indices_repeated.to(torch.int64), :max_seq_len:page_size
    ]
    page_table = (page_table // page_size).to(torch.int32)
    swa_topk_lengths = torch.clamp(seq_lens_casual, max=swa_window)
    return PageTablePositionsResult(
        seq_lens_casual=seq_lens_casual,
        positions_casual=positions_casual,
        page_table=page_table,
        swa_topk_lengths=swa_topk_lengths,
    )


@triton.jit
def _page_table_positions_kernel(
    req_to_token_ptr,
    req_pool_ptr,
    seq_lens_ptr,
    seq_lens_out_ptr,
    positions_out_ptr,
    page_table_ptr,
    topk_out_ptr,
    rt_stride,
    num_pages,
    page_size,
    swa_window,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    tl.store(seq_lens_out_ptr + row, seq_len)
    tl.store(positions_out_ptr + row, seq_len - 1)
    tl.store(topk_out_ptr + row, tl.minimum(seq_len, swa_window))

    rp = tl.load(req_pool_ptr + row).to(tl.int64)
    base = req_to_token_ptr + rp * rt_stride
    out_base = page_table_ptr + row.to(tl.int64) * num_pages
    for p0 in range(0, num_pages, BLOCK_P):
        p = p0 + tl.arange(0, BLOCK_P)
        pmask = p < num_pages
        tok = tl.load(base + p.to(tl.int64) * page_size, mask=pmask, other=0).to(
            tl.int32
        )
        tl.store(out_base + p, tok // page_size, mask=pmask)


def build_page_table_positions_triton(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    swa_window: int,
) -> PageTablePositionsResult:
    num_q = seq_lens_casual.shape[0]
    num_pages = (max_seq_len + page_size - 1) // page_size
    device = seq_lens_casual.device

    seq_lens_out = torch.empty(num_q, dtype=torch.int32, device=device)
    positions_out = torch.empty(num_q, dtype=torch.int32, device=device)
    page_table = torch.empty((num_q, num_pages), dtype=torch.int32, device=device)
    topk_out = torch.empty(num_q, dtype=torch.int32, device=device)
    BLOCK_P = 256
    _page_table_positions_kernel[(num_q,)](
        req_to_token,
        req_pool_indices_repeated,
        seq_lens_casual,
        seq_lens_out,
        positions_out,
        page_table,
        topk_out,
        req_to_token.stride(0),
        num_pages,
        page_size,
        swa_window,
        BLOCK_P=BLOCK_P,
    )
    return PageTablePositionsResult(
        seq_lens_casual=seq_lens_out,
        positions_casual=positions_out,
        page_table=page_table,
        swa_topk_lengths=topk_out,
    )


class BuildCausalSwaPageIndices:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        swa_window: int,
        page_index_aligned_size: int,
    ) -> torch.Tensor:
        return build_causal_swa_page_indices(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        swa_window: int,
        page_index_aligned_size: int,
    ) -> torch.Tensor:
        return build_causal_swa_page_indices_triton(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
        )


def build_causal_swa_page_indices(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    swa_window: int,
    page_index_aligned_size: int,
) -> torch.Tensor:
    device = seq_lens_casual.device
    pos_causal = seq_lens_casual - 1
    num_qo_tokens = seq_lens_casual.size(0)
    offsets = pos_causal.unsqueeze(1) - torch.arange(
        swa_window, dtype=torch.int32, device=device
    ).unsqueeze(0)
    invalid_offset_mask = offsets < 0
    offsets.masked_fill_(invalid_offset_mask, 0)
    raw_indices = req_to_token[req_pool_indices_repeated[:, None], offsets]
    assert raw_indices.shape == (num_qo_tokens, swa_window)
    raw_indices.masked_fill_(invalid_offset_mask, -1)
    swa_indices = full_to_swa_mapping[raw_indices]
    swa_indices = swa_indices.to(torch.int32)

    padded_width = (
        (swa_window + page_index_aligned_size - 1) // page_index_aligned_size
    ) * page_index_aligned_size
    if padded_width == swa_window:
        return swa_indices
    return torch.nn.functional.pad(
        swa_indices, (0, padded_width - swa_window), value=-1
    )


@triton.jit
def _causal_swa_page_indices_kernel(
    req_to_token_ptr,
    full_to_swa_ptr,
    req_pool_ptr,
    seq_lens_ptr,
    out_ptr,
    rt_stride,
    swa_window,
    padded_width,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.load(seq_lens_ptr + row).to(tl.int64) - 1
    rp = tl.load(req_pool_ptr + row).to(tl.int64)
    base = req_to_token_ptr + rp * rt_stride
    out_base = out_ptr + row.to(tl.int64) * padded_width

    for k0 in range(0, padded_width, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        kmask = k < padded_width
        off = pos - k.to(tl.int64)
        valid = (k < swa_window) & (off >= 0) & kmask
        full_loc = tl.load(base + tl.where(valid, off, 0), mask=valid, other=-1).to(
            tl.int64
        )
        swa = tl.load(full_to_swa_ptr + full_loc, mask=valid, other=-1).to(tl.int32)
        tl.store(out_base + k, tl.where(valid, swa, -1), mask=kmask)


def build_causal_swa_page_indices_triton(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    swa_window: int,
    page_index_aligned_size: int,
) -> torch.Tensor:
    num_qo_tokens = seq_lens_casual.size(0)
    padded_width = (
        (swa_window + page_index_aligned_size - 1) // page_index_aligned_size
    ) * page_index_aligned_size
    out = torch.empty(
        (num_qo_tokens, padded_width),
        dtype=torch.int32,
        device=seq_lens_casual.device,
    )
    BLOCK_K = 256
    _causal_swa_page_indices_kernel[(num_qo_tokens,)](
        req_to_token,
        full_to_swa_mapping,
        req_pool_indices_repeated,
        seq_lens_casual,
        out,
        req_to_token.stride(0),
        swa_window,
        padded_width,
        BLOCK_K=BLOCK_K,
    )
    return out


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
