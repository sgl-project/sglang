from __future__ import annotations

from typing import List, Optional, Tuple

import msgspec
import torch
import triton
import triton.language as tl


def _inputs_on_cuda(*args, **kwargs) -> bool:
    """Route kernel dispatch by input placement: the first tensor argument
    decides. CUDA inputs take the fused triton kernel; CPU inputs take the
    torch reference implementation (triton is CUDA-only, and CPU-side callers
    such as unit tests exercise the reference path)."""
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value.is_cuda
    raise AssertionError("kernel dispatch requires at least one tensor argument")


class ExpandPrefillCausallyResult(msgspec.Struct):
    seq_lens_casual: torch.Tensor
    req_pool_indices_repeated: torch.Tensor


class ExpandPrefillCausally:
    @classmethod
    def execute(cls, *args, **kwargs) -> ExpandPrefillCausallyResult:
        if _inputs_on_cuda(*args, **kwargs):
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
        if _inputs_on_cuda(*args, **kwargs):
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
        if _inputs_on_cuda(*args, **kwargs):
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


class ImageSpan(msgspec.Struct, frozen=True):
    """One image block's bidirectionally-visible extent, in flat-batch rows.

    ``left_origin`` is the row that ``visible_left`` counts from. It is below
    ``row_start`` when the span opened in an earlier prefill chunk, and equal to
    it when the span opens in this one.
    """

    row_start: int
    row_end: int
    left_origin: int


def build_image_visible_spans(
    *,
    spans: List[ImageSpan],
    num_tokens: int,
    max_image_tokens: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token visible counts inside each image span, for both directions.

    DeepSeek-V4-Flash-Vision attends bidirectionally within one image's span --
    the ``[IMAGE_START, IMAGE_END]`` range of its token block -- while the rest
    of the sequence stays causal. This returns, per query token, how many tokens
    of its own span sit to its left and to its right; both are 0 for a token
    outside every span, which leaves it on the plain sliding window.

    Spans arrive already resolved and clipped by the caller. Deliberately not
    inferred from runs of placeholder ids in ``input_ids``: two images whose
    blocks land next to each other form one run, and so do two requests whose
    positions happen to be contiguous across a batch boundary -- either merge
    would let one image attend into the next and push the gather length past the
    indices row it addresses.

    Both tensors are ``int32[num_tokens]``.
    """
    visible_left = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    visible_right = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    for span in spans:
        width = span.row_end - span.row_start + 1
        rows = slice(span.row_start, span.row_end + 1)
        first_left = span.row_start - span.left_origin
        visible_left[rows] = torch.arange(
            first_left, first_left + width, dtype=torch.int32, device=device
        )
        visible_right[rows] = torch.arange(
            width - 1, -1, -1, dtype=torch.int32, device=device
        )
    return (
        visible_left.clamp_(max=max_image_tokens - 1),
        visible_right.clamp_(max=max_image_tokens),
    )


class BuildVisionSwaPageIndices:
    """The sliding-window gather, widened where an image span is visible.

    Same contract as :class:`BuildCausalSwaPageIndices` -- a per-query list of
    SWA-pool slots plus the number of leading entries that are valid -- except
    that a query inside an image span reaches back to the span's start and
    forward to its end. The gathered range stays contiguous in position, so the
    valid entries stay packed at the front.
    """

    @classmethod
    def execute(cls, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if _inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(cls, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_vision_swa_page_indices(**kwargs)

    @classmethod
    def triton(cls, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_vision_swa_page_indices_triton(**kwargs)


def _vision_window_extent(
    seq_lens_casual: torch.Tensor,
    visible_left: torch.Tensor,
    visible_right: torch.Tensor,
    swa_window: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """First visible position per query, and how many positions are visible.

    ``visible_right`` must already be clipped so that ``pos + right`` stays
    inside the query's own request -- the caller does that when it resolves the
    spans, and only the caller knows where the request ends. Reaching past it
    would gather ``req_to_token`` columns the request never wrote.

    An exact span also keeps ``left + right`` inside one block, which bounds the
    result by ``swa_window + max_image_tokens`` -- the gather's own width. The
    clamp is the backstop for that one: a layout that broke the bound would
    otherwise hand the attention kernel a topk_length past the end of its
    indices row, and losing the far end of an image's forward reach beats
    reading out of bounds.
    """
    pos = seq_lens_casual.to(torch.int32) - 1
    back = torch.clamp(visible_left, min=swa_window - 1)
    back = torch.minimum(back, pos)
    lengths = (back + 1 + visible_right).clamp_(max=width)
    return pos - back, lengths


def build_vision_swa_page_indices(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    visible_left: torch.Tensor,
    visible_right: torch.Tensor,
    swa_window: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = seq_lens_casual.device
    first_pos, lengths = _vision_window_extent(
        seq_lens_casual, visible_left, visible_right, swa_window, width
    )
    offsets = first_pos.unsqueeze(1) + torch.arange(
        width, dtype=torch.int32, device=device
    ).unsqueeze(0)
    valid = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    raw_indices = req_to_token[
        req_pool_indices_repeated[:, None].to(torch.long),
        torch.where(valid, offsets, offsets.new_zeros(())).to(torch.long),
    ]
    swa_indices = full_to_swa_mapping[raw_indices.to(torch.long)].to(torch.int32)
    # -1 past topk_length, matching the triton path: the sentinel the attention
    # kernel's index contract documents, rather than whatever the mapping holds
    # for the slot the masked gather happened to read.
    swa_indices = torch.where(valid, swa_indices, swa_indices.new_full((), -1))
    return swa_indices, lengths


@triton.jit
def _vision_swa_page_indices_kernel(
    req_to_token_ptr,
    full_to_swa_ptr,
    req_pool_ptr,
    seq_lens_ptr,
    visible_left_ptr,
    visible_right_ptr,
    out_ptr,
    lengths_ptr,
    rt_stride,
    swa_window,
    width,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.load(seq_lens_ptr + row).to(tl.int64) - 1
    left = tl.load(visible_left_ptr + row).to(tl.int64)
    right = tl.load(visible_right_ptr + row).to(tl.int64)
    back = tl.minimum(tl.maximum(left, swa_window - 1), pos)
    first_pos = pos - back
    # Same backstop as the torch path: never address past the indices row.
    length = tl.minimum(back + 1 + right, width)
    tl.store(lengths_ptr + row, length.to(tl.int32))

    rp = tl.load(req_pool_ptr + row).to(tl.int64)
    base = req_to_token_ptr + rp * rt_stride
    out_base = out_ptr + row.to(tl.int64) * width

    for k0 in range(0, width, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        kmask = k < width
        off = first_pos + k.to(tl.int64)
        valid = (k.to(tl.int64) < length) & kmask
        full_loc = tl.load(base + tl.where(valid, off, 0), mask=valid, other=-1).to(
            tl.int64
        )
        swa = tl.load(full_to_swa_ptr + full_loc, mask=valid, other=-1).to(tl.int32)
        tl.store(out_base + k, tl.where(valid, swa, -1), mask=kmask)


def build_vision_swa_page_indices_triton(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    visible_left: torch.Tensor,
    visible_right: torch.Tensor,
    swa_window: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_qo_tokens = seq_lens_casual.size(0)
    device = seq_lens_casual.device
    out = torch.empty((num_qo_tokens, width), dtype=torch.int32, device=device)
    lengths = torch.empty(num_qo_tokens, dtype=torch.int32, device=device)
    _vision_swa_page_indices_kernel[(num_qo_tokens,)](
        req_to_token,
        full_to_swa_mapping,
        req_pool_indices_repeated,
        seq_lens_casual,
        visible_left,
        visible_right,
        out,
        lengths,
        req_to_token.stride(0),
        swa_window,
        width,
        BLOCK_K=256,
    )
    return out, lengths


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
