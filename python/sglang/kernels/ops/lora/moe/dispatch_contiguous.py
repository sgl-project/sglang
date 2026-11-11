"""Dispatch rows into aligned expert segments, optionally quantizing to FP8.

Negative expert IDs map to pair_to_row == -1, which every consumer skips.
Atomic slot order may vary, so every stage must use the same pair_to_row map.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.lora.moe.dispatch_checks import (
    check_fp8_width,
    check_source_rows,
)
from sglang.kernels.ops.lora.moe.fp8_quant import quantize_fp8_groups
from sglang.kernels.ops.lora.moe.quant_blockscaled import (
    BlockScaledRows,
    div_rn_ieee,
    quant_column_chunk,
    quantize_tile,
    store_quantized_rows,
)

PAIRS_PER_PROGRAM = 8
# The atomic count takes wider programs: fewer launches of the same atomics.
_COUNT_PAIRS_PER_PROGRAM = 32


def contiguous_m_pad_ceiling(num_pairs: int, num_experts: int, alignment: int) -> int:
    """Bound padded rows by allowing alignment - 1 extra rows per expert."""
    raw = num_pairs + num_experts * (alignment - 1)
    return -(-raw // alignment) * alignment


@triton.jit
def _count_slots_contiguous_kernel(
    topk_ids_ptr,  # [num_pairs]; a value < 0 marks a pair with no expert
    slot_out_ptr,  # [num_pairs] int32 out: slot of the pair inside its expert, -1 without one
    seg_counts_ptr,  # [E_local] int32 count and cursor; the caller must zero it
    num_pairs,
    PAIRS_PER_PROGRAM: tl.constexpr,
):
    pairs = tl.program_id(0).to(tl.int64) * PAIRS_PER_PROGRAM + tl.arange(
        0, PAIRS_PER_PROGRAM
    )
    in_range = pairs < num_pairs
    experts = tl.load(topk_ids_ptr + pairs, mask=in_range, other=-1)
    valid = in_range & (experts >= 0)
    slots = tl.atomic_add(seg_counts_ptr + experts, 1, mask=valid)
    tl.store(slot_out_ptr + pairs, tl.where(valid, slots, -1), mask=in_range)


@triton.jit
def _seg_layout_contiguous_kernel(
    seg_counts_ptr,  # [E_local] int32 per-expert routed-pair counts
    seg_offsets_ptr,  # [E_local + 1] int32 out: first row of each segment
    ALIGN: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_EXPERTS)
    counts = tl.load(seg_counts_ptr + offs, mask=offs < EXPERTS, other=0)
    aligned = ((counts + (ALIGN - 1)) // ALIGN) * ALIGN
    inclusive = tl.cumsum(aligned, axis=0)
    tl.store(seg_offsets_ptr + offs, inclusive - aligned, mask=offs < EXPERTS)
    tl.store(seg_offsets_ptr + EXPERTS, tl.sum(aligned))


@triton.jit
def _fill_rows_contiguous_bf16_kernel(
    input_ptr,  # [num_tokens, hidden] bf16 source rows
    compact_ptr,  # [m_pad_ceiling, hidden] bf16, viewed flat
    topk_ids_ptr,  # [num_pairs]
    pair_to_row_ptr,  # [num_pairs] int32 finalized compact rows
    hidden_size,
    num_pairs,
    TOPK: tl.constexpr,
    PAIRS_PER_PROGRAM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Row copying consumes the map finalized by the preceding launch.
    lane = tl.arange(0, PAIRS_PER_PROGRAM)
    base = tl.program_id(0).to(tl.int64) * PAIRS_PER_PROGRAM
    pairs = base + lane
    in_range = pairs < num_pairs
    experts_vec = tl.load(topk_ids_ptr + pairs, mask=in_range, other=-1)
    if tl.sum((experts_vec >= 0).to(tl.int32), axis=0) == 0:
        return

    vec = tl.arange(0, BLOCK_H)
    for i in tl.static_range(PAIRS_PER_PROGRAM):
        pair = base + i
        if pair < num_pairs:
            expert = tl.load(topk_ids_ptr + pair)
            if expert >= 0:
                dst = tl.load(pair_to_row_ptr + pair).to(tl.int64)
                token = pair // TOPK
                src = input_ptr + token * hidden_size
                out = compact_ptr + dst * hidden_size
                for off in tl.range(0, hidden_size, BLOCK_H):
                    mask = off + vec < hidden_size
                    tl.store(
                        out + off + vec,
                        tl.load(src + off + vec, mask=mask),
                        mask=mask,
                    )


@triton.jit
def _finalize_pair_to_row_contiguous_kernel(
    topk_ids_ptr,  # [num_pairs]; a value < 0 marks a pair with no expert
    pair_to_row_ptr,  # [num_pairs] int32: dense slots IN, compact rows OUT
    seg_offsets_ptr,  # [E_local + 1] int32 first row of each segment
    num_pairs,
    PAIRS_PER_PROGRAM: tl.constexpr,
):
    base = tl.program_id(0).to(tl.int64) * PAIRS_PER_PROGRAM
    for i in tl.static_range(PAIRS_PER_PROGRAM):
        pair = base + i
        if pair < num_pairs:
            expert = tl.load(topk_ids_ptr + pair)
            if expert >= 0:
                slot = tl.load(pair_to_row_ptr + pair)
                seg_base = tl.load(seg_offsets_ptr + expert)
                tl.store(pair_to_row_ptr + pair, seg_base + slot)
            else:
                # Finalize gates on this map, so invalid pairs must read as -1.
                tl.store(pair_to_row_ptr + pair, -1)


def dispatch_layout_contiguous(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_local_experts: int,
    top_k: int,
    alignment: int,
    *,
    seg_counts_out: torch.Tensor,
    seg_offsets_out: torch.Tensor,
    pair_to_row_out: torch.Tensor,
    finalize: bool = True,
) -> None:
    """Write compact row indices to pair_to_row; row copies run separately.

    Without ``finalize`` the map holds each pair's slot inside its expert (-1
    without one), for a fill that adds the segment base itself.
    """
    check_source_rows(hidden_states, topk_ids, top_k)
    num_pairs = topk_ids.numel()
    pair_grid = (-(-num_pairs // PAIRS_PER_PROGRAM),)

    # Clear atomic cursors on every forward, including graph replay.
    seg_counts_out.zero_()
    if num_pairs > 0:
        _count_slots_contiguous_kernel[(-(-num_pairs // _COUNT_PAIRS_PER_PROGRAM),)](
            topk_ids.view(-1),
            pair_to_row_out,
            seg_counts_out,
            num_pairs,
            PAIRS_PER_PROGRAM=_COUNT_PAIRS_PER_PROGRAM,
        )
    # Initialize offsets even for an empty batch.
    _seg_layout_contiguous_kernel[(1,)](
        seg_counts_out,
        seg_offsets_out,
        ALIGN=alignment,
        EXPERTS=num_local_experts,
        BLOCK_EXPERTS=max(triton.next_power_of_2(num_local_experts), 2),
    )
    if finalize and num_pairs > 0:
        _finalize_pair_to_row_contiguous_kernel[pair_grid](
            topk_ids.view(-1),
            pair_to_row_out,
            seg_offsets_out,
            num_pairs,
            PAIRS_PER_PROGRAM=PAIRS_PER_PROGRAM,
        )


def dispatch_fill_rows_contiguous_bf16(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    pair_to_row: torch.Tensor,
    *,
    hidden_compact_out: torch.Tensor,
) -> None:
    num_tokens = hidden_states.size(0)
    num_pairs = topk_ids.numel()
    if num_pairs == 0:
        return
    check_source_rows(hidden_states, topk_ids, num_pairs // num_tokens)
    _fill_rows_contiguous_bf16_kernel[(-(-num_pairs // PAIRS_PER_PROGRAM),)](
        hidden_states,
        hidden_compact_out,
        topk_ids.view(-1),
        pair_to_row,
        hidden_states.size(1),
        num_pairs,
        TOPK=num_pairs // num_tokens,
        PAIRS_PER_PROGRAM=PAIRS_PER_PROGRAM,
        BLOCK_H=1024,
    )


@triton.jit
def _fill_rows_contiguous_fp8_kernel(
    input_ptr,  # bf16 [num_tokens, K] source rows
    out_ptr,  # fp8e4m3 [m_pad_ceiling, K] contiguous
    scale_ptr,  # fp32 [m_pad_ceiling, GROUPS] contiguous
    topk_ids_ptr,  # [num_pairs]; < 0 = padding or EP-unrouted
    pair_to_row_ptr,  # [num_pairs] int32 compact rows (slots in the expert if FINALIZE)
    seg_offsets_ptr,  # [E_local + 1] int32 first row of each segment (FINALIZE only)
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,  # 128, or BLOCK_K for one scale per token
    GROUPS: tl.constexpr,  # K // GROUP, or 1 per token
    BLOCK_K: tl.constexpr,
    FINALIZE: tl.constexpr,
):
    # Quantize each token once, then fan out to its routed destinations.
    token = tl.program_id(0)
    if FINALIZE:
        # The token's pairs: slot inside the expert in, compact row out
        # (-1 without an expert); the row loop below reloads the map.
        lane = tl.arange(0, TOPK_P2)
        pairs = token.to(tl.int64) * TOPK + lane
        in_token = lane < TOPK
        experts = tl.load(topk_ids_ptr + pairs, mask=in_token, other=-1)
        routed = in_token & (experts >= 0)
        slots = tl.load(pair_to_row_ptr + pairs, mask=routed, other=0)
        bases = tl.load(seg_offsets_ptr + experts, mask=routed, other=0)
        tl.store(
            pair_to_row_ptr + pairs, tl.where(routed, bases + slots, -1), mask=in_token
        )
        tl.debug_barrier()
    offs = tl.arange(0, BLOCK_K)
    x = tl.load(input_ptr + token.to(tl.int64) * K + offs, mask=offs < K, other=0).to(
        tl.float32
    )
    grouped = tl.reshape(x, (BLOCK_K // GROUP, GROUP))
    q, scale = quantize_fp8_groups(grouped, GROUPS == 1)
    qf = tl.reshape(q, (BLOCK_K,)).to(out_ptr.dtype.element_ty)
    goffs = tl.arange(0, BLOCK_K // GROUP)
    for j in range(TOPK):
        pair = token * TOPK + j
        expert = tl.load(topk_ids_ptr + pair)
        if expert >= 0:
            dst = tl.load(pair_to_row_ptr + pair).to(tl.int64)
            tl.store(out_ptr + dst * K + offs, qf, mask=offs < K)
            tl.store(scale_ptr + dst * GROUPS + goffs, scale, mask=goffs < GROUPS)


def dispatch_fill_rows_contiguous_fp8(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    pair_to_row: torch.Tensor,
    *,
    rows_fp8_out: torch.Tensor,
    scale_out: torch.Tensor,
    per_token: bool = False,
    seg_offsets: torch.Tensor | None = None,
) -> None:
    """Quantize and copy rows using the pair_to_row map.

    Scales are FP32 per group of 128 values, or one per token row. With
    ``seg_offsets`` the map holds slots inside the expert (a layout prepared
    without finalize) and the fill finalizes it on the way.
    """
    num_tokens, k = hidden_states.shape
    num_pairs = topk_ids.numel()
    if num_pairs == 0:
        return
    check_source_rows(hidden_states, topk_ids, num_pairs // num_tokens)
    check_fp8_width(k)
    block_k = triton.next_power_of_2(k)
    top_k = num_pairs // num_tokens
    _fill_rows_contiguous_fp8_kernel[(num_tokens,)](
        hidden_states,
        rows_fp8_out,
        scale_out,
        topk_ids.view(-1),
        pair_to_row,
        pair_to_row if seg_offsets is None else seg_offsets,
        TOPK=top_k,
        TOPK_P2=triton.next_power_of_2(top_k),
        K=k,
        GROUP=block_k if per_token else 128,
        GROUPS=1 if per_token else k // 128,
        BLOCK_K=block_k,
        FINALIZE=seg_offsets is not None,
    )


@triton.jit
def _token_scale_kernel(
    input_ptr,  # bf16 [num_tokens, K]
    token_scale_ptr,  # fp32 [num_tokens] out
    row_denominator,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-token NVFP4: each token's stored scale, row_amax / row_denominator."""
    token = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    amax = tl.zeros((BLOCK_K,), tl.float32)
    for col0 in range(0, K, BLOCK_K):
        x = tl.load(input_ptr + token * K + col0 + offs)
        amax = tl.maximum(amax, tl.abs(x.to(tl.float32)))
    one = tl.zeros((1,), tl.float32)
    tl.store(
        token_scale_ptr + token + tl.arange(0, 1),
        div_rn_ieee(tl.max(amax, axis=0) + one, one + row_denominator),
    )


@triton.jit
def _fill_rows_contiguous_blockscaled_kernel(
    input_ptr,  # bf16 [num_tokens, K] source rows
    out_ptr,  # [m_pad_ceiling, K] E4M3 or [m_pad_ceiling, K // 2] E2M1 pairs
    sf_ptr,  # uint8 scale atoms
    topk_ids_ptr,  # [num_pairs]; < 0 = padding or EP-unrouted
    pair_to_row_ptr,  # [num_pairs] int32 finalized compact rows
    stride_out_row,
    rows_per_slab,
    row_atoms,
    k_atoms,
    global_scale,
    row_scale_ptr,  # [m_pad_ceiling] fp32 stored per-row scales (PER_ROW)
    row_denominator,
    token_scale_ptr,  # [num_tokens] fp32 scales from the pre-pass (PER_ROW)
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    VEC: tl.constexpr,
    ELEM_MAX: tl.constexpr,
    ELEM_MAX_LOG2: tl.constexpr,
    PACK_E2M1: tl.constexpr,
    NVFP4: tl.constexpr,
    PER_ROW: tl.constexpr,
    RECIPE_SELECT: tl.constexpr,
    ERR_MSE: tl.constexpr,
    ERR_FAST: tl.constexpr,
):
    # Quantize one column chunk of a token once and store it to every routed
    # row in one tile. Chunks divide K, so every lane is live; the token's
    # destinations load as one vector (-1 marks a pair without an expert).
    token = tl.program_id(0)
    col0 = tl.program_id(1) * BLOCK_K
    offs = col0 + tl.arange(0, BLOCK_K)
    x = tl.load(input_ptr + token.to(tl.int64) * K + offs)
    lanes = tl.arange(0, TOPK_P2)
    dsts = tl.load(
        pair_to_row_ptr + token.to(tl.int64) * TOPK + lanes, mask=lanes < TOPK, other=-1
    )
    if PER_ROW:
        row_scale = tl.load(token_scale_ptr + token + tl.arange(0, 1))
    else:
        row_scale = tl.zeros((1,), tl.float32)
    codes, sf_bytes = quantize_tile(
        x.to(tl.float32)[None, :],
        global_scale,
        row_scale,
        VEC,
        ELEM_MAX,
        ELEM_MAX_LOG2,
        PACK_E2M1,
        NVFP4,
        PER_ROW,
        RECIPE_SELECT,
        ERR_MSE,
        ERR_FAST,
        row_denominator,
    )
    store_quantized_rows(
        tl.broadcast_to(codes, (TOPK_P2, codes.shape[1])),
        tl.broadcast_to(sf_bytes, (TOPK_P2, sf_bytes.shape[1])),
        dsts.to(tl.int64),
        dsts >= 0,
        col0,
        K,
        out_ptr,
        stride_out_row,
        sf_ptr,
        rows_per_slab,
        row_atoms,
        k_atoms,
        VEC,
        PACK_E2M1,
    )
    if PER_ROW:
        # The first chunk's program records the token's scale at its rows.
        tl.store(
            row_scale_ptr + dsts.to(tl.int64),
            tl.broadcast_to(row_scale, (TOPK_P2,)),
            mask=(dsts >= 0) & (tl.program_id(1) == 0),
        )


def dispatch_fill_rows_contiguous_blockscaled(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    pair_to_row: torch.Tensor,
    *,
    rows_out: BlockScaledRows,
    token_scales: torch.Tensor | None = None,
) -> None:
    """Quantize into block-scaled rows using the finalized pair_to_row map.

    Per-token NVFP4 rows take ``token_scales``, an fp32 scratch of at least
    ``num_tokens`` entries that a pre-pass fills with each token's scale.
    """
    num_tokens, k = hidden_states.shape
    num_pairs = topk_ids.numel()
    if num_pairs == 0:
        return
    check_source_rows(hidden_states, topk_ids, num_pairs // num_tokens)
    assert k == rows_out.k and k % 128 == 0
    block_k = quant_column_chunk(k)
    top_k = num_pairs // num_tokens
    token_scale_ptr = hidden_states  # unread unless PER_ROW
    if rows_out.per_row:
        if (
            token_scales is None
            or token_scales.dtype != torch.float32
            or token_scales.numel() < num_tokens
        ):
            raise ValueError(
                "per-token block-scaled rows need an fp32 token-scale scratch "
                f"of at least {num_tokens} entries"
            )
        _token_scale_kernel[(num_tokens,)](
            hidden_states,
            token_scales,
            rows_out.row_denominator,
            K=k,
            BLOCK_K=block_k,
            num_warps=4,
        )
        token_scale_ptr = token_scales
    _fill_rows_contiguous_blockscaled_kernel[(num_tokens, k // block_k)](
        hidden_states,
        rows_out.data,
        rows_out.scales,
        topk_ids.view(-1),
        pair_to_row,
        token_scale_ptr=token_scale_ptr,
        TOPK=top_k,
        TOPK_P2=max(2, triton.next_power_of_2(top_k)),
        K=k,
        BLOCK_K=block_k,
        num_warps=4,
        **rows_out.kernel_args(),
    )
