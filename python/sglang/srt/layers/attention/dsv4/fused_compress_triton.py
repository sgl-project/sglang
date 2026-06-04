"""HIP fused compressor kernels using the NV/main metadata contract.

The public wrappers mirror ``compress_forward``:

 decode: indices, seq_lens, extra_data
 prefill: indices, compress_plan, write_plan, extra_data

Prefill plans are the upstream 16-byte ``PrefillPlan`` structs stored as
``uint8[:, 16]``. The wrappers reinterpret them as ``int32[:, 4]`` before
launching Triton kernels.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.dsv4.compress_old import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
)


@triton.jit
def _fused_ape_pool_norm_rope_kernel(
    kv_score_ptr,
    kv_score_stride_b,
    kv_score_stride_k,
    ape_ptr,
    ape_stride_r,
    rms_weight_ptr,
    rms_eps,
    freqs_ptr,
    freqs_stride_b,
    out_ptr,
    out_stride_b,
    head_dim,
    rope_head_dim,
    half_dim,
    RATIO: tl.constexpr,
    K_POOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    bid = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim

    m_prev = tl.full([BLOCK_D], float("-inf"), tl.float32)
    kv_acc = tl.zeros([BLOCK_D], tl.float32)
    w_acc = tl.zeros([BLOCK_D], tl.float32)

    batch_base = bid * kv_score_stride_b

    for k in tl.range(0, K_POOL):
        if OVERLAP:
            is_b = k >= RATIO
            col_off = tl.where(is_b, head_dim, 0)
        else:
            col_off = 0

        row_off = batch_base + k * kv_score_stride_k
        kv_val = tl.load(
            kv_score_ptr + row_off + col_off + d, mask=d_mask, other=0.0
        ).to(tl.float32)
        sc_val = tl.load(
            kv_score_ptr + row_off + half_dim + col_off + d, mask=d_mask, other=0.0
        ).to(tl.float32)

        ape_val = tl.load(
            ape_ptr + (k % RATIO) * ape_stride_r + col_off + d, mask=d_mask, other=0.0
        ).to(tl.float32)
        score_k = sc_val + ape_val

        m_new = tl.maximum(m_prev, score_k)
        exp_old = tl.where(m_prev == float("-inf"), 0.0, tl.exp(m_prev - m_new))
        exp_cur = tl.where(score_k == float("-inf"), 0.0, tl.exp(score_k - m_new))
        kv_acc = kv_acc * exp_old + exp_cur * kv_val
        w_acc = w_acc * exp_old + exp_cur
        m_prev = m_new

    compressed = kv_acc / w_acc
    rms_w = tl.load(rms_weight_ptr + d, mask=d_mask, other=0.0)
    c_sq = tl.where(d_mask, compressed * compressed, 0.0)
    var = tl.sum(c_sq, axis=0) / head_dim
    normed = compressed * tl.rsqrt(var + rms_eps) * rms_w

    out_base = out_ptr + bid * out_stride_b
    tl.store(out_base + d, normed.to(out_ptr.dtype.element_ty), mask=d_mask)

    rope_start = head_dim - rope_head_dim
    p = tl.arange(0, HALF_ROPE)
    pmask = p < (rope_head_dim // 2)
    xr = tl.load(out_base + rope_start + 2 * p, mask=pmask, other=0.0).to(tl.float32)
    xi = tl.load(out_base + rope_start + 2 * p + 1, mask=pmask, other=0.0).to(
        tl.float32
    )

    freq_base = bid * freqs_stride_b
    fr = tl.load(freqs_ptr + freq_base + 2 * p, mask=pmask, other=1.0).to(tl.float32)
    fi = tl.load(freqs_ptr + freq_base + 2 * p + 1, mask=pmask, other=0.0).to(
        tl.float32
    )

    tl.store(
        out_base + rope_start + 2 * p,
        (xr * fr - xi * fi).to(out_ptr.dtype.element_ty),
        mask=pmask,
    )
    tl.store(
        out_base + rope_start + 2 * p + 1,
        (xr * fi + xi * fr).to(out_ptr.dtype.element_ty),
        mask=pmask,
    )


def fused_ape_pool_norm_rope(
    kv_score_gathered: torch.Tensor,
    ape: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    freqs_cis_real: torch.Tensor,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    overlap: bool,
) -> torch.Tensor:
    """Fused APE-add + overlap-transform + softmax-pool + RMSNorm + RoPE."""
    coff = 2 if overlap else 1
    bs = kv_score_gathered.shape[0]
    k_in = kv_score_gathered.shape[1]
    last_dim = kv_score_gathered.shape[2]
    half_dim = last_dim // 2
    assert k_in == ratio * coff, f"k_in={k_in} != ratio*coff={ratio}*{coff}"

    out = torch.empty(
        bs, head_dim, dtype=torch.float32, device=kv_score_gathered.device
    )
    if bs == 0:
        return out

    block_d = triton.next_power_of_2(head_dim)
    half_rope = triton.next_power_of_2(rope_head_dim // 2)
    num_warps = 4 if head_dim <= 256 else 8

    _fused_ape_pool_norm_rope_kernel[(bs,)](
        kv_score_gathered,
        kv_score_gathered.stride(0),
        kv_score_gathered.stride(1),
        ape,
        ape.stride(0),
        rms_weight,
        rms_eps,
        freqs_cis_real,
        freqs_cis_real.stride(0),
        out,
        out.stride(0),
        head_dim,
        rope_head_dim,
        half_dim,
        RATIO=ratio,
        K_POOL=k_in,
        BLOCK_D=block_d,
        HALF_ROPE=half_rope,
        OVERLAP=int(overlap),
        num_warps=num_warps,
    )
    return out


@triton.jit
def _c4_decode_kernel(
    kv_in_ptr,
    out_ptr,
    buffer_ptr,
    ape_ptr,
    indices_ptr,
    seq_lens_ptr,
    extra_ptr,
    kv_in_row_stride,
    out_row_stride,
    buffer_page_stride,
    buffer_slot_stride,
    ape_row_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bid = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    index = tl.load(indices_ptr + bid).to(tl.int64)
    index_prev = tl.load(extra_ptr + bid).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + bid).to(tl.int32)
    write_slot = (seq_len + 3) % 4

    in_base = bid.to(tl.int64) * kv_in_row_stride
    page_base = (
        index * buffer_page_stride + write_slot.to(tl.int64) * buffer_slot_stride
    )

    valid_index = index >= 0
    for ch in tl.static_range(4):
        ch_off = ch * HEAD_DIM
        val = tl.load(kv_in_ptr + in_base + ch_off + d_offs, mask=d_mask, other=0.0)
        tl.store(
            buffer_ptr + page_base + ch_off + d_offs,
            val,
            mask=d_mask & valid_index,
        )

    NEG_BIG: tl.constexpr = -1.0e9
    running_max = tl.full((BLOCK_D,), NEG_BIG, tl.float32)
    running_sum = tl.zeros((BLOCK_D,), tl.float32)
    weighted = tl.zeros((BLOCK_D,), tl.float32)

    for slot in tl.static_range(8):
        if slot < 4:
            page = index_prev
            kv_off = 0
            score_off = 2 * HEAD_DIM
        else:
            page = index
            kv_off = HEAD_DIM
            score_off = 3 * HEAD_DIM

        src_pos = seq_len - 8 + slot
        is_input = slot == 7
        write_pos = ((seq_len - 1) // 4) * 4
        page = tl.where(src_pos < write_pos, index_prev, index)
        slot_in_page = src_pos % 4
        slot_base = (
            page * buffer_page_stride + slot_in_page.to(tl.int64) * buffer_slot_stride
        )
        valid = src_pos >= 0
        if slot == 7:
            kv = tl.load(
                kv_in_ptr + in_base + kv_off + d_offs,
                mask=d_mask & valid,
                other=0.0,
            )
            score = tl.load(
                kv_in_ptr + in_base + score_off + d_offs,
                mask=d_mask & valid,
                other=NEG_BIG,
            )
        else:
            kv = tl.load(
                buffer_ptr + slot_base + kv_off + d_offs,
                mask=d_mask & valid,
                other=0.0,
            )
            score = tl.load(
                buffer_ptr + slot_base + score_off + d_offs,
                mask=d_mask & valid,
                other=NEG_BIG,
            )
        bias = tl.load(ape_ptr + slot * ape_row_stride + d_offs, mask=d_mask, other=0.0)
        s = score + bias
        new_max = tl.maximum(running_max, s)
        factor = tl.exp(running_max - new_max)
        e = tl.where(valid, tl.exp(s - new_max), 0.0)
        running_sum = running_sum * factor + e
        weighted = weighted * factor + kv * e
        running_max = new_max

    tl.store(
        out_ptr + bid.to(tl.int64) * out_row_stride + d_offs,
        weighted / running_sum,
        mask=d_mask,
    )


@triton.jit
def _c4_prefill_compress_kernel(
    kv_in_ptr,
    out_ptr,
    buffer_ptr,
    ape_ptr,
    indices_ptr,
    extra_ptr,
    plan_ptr,
    kv_in_row_stride,
    out_row_stride,
    buffer_page_stride,
    buffer_slot_stride,
    ape_row_stride,
    plan_row_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    plan_base = plan_ptr + pid_p * plan_row_stride
    ragged_id = tl.load(plan_base + 0).to(tl.int32)
    batch_id = tl.load(plan_base + 1).to(tl.int32)
    position = tl.load(plan_base + 2).to(tl.int32)
    window_len = tl.load(plan_base + 3).to(tl.int32)
    if ragged_id < 0:
        return

    extra_base = extra_ptr + batch_id.to(tl.int64) * 4
    load_first_page = tl.load(extra_base + 0).to(tl.int64)
    load_second_page = tl.load(extra_base + 1).to(tl.int64)

    NEG_BIG: tl.constexpr = -1.0e9
    running_max = tl.full((BLOCK_D,), NEG_BIG, tl.float32)
    running_sum = tl.zeros((BLOCK_D,), tl.float32)
    weighted = tl.zeros((BLOCK_D,), tl.float32)

    for slot in tl.static_range(8):
        in_state = slot < window_len
        if slot < 4:
            page = tl.where(window_len <= 4, load_second_page, load_first_page)
            kv_off = 0
            score_off = 2 * HEAD_DIM
            slot_in_page = slot
        else:
            page = load_second_page
            kv_off = HEAD_DIM
            score_off = 3 * HEAD_DIM
            slot_in_page = slot - 4

        src_pos = position - 7 + slot
        state_valid = in_state & (src_pos >= 0)
        slot_base = page * buffer_page_stride + slot_in_page * buffer_slot_stride
        in_row = ragged_id - (7 - slot)
        in_row_safe = tl.where(in_state, 0, in_row)
        in_base = in_row_safe.to(tl.int64) * kv_in_row_stride

        kv_state = tl.load(
            buffer_ptr + slot_base + kv_off + d_offs,
            mask=d_mask & state_valid,
            other=0.0,
        )
        score_state = tl.load(
            buffer_ptr + slot_base + score_off + d_offs,
            mask=d_mask & state_valid,
            other=NEG_BIG,
        )
        kv_input = tl.load(
            kv_in_ptr + in_base + kv_off + d_offs,
            mask=d_mask & (~in_state),
            other=0.0,
        )
        score_input = tl.load(
            kv_in_ptr + in_base + score_off + d_offs,
            mask=d_mask & (~in_state),
            other=NEG_BIG,
        )
        kv = tl.where(in_state, kv_state, kv_input)
        score = tl.where(in_state, score_state, score_input)
        bias = tl.load(ape_ptr + slot * ape_row_stride + d_offs, mask=d_mask, other=0.0)

        s = score + bias
        new_max = tl.maximum(running_max, s)
        factor = tl.exp(running_max - new_max)
        e = tl.exp(s - new_max)
        running_sum = running_sum * factor + e
        weighted = weighted * factor + kv * e
        running_max = new_max

    tl.store(
        out_ptr + ragged_id.to(tl.int64) * out_row_stride + d_offs,
        weighted / running_sum,
        mask=d_mask,
    )


@triton.jit
def _c4_prefill_write_kernel(
    kv_in_ptr,
    buffer_ptr,
    indices_ptr,
    extra_ptr,
    plan_ptr,
    kv_in_row_stride,
    buffer_page_stride,
    buffer_slot_stride,
    plan_row_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    plan_base = plan_ptr + pid_p * plan_row_stride
    ragged_id = tl.load(plan_base + 0).to(tl.int32)
    batch_id = tl.load(plan_base + 1).to(tl.int32)
    position = tl.load(plan_base + 2).to(tl.int32)
    if ragged_id < 0:
        return

    extra_base = extra_ptr + batch_id.to(tl.int64) * 4
    write_first_page = tl.load(extra_base + 2).to(tl.int64)
    last_position = tl.load(extra_base + 3).to(tl.int32)
    write_second_page = tl.load(indices_ptr + batch_id).to(tl.int64)
    page = tl.where(position < last_position, write_first_page, write_second_page)
    slot = position % 4

    in_base = ragged_id.to(tl.int64) * kv_in_row_stride
    dst_base = page * buffer_page_stride + slot.to(tl.int64) * buffer_slot_stride
    for ch in tl.static_range(4):
        ch_off = ch * HEAD_DIM
        val = tl.load(kv_in_ptr + in_base + ch_off + d_offs, mask=d_mask, other=0.0)
        tl.store(buffer_ptr + dst_base + ch_off + d_offs, val, mask=d_mask)


@triton.jit
def _c128_decode_kernel(
    kv_in_ptr,
    out_ptr,
    buffer_ptr,
    ape_ptr,
    indices_ptr,
    seq_lens_ptr,
    extra_ptr,
    kv_in_row_stride,
    out_row_stride,
    buffer_page_stride,
    buffer_slot_stride,
    ape_row_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    bid = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    index = tl.load(indices_ptr + bid).to(tl.int64)
    index_prev = tl.load(extra_ptr + bid).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + bid).to(tl.int32)
    write_slot = (seq_len + 127) % 128
    in_base = bid.to(tl.int64) * kv_in_row_stride
    dst_base = index * buffer_page_stride + write_slot.to(tl.int64) * buffer_slot_stride

    for ch in tl.static_range(2):
        ch_off = ch * HEAD_DIM
        val = tl.load(kv_in_ptr + in_base + ch_off + d_offs, mask=d_mask, other=0.0)
        tl.store(buffer_ptr + dst_base + ch_off + d_offs, val, mask=d_mask)

    NEG_BIG: tl.constexpr = -1.0e9
    running_max = tl.full((BLOCK_D,), NEG_BIG, tl.float32)
    running_sum = tl.zeros((BLOCK_D,), tl.float32)
    weighted = tl.zeros((BLOCK_D,), tl.float32)

    for chunk_start in tl.static_range(0, 128, BLOCK_S):
        slot_offs = chunk_start + tl.arange(0, BLOCK_S)
        src_pos = seq_len - 128 + slot_offs
        valid = src_pos >= 0
        is_input = slot_offs == 127
        write_pos = ((seq_len - 1) // 128) * 128
        pages = tl.where(src_pos < write_pos, index_prev, index)
        slot_in_page = src_pos % 128
        slot_bases = (
            pages * buffer_page_stride + slot_in_page.to(tl.int64) * buffer_slot_stride
        )
        kv_tile = tl.load(
            buffer_ptr + slot_bases[:, None] + d_offs[None, :],
            mask=valid[:, None] & (~is_input)[:, None] & d_mask[None, :],
            other=0.0,
        )
        score_tile = tl.load(
            buffer_ptr + slot_bases[:, None] + HEAD_DIM + d_offs[None, :],
            mask=valid[:, None] & (~is_input)[:, None] & d_mask[None, :],
            other=NEG_BIG,
        )
        kv_input_tile = tl.load(
            kv_in_ptr + in_base + d_offs[None, :],
            mask=valid[:, None] & is_input[:, None] & d_mask[None, :],
            other=0.0,
        )
        score_input_tile = tl.load(
            kv_in_ptr + in_base + HEAD_DIM + d_offs[None, :],
            mask=valid[:, None] & is_input[:, None] & d_mask[None, :],
            other=NEG_BIG,
        )
        kv_tile = tl.where(is_input[:, None], kv_input_tile, kv_tile)
        score_tile = tl.where(is_input[:, None], score_input_tile, score_tile)
        bias_tile = tl.load(
            ape_ptr + slot_offs[:, None] * ape_row_stride + d_offs[None, :],
            mask=d_mask[None, :],
            other=0.0,
        )
        s = score_tile + bias_tile
        local_max = tl.max(s, axis=0)
        new_max = tl.maximum(running_max, local_max)
        exp_s = tl.exp(s - new_max[None, :])
        exp_s = tl.where(valid[:, None], exp_s, 0.0)
        factor = tl.exp(running_max - new_max)
        running_sum = running_sum * factor + tl.sum(exp_s, axis=0)
        weighted = weighted * factor + tl.sum(kv_tile * exp_s, axis=0)
        running_max = new_max

    tl.store(
        out_ptr + bid.to(tl.int64) * out_row_stride + d_offs,
        weighted / running_sum,
        mask=d_mask,
    )


@triton.jit
def _c128_prefill_compress_kernel(
    kv_in_ptr,
    out_ptr,
    buffer_ptr,
    ape_ptr,
    indices_ptr,
    plan_ptr,
    kv_in_row_stride,
    out_row_stride,
    buffer_page_stride,
    buffer_slot_stride,
    ape_row_stride,
    plan_row_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    plan_base = plan_ptr + pid_p * plan_row_stride
    ragged_id = tl.load(plan_base + 0).to(tl.int32)
    batch_id = tl.load(plan_base + 1).to(tl.int32)
    position = tl.load(plan_base + 2).to(tl.int32)
    window_len = tl.load(plan_base + 3).to(tl.int32)
    if ragged_id < 0:
        return

    index = tl.load(indices_ptr + batch_id).to(tl.int64)
    NEG_BIG: tl.constexpr = -1.0e9
    running_max = tl.full((BLOCK_D,), NEG_BIG, tl.float32)
    running_sum = tl.zeros((BLOCK_D,), tl.float32)
    weighted = tl.zeros((BLOCK_D,), tl.float32)

    for chunk_start in tl.static_range(0, 128, BLOCK_S):
        slot_offs = chunk_start + tl.arange(0, BLOCK_S)
        is_state = slot_offs < window_len
        src_pos = position - 127 + slot_offs
        state_valid = is_state & (src_pos >= 0)
        slot_bases = (
            index * buffer_page_stride + slot_offs.to(tl.int64) * buffer_slot_stride
        )
        in_rows = ragged_id - (127 - slot_offs)
        in_rows_safe = tl.where(is_state, tl.zeros_like(in_rows), in_rows)
        in_bases = in_rows_safe.to(tl.int64) * kv_in_row_stride

        kv_state = tl.load(
            buffer_ptr + slot_bases[:, None] + d_offs[None, :],
            mask=state_valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        score_state = tl.load(
            buffer_ptr + slot_bases[:, None] + HEAD_DIM + d_offs[None, :],
            mask=state_valid[:, None] & d_mask[None, :],
            other=NEG_BIG,
        )
        kv_input = tl.load(
            kv_in_ptr + in_bases[:, None] + d_offs[None, :],
            mask=(~is_state)[:, None] & d_mask[None, :],
            other=0.0,
        )
        score_input = tl.load(
            kv_in_ptr + in_bases[:, None] + HEAD_DIM + d_offs[None, :],
            mask=(~is_state)[:, None] & d_mask[None, :],
            other=NEG_BIG,
        )
        kv_tile = tl.where(is_state[:, None], kv_state, kv_input)
        score_tile = tl.where(is_state[:, None], score_state, score_input)
        bias_tile = tl.load(
            ape_ptr + slot_offs[:, None] * ape_row_stride + d_offs[None, :],
            mask=d_mask[None, :],
            other=0.0,
        )

        s = score_tile + bias_tile
        local_max = tl.max(s, axis=0)
        new_max = tl.maximum(running_max, local_max)
        exp_s = tl.exp(s - new_max[None, :])
        # Keep input-path entries valid; only state-path entries need src_pos guard.
        valid = state_valid | (~is_state)
        exp_s = tl.where(valid[:, None], exp_s, 0.0)
        factor = tl.exp(running_max - new_max)
        running_sum = running_sum * factor + tl.sum(exp_s, axis=0)
        weighted = weighted * factor + tl.sum(kv_tile * exp_s, axis=0)
        running_max = new_max

    tl.store(
        out_ptr + ragged_id.to(tl.int64) * out_row_stride + d_offs,
        weighted / running_sum,
        mask=d_mask,
    )


@triton.jit
def _c128_prefill_write_kernel(
    kv_in_ptr,
    buffer_ptr,
    indices_ptr,
    plan_ptr,
    kv_in_row_stride,
    buffer_page_stride,
    buffer_slot_stride,
    plan_row_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    plan_base = plan_ptr + pid_p * plan_row_stride
    ragged_id = tl.load(plan_base + 0).to(tl.int32)
    batch_id = tl.load(plan_base + 1).to(tl.int32)
    position = tl.load(plan_base + 2).to(tl.int32)
    if ragged_id < 0:
        return

    index = tl.load(indices_ptr + batch_id).to(tl.int64)
    slot = position % 128
    in_base = ragged_id.to(tl.int64) * kv_in_row_stride
    dst_base = index * buffer_page_stride + slot.to(tl.int64) * buffer_slot_stride

    for ch in tl.static_range(2):
        ch_off = ch * HEAD_DIM
        val = tl.load(kv_in_ptr + in_base + ch_off + d_offs, mask=d_mask, other=0.0)
        tl.store(buffer_ptr + dst_base + ch_off + d_offs, val, mask=d_mask)


@triton.jit
def _compress_norm_rope_kernel(
    kv_ptr,
    weight_ptr,
    freqs_ptr,
    handle_ptr,
    eps,
    kv_row_stride,
    freqs_row_stride,
    plan_row_stride,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROPE_PAIR_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    IS_DECODE: tl.constexpr,
):
    work_id = tl.program_id(0)

    if IS_DECODE:
        row = work_id
        seq_len = tl.load(handle_ptr + work_id).to(tl.int32)
        position = ((seq_len - 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    else:
        plan_base = handle_ptr + work_id * plan_row_stride
        row = tl.load(plan_base + 0).to(tl.int32)
        plan_position = tl.load(plan_base + 2).to(tl.int32)
        if row < 0:
            return
        position = plan_position + 1 - COMPRESS_RATIO

    base = row.to(tl.int64) * kv_row_stride
    offs = tl.arange(0, HEAD_BLOCK)
    mask = offs < HEAD_DIM
    x = tl.load(kv_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rms_inv = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
    x_normed = x * rms_inv * w

    rope_start: tl.constexpr = HEAD_DIM - ROPE_DIM
    pair_offs = tl.arange(0, ROPE_PAIR_BLOCK)
    pair_mask = pair_offs < (ROPE_DIM // 2)
    x_real = tl.load(
        kv_ptr + base + rope_start + 2 * pair_offs,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    x_imag = tl.load(
        kv_ptr + base + rope_start + 2 * pair_offs + 1,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    w_real = tl.load(
        weight_ptr + rope_start + 2 * pair_offs,
        mask=pair_mask,
        other=1.0,
    ).to(tl.float32)
    w_imag = tl.load(
        weight_ptr + rope_start + 2 * pair_offs + 1,
        mask=pair_mask,
        other=1.0,
    ).to(tl.float32)
    x_real = x_real * rms_inv * w_real
    x_imag = x_imag * rms_inv * w_imag

    freq_base = position.to(tl.int64) * freqs_row_stride
    f_real = tl.load(freqs_ptr + freq_base + 2 * pair_offs, mask=pair_mask, other=0.0)
    f_imag = tl.load(
        freqs_ptr + freq_base + 2 * pair_offs + 1,
        mask=pair_mask,
        other=0.0,
    )
    out_real = x_real * f_real - x_imag * f_imag
    out_imag = x_real * f_imag + x_imag * f_real

    tl.store(kv_ptr + base + offs, x_normed, mask=mask & (offs < rope_start))
    tl.store(kv_ptr + base + rope_start + 2 * pair_offs, out_real, mask=pair_mask)
    tl.store(kv_ptr + base + rope_start + 2 * pair_offs + 1, out_imag, mask=pair_mask)


@triton.jit
def _compress_norm_rope_hadamard_kernel(
    kv_ptr,
    weight_ptr,
    freqs_ptr,
    handle_ptr,
    eps,
    hadamard_scale,
    kv_row_stride,
    freqs_row_stride,
    plan_row_stride,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROPE_PAIR_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    IS_DECODE: tl.constexpr,
    LOG2_HEAD_DIM: tl.constexpr,
):
    work_id = tl.program_id(0)

    if IS_DECODE:
        row = work_id
        seq_len = tl.load(handle_ptr + work_id).to(tl.int32)
        position = ((seq_len - 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    else:
        plan_base = handle_ptr + work_id * plan_row_stride
        row = tl.load(plan_base + 0).to(tl.int32)
        plan_position = tl.load(plan_base + 2).to(tl.int32)
        if row < 0:
            return
        position = plan_position + 1 - COMPRESS_RATIO

    base = row.to(tl.int64) * kv_row_stride
    offs = tl.arange(0, HEAD_BLOCK)
    mask = offs < HEAD_DIM
    x = tl.load(kv_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rms_inv = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
    x_normed = x * rms_inv * w

    rope_start: tl.constexpr = HEAD_DIM - ROPE_DIM
    pair_offs = tl.arange(0, ROPE_PAIR_BLOCK)
    pair_mask = pair_offs < (ROPE_DIM // 2)
    x_real = tl.load(
        kv_ptr + base + rope_start + 2 * pair_offs,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    x_imag = tl.load(
        kv_ptr + base + rope_start + 2 * pair_offs + 1,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    w_real = tl.load(
        weight_ptr + rope_start + 2 * pair_offs,
        mask=pair_mask,
        other=1.0,
    ).to(tl.float32)
    w_imag = tl.load(
        weight_ptr + rope_start + 2 * pair_offs + 1,
        mask=pair_mask,
        other=1.0,
    ).to(tl.float32)
    x_real = x_real * rms_inv * w_real
    x_imag = x_imag * rms_inv * w_imag

    freq_base = position.to(tl.int64) * freqs_row_stride
    f_real = tl.load(freqs_ptr + freq_base + 2 * pair_offs, mask=pair_mask, other=0.0)
    f_imag = tl.load(
        freqs_ptr + freq_base + 2 * pair_offs + 1,
        mask=pair_mask,
        other=0.0,
    )
    out_real = x_real * f_real - x_imag * f_imag
    out_imag = x_real * f_imag + x_imag * f_real

    # Store norm+rope result to kv_ptr (will be used for butterfly stages)
    tl.store(kv_ptr + base + offs, x_normed, mask=mask & (offs < rope_start))
    tl.store(kv_ptr + base + rope_start + 2 * pair_offs, out_real, mask=pair_mask)
    tl.store(kv_ptr + base + rope_start + 2 * pair_offs + 1, out_imag, mask=pair_mask)

    # Walsh-Hadamard butterfly transform via store-reload through L1 cache.
    # Barriers are required because multiple warps share the same row in memory;
    # without them a fast warp can overwrite a partner value before a slow warp reads it.
    for stage in tl.static_range(LOG2_HEAD_DIM):
        stride = 1 << stage
        is_even = ((offs >> stage) & 1) == 0
        partner = tl.where(is_even, offs + stride, offs - stride)
        tl.debug_barrier()
        x_self = tl.load(kv_ptr + base + offs, mask=mask)
        x_partner = tl.load(kv_ptr + base + partner, mask=mask)
        result = tl.where(is_even, x_self + x_partner, x_partner - x_self)
        if stage == LOG2_HEAD_DIM - 1:
            result = result * hadamard_scale
        tl.debug_barrier()
        tl.store(kv_ptr + base + offs, result, mask=mask)


def _plan_as_i32(plan: torch.Tensor) -> torch.Tensor:
    assert plan.dtype == torch.uint8 and plan.dim() == 2 and plan.shape[1] == 16
    return plan.view(torch.int32).view(-1, 4)


def _block_d(head_dim: int) -> int:
    return min(32, triton.next_power_of_2(head_dim))


def _check_common(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    head_dim: int,
    compress_ratio: int,
) -> None:
    coff = 2 if compress_ratio == 4 else 1
    assert kv_score_input.is_cuda and kv_score_buffer.is_cuda
    assert kv_score_input.dim() == 2 and kv_score_input.dtype == torch.float32
    assert kv_score_input.shape[1] == 2 * coff * head_dim
    assert kv_score_buffer.dim() == 3 and kv_score_buffer.dtype == torch.float32
    assert kv_score_buffer.shape[1:] == (compress_ratio, 2 * coff * head_dim)
    assert out.shape == (kv_score_input.shape[0], head_dim)
    assert out.dtype == torch.float32 and out.is_cuda
    assert ape.shape == (compress_ratio * coff, head_dim)
    assert ape.dtype == torch.float32 and ape.is_cuda
    assert indices.dtype == torch.int32 and indices.is_cuda


def _is_decode_plan(plan: Union[CompressorDecodePlan, CompressorPrefillPlan]) -> bool:
    return isinstance(plan, CompressorDecodePlan)


def hip_compress_forward(
    *,
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    extra_data: Optional[torch.Tensor],
    head_dim: int,
    compress_ratio: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if compress_ratio not in (4, 128):
        raise ValueError(f"unsupported {compress_ratio=}")
    if out is None:
        out = kv_score_input.new_empty((kv_score_input.shape[0], head_dim))
    is_decode = _is_decode_plan(plan)
    if not is_decode:
        out.fill_(10000.0)

    _check_common(
        kv_score_buffer,
        kv_score_input,
        out,
        ape,
        indices,
        head_dim,
        compress_ratio,
    )

    BLOCK_D = _block_d(head_dim)
    num_d_chunks = triton.cdiv(head_dim, BLOCK_D)

    if is_decode:
        seq_lens = plan.seq_lens
        assert seq_lens.dtype == torch.int32 and seq_lens.is_cuda
        assert seq_lens.shape == indices.shape
        grid = (seq_lens.numel(), num_d_chunks)
        if compress_ratio == 4:
            assert extra_data is not None
            assert extra_data.shape == (seq_lens.numel(), 1)
            _c4_decode_kernel[grid](
                kv_score_input,
                out,
                kv_score_buffer,
                ape,
                indices,
                seq_lens,
                extra_data,
                kv_score_input.stride(0),
                out.stride(0),
                kv_score_buffer.stride(0),
                kv_score_buffer.stride(1),
                ape.stride(0),
                HEAD_DIM=head_dim,
                BLOCK_D=BLOCK_D,
            )
        else:
            assert extra_data is not None
            assert extra_data.shape == seq_lens.shape
            _c128_decode_kernel[grid](
                kv_score_input,
                out,
                kv_score_buffer,
                ape,
                indices,
                seq_lens,
                extra_data,
                kv_score_input.stride(0),
                out.stride(0),
                kv_score_buffer.stride(0),
                kv_score_buffer.stride(1),
                ape.stride(0),
                HEAD_DIM=head_dim,
                BLOCK_D=BLOCK_D,
                BLOCK_S=64,
            )
        return out

    compress_plan = _plan_as_i32(plan.compress_plan)
    write_plan = _plan_as_i32(plan.write_plan)
    if compress_ratio == 4:
        assert extra_data is not None
        assert extra_data.dim() == 2 and extra_data.shape[1] == 4
        compress_grid = (compress_plan.shape[0], num_d_chunks)
        write_grid = (write_plan.shape[0], num_d_chunks)
        _c4_prefill_compress_kernel[compress_grid](
            kv_score_input,
            out,
            kv_score_buffer,
            ape,
            indices,
            extra_data,
            compress_plan,
            kv_score_input.stride(0),
            out.stride(0),
            kv_score_buffer.stride(0),
            kv_score_buffer.stride(1),
            ape.stride(0),
            compress_plan.stride(0),
            HEAD_DIM=head_dim,
            BLOCK_D=BLOCK_D,
        )
        _c4_prefill_write_kernel[write_grid](
            kv_score_input,
            kv_score_buffer,
            indices,
            extra_data,
            write_plan,
            kv_score_input.stride(0),
            kv_score_buffer.stride(0),
            kv_score_buffer.stride(1),
            write_plan.stride(0),
            HEAD_DIM=head_dim,
            BLOCK_D=BLOCK_D,
        )
    else:
        load_indices = indices if extra_data is None else extra_data
        assert load_indices.dim() == 1 and load_indices.dtype == torch.int32
        compress_grid = (compress_plan.shape[0], num_d_chunks)
        write_grid = (write_plan.shape[0], num_d_chunks)
        _c128_prefill_compress_kernel[compress_grid](
            kv_score_input,
            out,
            kv_score_buffer,
            ape,
            load_indices,
            compress_plan,
            kv_score_input.stride(0),
            out.stride(0),
            kv_score_buffer.stride(0),
            kv_score_buffer.stride(1),
            ape.stride(0),
            compress_plan.stride(0),
            HEAD_DIM=head_dim,
            BLOCK_D=BLOCK_D,
            BLOCK_S=64,
        )
        _c128_prefill_write_kernel[write_grid](
            kv_score_input,
            kv_score_buffer,
            indices,
            write_plan,
            kv_score_input.stride(0),
            kv_score_buffer.stride(0),
            kv_score_buffer.stride(1),
            write_plan.stride(0),
            HEAD_DIM=head_dim,
            BLOCK_D=BLOCK_D,
        )
    return out


def hip_compress_fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
) -> None:
    assert kv.dim() == 2 and kv.stride(-1) == 1
    assert weight.shape == (kv.shape[1],)
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    head_dim = kv.shape[1]
    rope_dim = freqs_real.shape[-1]
    assert head_dim >= rope_dim and rope_dim % 2 == 0

    is_decode = _is_decode_plan(plan)
    if is_decode:
        handle = plan.seq_lens
    else:
        handle = _plan_as_i32(plan.compress_plan)

    if handle.numel() == 0:
        return

    HEAD_BLOCK = triton.next_power_of_2(head_dim)
    ROPE_PAIR_BLOCK = max(triton.next_power_of_2(rope_dim // 2), 1)
    _compress_norm_rope_kernel[(handle.shape[0],)](
        kv,
        weight,
        freqs_real,
        handle,
        eps,
        kv.stride(0),
        freqs_real.stride(0),
        handle.stride(0) if not is_decode else 0,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        HEAD_BLOCK=HEAD_BLOCK,
        ROPE_PAIR_BLOCK=ROPE_PAIR_BLOCK,
        COMPRESS_RATIO=plan.compress_ratio,
        IS_DECODE=is_decode,
    )


def hip_compress_fused_norm_rope_hadamard_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    head_dim: int,
) -> None:
    assert kv.dim() == 2 and kv.stride(-1) == 1
    assert weight.shape == (kv.shape[1],)
    assert kv.shape[1] == head_dim
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    rope_dim = freqs_real.shape[-1]
    assert head_dim >= rope_dim and rope_dim % 2 == 0
    assert (head_dim & (head_dim - 1)) == 0, "head_dim must be power of 2"

    is_decode = _is_decode_plan(plan)
    if is_decode:
        handle = plan.seq_lens
    else:
        handle = _plan_as_i32(plan.compress_plan)

    if handle.numel() == 0:
        return

    import math

    log2_head_dim = int(math.log2(head_dim))
    hadamard_scale = head_dim**-0.5

    HEAD_BLOCK = triton.next_power_of_2(head_dim)
    ROPE_PAIR_BLOCK = max(triton.next_power_of_2(rope_dim // 2), 1)
    _compress_norm_rope_hadamard_kernel[(handle.shape[0],)](
        kv,
        weight,
        freqs_real,
        handle,
        eps,
        hadamard_scale,
        kv.stride(0),
        freqs_real.stride(0),
        handle.stride(0) if not is_decode else 0,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        HEAD_BLOCK=HEAD_BLOCK,
        ROPE_PAIR_BLOCK=ROPE_PAIR_BLOCK,
        COMPRESS_RATIO=plan.compress_ratio,
        IS_DECODE=is_decode,
        LOG2_HEAD_DIM=log2_head_dim,
    )
