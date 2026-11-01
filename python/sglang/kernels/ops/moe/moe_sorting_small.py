# SPDX-License-Identifier: Apache-2.0
"""Single-launch MoE sorting (+ stage1 mxfp8 activation quant) for decode-sized
batches on the aiter MoE runner, installed as runtime patches on ``aiter.fused_moe``.

Output layouts match the aiter kernels bit-for-bit:
  sorted_ids[i]        = (topk_slot << 24) | token   (padding: (topk << 24) | M)
  sorted_weights       = pair weight                 (padding: 0)
  sorted_expert_ids[b] = expert of block b
  num_valid_ids        = [num_blocks * block_size, M]
  quant a1             = per-token fp8 rows; e8m0 RoundUp scale byte per
                         (sorted_row, group) at aiter's mx_scale_shuffle_idx address
"""

from __future__ import annotations

import contextlib
import functools
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.k3_route_sort import _mx_e8m0, _mx_scale_col, _mx_scale_row
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.kernels.ops.moe.k3_route_sort_mid import RouteState

logger = logging.getLogger(__name__)


# P <= 64: one sort CTA does the whole P x P rank compare
# per-M ints stay unspecialized so prefill sizes reuse the graph-capture compiles
@triton.jit(do_not_specialize=["M", "moe_buf_numel", "num_buf"])
def _moe_sorting_small_kernel(
    topk_ids_ptr,  # [M, topk] i32
    topk_weights_ptr,  # [M, topk] fp32
    sorted_ids_ptr,  # [max_padded] i32
    sorted_weights_ptr,  # [max_padded] fp32
    sorted_expert_ids_ptr,  # [max_blocks] i32
    num_valid_ids_ptr,  # [2] i32
    moe_buf_ptr,
    moe_buf_numel,
    qx_ptr,  # [M, N_COLS] activations to mx-quantize (EMIT_MX only)
    stride_qx,
    qout_ptr,  # [M, N_COLS] fp8 out
    qscale_ptr,  # swizzled e8m0 bytes, one per (sorted_row, group)
    M,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    P_POW2: tl.constexpr,  # >= M * topk
    PAD_POW2: tl.constexpr,  # >= (M * topk) * BLOCK_SIZE (worst-case padded len)
    BUF_BLOCK: tl.constexpr,
    num_buf,  # buf-zero CTAs occupy pids [1, num_buf]
    EMIT_MX: tl.constexpr,  # also emit the mxfp8 quant of qx (group_size 32)
    N_COLS: tl.constexpr,
    QCHUNK: tl.constexpr,  # columns per quant iteration (multiple of 32)
    SCALEN_PAD: tl.constexpr,  # ceil(N_COLS/32 / 8) * 8
):
    pid = tl.program_id(0)
    if pid > 0 and pid <= num_buf:
        offs = (pid - 1) * BUF_BLOCK + tl.arange(0, BUF_BLOCK)
        tl.store(
            moe_buf_ptr + offs,
            tl.zeros((BUF_BLOCK,), moe_buf_ptr.dtype.element_ty),
            mask=offs < moe_buf_numel,
        )
        return

    P = M * TOPK
    offs_p = tl.arange(0, P_POW2)
    # the quant CTAs recompute the P x P sort math instead of waiting on pid 0
    mask_p = offs_p < P
    # sentinel expert keeps inactive lanes out of every "smaller expert" count
    e = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=0x7FFFFFFF)
    w = tl.load(topk_weights_ptr + offs_p, mask=mask_p, other=0.0)
    token = offs_p // TOPK
    slot = offs_p % TOPK

    # Stable rank within the pair's expert, and per-expert pair count.
    same = e[:, None] == e[None, :]
    rank = tl.sum(tl.where(same & (offs_p[None, :] < offs_p[:, None]), 1, 0), axis=1)
    cnt = tl.sum(tl.where(same & mask_p[None, :], 1, 0), axis=1)
    blocks_of_e = (cnt + BLOCK_SIZE - 1) // BLOCK_SIZE
    is_leader = (rank == 0) & mask_p

    # leaders carry their expert's block count
    smaller = e[None, :] < e[:, None]
    blocks_before = tl.sum(
        tl.where(smaller & is_leader[None, :], blocks_of_e[None, :], 0), axis=1
    )
    dest = blocks_before * BLOCK_SIZE + rank

    if EMIT_MX and pid > num_buf:
        # quant CTA: one (pair, column-chunk) slice, mirroring fused_dynamic_mxfp8_quant_moe_sort
        q_id = pid - num_buf - 1
        CHUNKS: tl.constexpr = N_COLS // QCHUNK
        p = q_id // CHUNKS
        c0 = (q_id % CHUNKS) * QCHUNK

        if p < P:
            offs_q = tl.arange(0, QCHUNK)
            offs_g = tl.arange(0, QCHUNK // 32)
            token_p = tl.sum(tl.where(offs_p == p, token, 0), axis=0)
            dest_p = tl.sum(tl.where(offs_p == p, dest, 0), axis=0)
            base_sw = _mx_scale_row(dest_p, SCALEN_PAD)
            x = tl.load(qx_ptr + token_p * stride_qx + c0 + offs_q).to(tl.float32)
            x2 = tl.reshape(x, (QCHUNK // 32, 32))
            exp, scale = _mx_e8m0(x2)
            if p % TOPK == 0:
                # one fp8 out row per token (pairs are token-major)
                q = tl.clamp(x2 / scale[:, None], -448.0, 448.0)
                tl.store(
                    qout_ptr + token_p * N_COLS + c0 + offs_q,
                    tl.reshape(q, (QCHUNK,)).to(qout_ptr.dtype.element_ty),
                )
            y = c0 // 32 + offs_g
            sw = base_sw + _mx_scale_col(y)
            tl.store(qscale_ptr + sw, exp.to(tl.uint8))
        return

    total_blocks = tl.sum(tl.where(is_leader, blocks_of_e, 0), axis=0)
    num_valid = total_blocks * BLOCK_SIZE

    # pid 0: pad the whole used region first, then scatter the real pairs over it
    offs_pad = tl.arange(0, PAD_POW2)
    pad_mask = offs_pad < num_valid
    pad_val = (TOPK << 24) | M
    tl.store(
        sorted_ids_ptr + offs_pad,
        tl.full((PAD_POW2,), 0, tl.int32) + pad_val,
        mask=pad_mask,
    )
    tl.store(
        sorted_weights_ptr + offs_pad, tl.zeros((PAD_POW2,), tl.float32), mask=pad_mask
    )
    tl.debug_barrier()
    tl.store(sorted_ids_ptr + dest, (slot << 24) | token, mask=mask_p)
    tl.store(sorted_weights_ptr + dest, w, mask=mask_p)

    # leaders write their expert into each of their blocks: at most 2 with P <= 2 * BLOCK_SIZE
    for j in tl.static_range(2):
        bm = is_leader & (j < blocks_of_e)
        tl.store(sorted_expert_ids_ptr + blocks_before + j, e, mask=bm)

    tl.store(
        num_valid_ids_ptr + tl.arange(0, 2),
        tl.where(tl.arange(0, 2) == 0, num_valid, M),
    )


@triton.jit
def _expert_chunk_blocks(e, offs_chunk, BLOCK_SIZE: tl.constexpr):
    cnt = tl.sum((e[:, None] == offs_chunk[None, :]).to(tl.int32), axis=0)
    return cnt, (cnt + BLOCK_SIZE - 1) // BLOCK_SIZE


# per-M ints stay unspecialized so prefill sizes reuse the graph-capture compiles
@triton.jit(do_not_specialize=["M", "moe_buf_numel", "num_buf"])
def _moe_sorting_small_kernel_distributed(
    topk_ids_ptr,  # [M, topk] i32
    topk_weights_ptr,  # [M, topk] fp32
    sorted_ids_ptr,  # [max_padded] i32
    sorted_weights_ptr,  # [max_padded] fp32
    sorted_expert_ids_ptr,  # [max_blocks] i32
    num_valid_ids_ptr,  # [2] i32
    moe_buf_ptr,
    moe_buf_numel,
    qx_ptr,  # [M, N_COLS] activations to mx-quantize (EMIT_MX only)
    stride_qx,
    qout_ptr,  # [M, N_COLS] fp8 out
    qscale_ptr,  # swizzled e8m0 bytes, one per (sorted_row, group)
    M,
    num_experts,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    P_POW2: tl.constexpr,  # >= M * topk
    P_CHUNK: tl.constexpr,  # tile width for pairwise/expert-space loops
    E_CEIL: tl.constexpr,  # num_experts rounded up to P_CHUNK
    PAD_POW2: tl.constexpr,  # >= (M * topk) * BLOCK_SIZE (worst-case padded len)
    BUF_BLOCK: tl.constexpr,
    num_buf,  # buf-zero CTAs occupy pids [1, num_buf]
    EMIT_MX: tl.constexpr,  # also emit the mxfp8 quant of qx (group_size 32)
    N_COLS: tl.constexpr,
    QCHUNK: tl.constexpr,  # columns per quant iteration (multiple of 32)
    SCALEN_PAD: tl.constexpr,  # ceil(N_COLS/32 / 8) * 8
):
    pid = tl.program_id(0)
    P = M * TOPK
    offs_p = tl.arange(0, P_POW2)
    mask_p = offs_p < P
    # sentinel expert keeps inactive lanes out of every "smaller expert" count
    e = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=0x7FFFFFFF)

    # one writer per byte: expert-chunk CTAs, then num_buf zero-fill CTAs, then one CTA per pair
    NUM_ECHUNK: tl.constexpr = E_CEIL // P_CHUNK

    if pid < NUM_ECHUNK:
        # expert-chunk CTA: expert-id table, block-tail padding and (last chunk) num_valid
        x0 = pid * P_CHUNK
        offs_x = x0 + tl.arange(0, P_CHUNK)
        cnt_x, blocks_x = _expert_chunk_blocks(e, offs_x, BLOCK_SIZE)
        # a runtime `if` inside static_range mis-lowers, so accumulate under a uniform mask
        blocks_before_chunk = tl.zeros((), tl.int32)
        for y0 in tl.static_range(0, E_CEIL, P_CHUNK):
            offs_y = y0 + tl.arange(0, P_CHUNK)
            cnt_y, blocks_y = _expert_chunk_blocks(e, offs_y, BLOCK_SIZE)
            blocks_before_chunk += tl.sum(tl.where(offs_y < x0, blocks_y, 0), axis=0)
        # exclusive prefix within the chunk
        prefix_x = tl.sum(
            tl.where(offs_x[None, :] < offs_x[:, None], blocks_x[None, :], 0), axis=1
        )
        bb_x = blocks_before_chunk + prefix_x

        MAX_BLOCKS_PER_EXPERT: tl.constexpr = (P_POW2 + BLOCK_SIZE - 1) // BLOCK_SIZE
        for j in tl.static_range(MAX_BLOCKS_PER_EXPERT):
            bm = (offs_x < num_experts) & (j < blocks_x)
            tl.store(sorted_expert_ids_ptr + bb_x + j, offs_x, mask=bm)

        pad_val = (TOPK << 24) | M
        for j in tl.static_range(MAX_BLOCKS_PER_EXPERT * BLOCK_SIZE):
            pm = (j >= cnt_x) & (j < blocks_x * BLOCK_SIZE)
            tl.store(sorted_ids_ptr + bb_x * BLOCK_SIZE + j, pad_val, mask=pm)
            tl.store(sorted_weights_ptr + bb_x * BLOCK_SIZE + j, 0.0, mask=pm)

        if pid == NUM_ECHUNK - 1:
            num_valid = (blocks_before_chunk + tl.sum(blocks_x, axis=0)) * BLOCK_SIZE
            tl.store(
                num_valid_ids_ptr + tl.arange(0, 2),
                tl.where(tl.arange(0, 2) == 0, num_valid, M),
            )
        return

    if pid < NUM_ECHUNK + num_buf:
        offs = (pid - NUM_ECHUNK) * BUF_BLOCK + tl.arange(0, BUF_BLOCK)
        tl.store(
            moe_buf_ptr + offs,
            tl.zeros((BUF_BLOCK,), moe_buf_ptr.dtype.element_ty),
            mask=offs < moe_buf_numel,
        )
        return

    # per-pair CTA: dest = blocks of smaller experts * BLOCK_SIZE + stable rank in its expert
    p = pid - NUM_ECHUNK - num_buf
    if p >= P:
        return

    e_p = tl.sum(tl.where(offs_p == p, e, 0), axis=0)
    rank_p = tl.sum(tl.where((offs_p < p) & (e == e_p), 1, 0), axis=0)
    bb_p = tl.zeros((), tl.int32)
    for x0 in tl.static_range(0, E_CEIL, P_CHUNK):
        offs_x = x0 + tl.arange(0, P_CHUNK)
        cnt_x, blocks_x = _expert_chunk_blocks(e, offs_x, BLOCK_SIZE)
        bb_p += tl.sum(tl.where(offs_x < e_p, blocks_x, 0), axis=0)
    dest_p = bb_p * BLOCK_SIZE + rank_p

    token_p2 = p // TOPK
    slot_p = p % TOPK
    w_p = tl.load(topk_weights_ptr + p)
    tl.store(sorted_ids_ptr + dest_p, (slot_p << 24) | token_p2)
    tl.store(sorted_weights_ptr + dest_p, w_p)

    if EMIT_MX:
        # mirrors fused_dynamic_mxfp8_quant_moe_sort: e8m0 RoundUp scale, swizzled address
        token_p = p // TOPK
        offs_q = tl.arange(0, QCHUNK)
        offs_g = tl.arange(0, QCHUNK // 32)
        base_sw = _mx_scale_row(dest_p, SCALEN_PAD)
        for cc in tl.static_range(N_COLS // QCHUNK):
            c0 = cc * QCHUNK
            x = tl.load(qx_ptr + token_p * stride_qx + c0 + offs_q).to(tl.float32)
            x2 = tl.reshape(x, (QCHUNK // 32, 32))
            exp, scale = _mx_e8m0(x2)
            if p % TOPK == 0:
                # one fp8 out row per token (pairs are token-major)
                q = tl.clamp(x2 / scale[:, None], -448.0, 448.0)
                tl.store(
                    qout_ptr + token_p * N_COLS + c0 + offs_q,
                    tl.reshape(q, (QCHUNK,)).to(qout_ptr.dtype.element_ty),
                )
            y = c0 // 32 + offs_g
            sw = base_sw + _mx_scale_col(y)
            tl.store(qscale_ptr + sw, exp.to(tl.uint8))


# no-quant path: histogram + bitonic sort keep every pair in registers of one CTA
@triton.jit(do_not_specialize=["M", "moe_buf_numel", "num_buf"])
def _moe_sorting_small_kernel_sorted(
    topk_ids_ptr,  # [M, topk] i32
    topk_weights_ptr,  # [M, topk] fp32
    sorted_ids_ptr,  # [max_padded] i32
    sorted_weights_ptr,  # [max_padded] fp32
    sorted_expert_ids_ptr,  # [max_blocks] i32
    num_valid_ids_ptr,  # [2] i32
    moe_buf_ptr,
    moe_buf_numel,
    M,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    P_POW2: tl.constexpr,  # >= M * topk
    E_POW2: tl.constexpr,  # > num_experts; bin E_POW2 - 1 holds the inactive lanes
    PAD_POW2: tl.constexpr,  # >= (M * topk) * BLOCK_SIZE (worst-case padded len)
    BUF_BLOCK: tl.constexpr,
    num_buf,  # buf-zero CTAs occupy pids [1, num_buf]
):
    pid = tl.program_id(0)
    if pid > 0:
        offs = (pid - 1) * BUF_BLOCK + tl.arange(0, BUF_BLOCK)
        tl.store(
            moe_buf_ptr + offs,
            tl.zeros((BUF_BLOCK,), moe_buf_ptr.dtype.element_ty),
            mask=offs < moe_buf_numel,
        )
        return

    P = M * TOPK
    offs_p = tl.arange(0, P_POW2)
    e = tl.load(topk_ids_ptr + offs_p, mask=offs_p < P, other=E_POW2 - 1)
    offs_e = tl.arange(0, E_POW2)
    cnt = tl.where(offs_e < E_POW2 - 1, tl.histogram(e, E_POW2), 0)
    blocks = (cnt + BLOCK_SIZE - 1) // BLOCK_SIZE
    # sorted position of a pair -> padded dest: add the block-tail padding of smaller experts
    shift = (tl.cumsum(blocks, 0) - blocks) * BLOCK_SIZE - (tl.cumsum(cnt, 0) - cnt)
    num_valid = tl.sum(blocks, 0) * BLOCK_SIZE

    # pad the whole used region first, then scatter the real pairs over it
    offs_pad = tl.arange(0, PAD_POW2)
    pad_mask = offs_pad < num_valid
    tl.store(
        sorted_ids_ptr + offs_pad,
        tl.full((PAD_POW2,), (TOPK << 24) | M, tl.int32),
        mask=pad_mask,
    )
    tl.store(
        sorted_weights_ptr + offs_pad, tl.zeros((PAD_POW2,), tl.float32), mask=pad_mask
    )
    tl.debug_barrier()

    # stable (expert, pair) order; inactive lanes sort last
    key = tl.sort(e * P_POW2 + offs_p)
    es = key // P_POW2
    ps = key % P_POW2
    valid = offs_p < P
    dest = offs_p + tl.gather(shift, tl.where(valid, es, 0), 0)
    tl.store(sorted_ids_ptr + dest, ((ps % TOPK) << 24) | (ps // TOPK), mask=valid)
    tl.store(
        sorted_weights_ptr + dest,
        tl.load(topk_weights_ptr + ps, mask=valid, other=0.0),
        mask=valid,
    )
    # every block holds a real pair; its pairs all write the same expert id
    tl.store(sorted_expert_ids_ptr + dest // BLOCK_SIZE, es, mask=valid)
    tl.store(
        num_valid_ids_ptr + tl.arange(0, 2),
        tl.where(tl.arange(0, 2) == 0, num_valid, M),
    )


def _mid_sort_supported(
    topk_ids, block_size, expert_mask, num_local_tokens, num_experts, qx
):
    from sglang.kernels.ops.moe import k3_route_sort_mid

    if qx is None:
        return False
    m, topk = topk_ids.shape
    return (
        expert_mask is None
        and num_local_tokens is None
        and m * topk > 256  # smaller batches take _run_small_sort
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and qx.shape[0] == m
        and qx.dtype in (torch.bfloat16, torch.float16)
        and k3_route_sort_mid.supported(m, topk, num_experts, block_size, qx.shape[-1])
    )


def _run_mid_sort(topk_ids, topk_weights, num_experts, block_size, moe_buf, qx):
    """16 < M <= 512: (launch A, unless the model already ran it with top-k
    fused in) + launch B of k3_route_sort_mid. Buffer sizes as aiter's."""
    from sglang.kernels.ops.moe import k3_route_sort_mid as mid

    device = topk_ids.device
    m, topk = topk_ids.shape
    max_blocks = (
        topk_ids.numel() + num_experts * block_size - topk + block_size - 1
    ) // block_size
    sorted_ids = torch.empty(max_blocks * block_size, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(
        max_blocks * block_size, dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    st = _k3_sort_plan.get().take_route_state(topk_ids, qx)
    if st is None:
        st = mid.route_quant(
            qx,
            mid.counts_buffer(device, None, num_experts),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_experts=num_experts,
        )
    elif st.topk_weights.data_ptr() != topk_weights.data_ptr():
        st.topk_weights = topk_weights
    quant = mid.sort_scatter(
        st,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        block_size,
        num_experts,
    )
    return (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
    ), quant


def _small_sort_supported(
    topk_ids, block_size, expert_mask, num_local_tokens, num_experts, emit_mx=True
):
    m, topk = topk_ids.shape
    p = m * topk
    return (
        expert_mask is None
        and num_local_tokens is None
        and p <= 256
        # ponytail: the distributed variant scans every 64-expert chunk per pair, so it beats
        # aiter at E=129 but loses at E=513; cutoff untuned in between. The no-quant
        # path past 32 pairs takes the histogram + bitonic sort kernel instead, which
        # does not scan per expert chunk (measured up to E=385, i.e. E_POW2 <= 512).
        # A K3 sort plan fuses the shared GEMM and quant into this sort: wins at any E
        and (
            (p <= 64 and p <= 2 * block_size)
            or num_experts <= 256
            or (not emit_mx and p > 32 and num_experts < 512)
            or _k3_sort_plan.get() is not None
        )
        and topk < 128
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
    )


def _run_small_sort(
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    block_size,
    mx_quant_input,
    num_experts,
):
    m, topk = topk_ids.shape
    p = m * topk
    buf_block = 4096
    emit_mx = mx_quant_input is not None
    if emit_mx:
        n_cols = mx_quant_input.shape[-1]
        scalen_pad = ((n_cols // 32 + 7) // 8) * 8
        max_padded = sorted_ids.shape[0]
        qout = torch.empty(m, n_cols, dtype=torch.float8_e4m3fn, device=topk_ids.device)
        qscale = torch.empty(
            ((max_padded + 31) // 32) * 32,
            scalen_pad,
            dtype=torch.uint8,
            device=topk_ids.device,
        )
    else:
        n_cols, scalen_pad = 32, 8
        qout = qscale = moe_buf  # unused placeholder pointers
    plan = _k3_sort_plan.get()
    if emit_mx and plan is not None:
        from sglang.kernels.ops.moe import k3_route_sort

        if k3_route_sort.sort_quant_supported(m, topk, block_size, n_cols) and (
            moe_buf.numel() in (0, m * n_cols)
        ):
            from sglang.kernels.ops.moe import k3_moe_epi

            # The model planned its shared down GEMM for this top-k; run it in
            # the sort's launch
            pending = plan.take_shared_gemm(topk_ids)
            if pending is not None:
                k3_moe_epi.shared_gemm_sort_quant(
                    pending,
                    topk_ids,
                    topk_weights,
                    sorted_ids,
                    sorted_weights,
                    sorted_expert_ids,
                    num_valid_ids,
                    moe_buf,
                    block_size,
                    mx_quant_input,
                    qout,
                    qscale,
                )
            else:
                k3_route_sort.sort_quant(
                    topk_ids,
                    topk_weights,
                    sorted_ids,
                    sorted_weights,
                    sorted_expert_ids,
                    num_valid_ids,
                    moe_buf,
                    block_size,
                    mx_quant_input,
                    qout,
                    qscale,
                )
            return qout, qscale.view(torch.float8_e8m0fnu)
    num_buf = triton.cdiv(max(moe_buf.numel(), 1), buf_block)
    if not emit_mx and p > 32:
        # beyond one 32-lane tile the sort variant beats both P x P compare kernels
        _moe_sorting_small_kernel_sorted[(1 + num_buf,)](
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            moe_buf.numel(),
            m,
            TOPK=topk,
            BLOCK_SIZE=block_size,
            P_POW2=triton.next_power_of_2(p),
            E_POW2=triton.next_power_of_2(num_experts + 1),
            PAD_POW2=triton.next_power_of_2(p * block_size),
            BUF_BLOCK=buf_block,
            num_buf=num_buf,
            num_warps=4,
        )
        return None
    if p <= 64 and p <= 2 * block_size:
        # compact variant: one sort CTA does the P x P rank compare
        num_quant = (p * (n_cols // _mx_quant_chunk(n_cols))) if emit_mx else 0
        grid = (1 + num_buf + num_quant,)
        _moe_sorting_small_kernel[grid](
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            moe_buf.numel(),
            mx_quant_input if emit_mx else moe_buf,
            mx_quant_input.stride(0) if emit_mx else 0,
            qout,
            qscale,
            m,
            TOPK=topk,
            BLOCK_SIZE=block_size,
            P_POW2=triton.next_power_of_2(p),
            PAD_POW2=triton.next_power_of_2(p * block_size),
            BUF_BLOCK=buf_block,
            num_buf=num_buf,
            EMIT_MX=emit_mx,
            N_COLS=n_cols,
            QCHUNK=_mx_quant_chunk(n_cols) if emit_mx else 32,
            SCALEN_PAD=scalen_pad,
            num_warps=4,
        )
        if emit_mx:
            return qout, qscale.view(torch.float8_e8m0fnu)
        return None
    # distributed variant: 64-wide expert/pair tiles, one writer per byte
    p_chunk = 64
    num_echunk = (num_experts + p_chunk - 1) // p_chunk
    num_pair = p
    grid = (num_echunk + num_buf + num_pair,)
    _moe_sorting_small_kernel_distributed[grid](
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        moe_buf.numel(),
        mx_quant_input if emit_mx else moe_buf,
        mx_quant_input.stride(0) if emit_mx else 0,
        qout,
        qscale,
        m,
        num_experts,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        P_POW2=triton.next_power_of_2(p),
        P_CHUNK=p_chunk,
        E_CEIL=(num_experts + p_chunk - 1) // p_chunk * p_chunk,
        PAD_POW2=triton.next_power_of_2(p * block_size),
        BUF_BLOCK=buf_block,
        num_buf=num_buf,
        EMIT_MX=emit_mx,
        N_COLS=n_cols,
        QCHUNK=_mx_quant_chunk(n_cols) if emit_mx else 32,
        SCALEN_PAD=scalen_pad,
        num_warps=8,
    )
    if emit_mx:
        return qout, qscale.view(torch.float8_e8m0fnu)
    return None


def _mx_quant_chunk(n_cols: int) -> int | None:
    """Largest power-of-two column chunk (32..2048) that tiles n_cols exactly.

    The quant loop needs N_COLS % QCHUNK == 0 and a power-of-two tl.arange; K3's
    3584-wide latent tiles with 512 (7 chunks), which the old min(2048, n_cols)
    choice could not express."""
    for chunk in (2048, 1024, 512, 256, 128, 64, 32):
        if chunk <= n_cols and n_cols % chunk == 0:
            return chunk
    return None


@functools.cache
def _mid_max_tokens() -> int:
    from sglang.kernels.ops.moe import k3_route_sort_mid

    return k3_route_sort_mid.max_tokens()


@dataclass
class K3SortPlan:
    """What the K3 route/sort/quant kernels need from the model for one aiter
    ``fused_moe`` call. aiter runs the sort internally and forwards none of it,
    so the model enters the plan with ``k3_sort_plan`` around its experts call;
    an active plan also opts that call into the K3 kernels.

    route_state: launch-A products of k3_route_sort_mid (top-k, counts, quant).
    shared_gemm: (topk_ids, act, w_down, out) of the shared down GEMM to run in
        the small-batch sort launch (k3_moe_epi.shared_gemm_sort_quant).
    Both belong to the top-k tensor they were computed for, and the sort takes
    them only for that tensor; ``finish`` resolves whatever was not taken.
    """

    route_state: RouteState | None = None
    shared_gemm: tuple | None = None

    def take_route_state(self, topk_ids, qx) -> RouteState | None:
        st = self.route_state
        if (
            st is None
            or st.consumed
            or topk_ids.data_ptr() != st.topk_ids.data_ptr()
            or topk_ids.shape != st.topk_ids.shape
            or qx is None
            or qx.data_ptr() != st.qx.data_ptr()
            or qx.shape != st.qx.shape
            or qx.stride() != st.qx.stride()
        ):
            return None
        return st

    def take_shared_gemm(self, topk_ids):
        """(act, w_down, out) if the GEMM was planned for exactly this top-k."""
        p = self.shared_gemm
        if (
            p is not None
            and p[0].data_ptr() == topk_ids.data_ptr()
            and p[0].shape == topk_ids.shape
        ):
            self.shared_gemm = None
            return p[1:]
        return None

    def finish(self) -> None:
        st = self.route_state
        if st is not None and not st.consumed:
            # launch A ran but its sort launch B never did
            st.counts.zero_()
        if self.shared_gemm is not None:
            from sglang.kernels.ops.moe import k3_moe_epi

            _, act, w_down, out = self.shared_gemm
            self.shared_gemm = None
            _warn_shared_gemm_untaken(act.shape[0])
            k3_moe_epi.shared_gemm_preact(act, w_down, out)


@functools.cache
def _warn_shared_gemm_untaken(m: int) -> None:
    logger.warning(
        "The MoE sort did not take the shared GEMM (M=%d); running it "
        "standalone (correct, no launch saved)",
        m,
    )


_k3_sort_plan: ContextVar[K3SortPlan | None] = ContextVar(
    "aiter_k3_sort_plan", default=None
)


@contextlib.contextmanager
def k3_sort_plan(plan: K3SortPlan | None = None):
    """Run the enclosed aiter fused_moe calls with the K3 sort kernels and
    ``plan``'s inputs; resolves the untaken parts of the plan on exit."""
    plan = K3SortPlan() if plan is None else plan
    token = _k3_sort_plan.set(plan)
    try:
        yield plan
    finally:
        _k3_sort_plan.reset(token)
        plan.finish()


# aiter calls _moe_sorting_impl from inside fused_moe without passing hidden_states through
_pending_quant_input: ContextVar[torch.Tensor | None] = ContextVar(
    "aiter_pending_quant_input", default=None
)
# the sort-time quant, handed on to the patched fused_dynamic_mxfp8_quant_moe_sort
_emitted_quant: ContextVar[tuple[torch.Tensor, torch.Tensor] | None] = ContextVar(
    "aiter_emitted_quant", default=None
)


@functools.cache
def apply_aiter_small_moe_sort_patch() -> None:
    """Patch a stock aiter so decode-sized MoE sorting (+ stage1 mxfp8 quant) is one launch."""
    if not is_gfx95_supported():
        return

    try:
        import aiter.fused_moe as fm
        from aiter import dtypes

        orig_fused_moe = fm.fused_moe
        orig_sorting_impl = fm._moe_sorting_impl
        orig_mx_quant = fm.fused_dynamic_mxfp8_quant_moe_sort
    except (ImportError, AttributeError) as exc:
        logger.info("aiter small-batch MoE sorting patch not applied: %s", exc)
        return

    @functools.wraps(orig_fused_moe)
    def fused_moe_wrapper(
        hidden_states, w1, w2, topk_weight, topk_ids, *args, **kwargs
    ):
        quant_type = kwargs.get("quant_type", fm.QuantType.No)
        k3 = _k3_sort_plan.get() is not None
        emit = (
            quant_type == fm.QuantType.per_1x32
            and w1.dtype in (dtypes.fp4x2, dtypes.fp8)
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
            and (
                hidden_states.dim() == 2
                and hidden_states.stride(-1) == 1
                and _mx_quant_chunk(hidden_states.shape[-1]) is not None
                and (topk_ids.numel() <= 256 or topk_ids.shape[0] <= _mid_max_tokens())
                if k3
                else hidden_states.is_contiguous()
                and hidden_states.shape[-1] % 2048 == 0
                and topk_ids.numel() <= 256
            )
        )
        if k3 and not emit and not hidden_states.is_contiguous():
            # only the fused sort + quant reads strided rows
            hidden_states = hidden_states.contiguous()
        input_token = _pending_quant_input.set(hidden_states if emit else None)
        emitted_token = _emitted_quant.set(None)
        try:
            return orig_fused_moe(
                hidden_states, w1, w2, topk_weight, topk_ids, *args, **kwargs
            )
        finally:
            _emitted_quant.reset(emitted_token)
            _pending_quant_input.reset(input_token)

    @functools.wraps(orig_sorting_impl)
    def sorting_impl_wrapper(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size,
        expert_mask,
        num_local_tokens,
        dispatch_policy,
        use_opus,
        return_local_topk_ids=False,
        accumulate=True,
        output_aux=False,
        **orig_kwargs,
    ):
        # a caller-owned moe_buf (output=) is zeroed by the sort kernel like ours
        k3 = _k3_sort_plan.get() is not None
        output = orig_kwargs.get("output")
        own_output = output is None or (k3 and accumulate)
        if (
            k3
            and not output_aux
            and not return_local_topk_ids
            and own_output
            and _mid_sort_supported(
                topk_ids,
                int(block_size),
                expert_mask,
                num_local_tokens,
                int(num_experts),
                _pending_quant_input.get(),
            )
        ):
            M = topk_ids.shape[0]
            if output is not None:
                moe_buf = output
            elif accumulate:
                moe_buf = torch.empty(
                    (M, model_dim), dtype=moebuf_dtype, device=topk_ids.device
                )
            else:
                moe_buf = torch.empty(
                    (0, 0), dtype=moebuf_dtype, device=topk_ids.device
                )
            qx = _pending_quant_input.get()
            if moe_buf.numel() == 0 or (
                moe_buf.is_contiguous() and moe_buf.numel() == M * qx.shape[-1]
            ):
                ret, quant_ret = _run_mid_sort(
                    topk_ids,
                    topk_weights,
                    int(num_experts),
                    int(block_size),
                    moe_buf,
                    _pending_quant_input.get(),
                )
                _emitted_quant.set(quant_ret)
                return ret
        if (
            not output_aux
            and not return_local_topk_ids
            and own_output
            and _small_sort_supported(
                topk_ids,
                int(block_size),
                expert_mask,
                num_local_tokens,
                int(num_experts),
                emit_mx=_pending_quant_input.get() is not None,
            )
        ):
            device = topk_ids.device
            M, topk = topk_ids.shape
            max_num_tokens_padded = int(
                topk_ids.numel() + num_experts * block_size - topk
            )
            max_num_m_blocks = int(
                (max_num_tokens_padded + block_size - 1) // block_size
            )
            sorted_ids = torch.empty(
                max_num_tokens_padded, dtype=dtypes.i32, device=device
            )
            sorted_weights = torch.empty(
                max_num_tokens_padded, dtype=dtypes.fp32, device=device
            )
            sorted_expert_ids = torch.empty(
                max_num_m_blocks, dtype=dtypes.i32, device=device
            )
            num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
            if output is not None:
                moe_buf = output
            elif accumulate:
                moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
            else:
                moe_buf = torch.empty((0, 0), dtype=moebuf_dtype, device=device)
            quant_ret = _run_small_sort(
                topk_ids,
                topk_weights,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                moe_buf,
                int(block_size),
                _pending_quant_input.get(),
                int(num_experts),
            )
            if quant_ret is not None:
                _emitted_quant.set(quant_ret)
            return (
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                moe_buf,
            )
        return orig_sorting_impl(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            moebuf_dtype,
            block_size,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
            use_opus,
            return_local_topk_ids=return_local_topk_ids,
            accumulate=accumulate,
            output_aux=output_aux,
            **orig_kwargs,
        )

    @functools.wraps(orig_mx_quant)
    def mx_quant_wrapper(input, sorted_ids, *args, **kwargs):
        pre = _emitted_quant.get()
        if pre is not None and pre[0].shape == input.shape:
            _emitted_quant.set(None)
            return pre
        if (
            _k3_sort_plan.get() is not None
            and input.dim() == 2
            and not input.is_contiguous()
        ):
            input = input.contiguous()
        return orig_mx_quant(input, sorted_ids, *args, **kwargs)

    fm.fused_moe = fused_moe_wrapper
    fm._moe_sorting_impl = sorting_impl_wrapper
    fm.fused_dynamic_mxfp8_quant_moe_sort = mx_quant_wrapper
    logger.info("aiter small-batch MoE sorting patch applied")
