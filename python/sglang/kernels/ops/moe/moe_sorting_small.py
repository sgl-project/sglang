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

import functools
import logging
from contextvars import ContextVar

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_gfx95_supported

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
            base_sw = (
                (dest_p // 32) * (SCALEN_PAD * 32)
                + (dest_p % 16) * 4
                + (dest_p % 32) // 16
            )
            x = tl.load(qx_ptr + token_p * N_COLS + c0 + offs_q).to(tl.float32)
            x2 = tl.reshape(x, (QCHUNK // 32, 32))
            amax = tl.maximum(tl.max(tl.abs(x2), axis=1), 1e-10)
            sf = amax * (1.0 / 448.0)
            bits = sf.to(tl.int32, bitcast=True)
            exp = (bits >> 23) & 0xFF
            exp = tl.where((bits & 0x7FFFFF) != 0, exp + 1, exp)
            scale = (exp << 23).to(tl.float32, bitcast=True)
            if p % TOPK == 0:
                # one fp8 out row per token (pairs are token-major)
                q = tl.clamp(x2 / scale[:, None], -448.0, 448.0)
                tl.store(
                    qout_ptr + token_p * N_COLS + c0 + offs_q,
                    tl.reshape(q, (QCHUNK,)).to(qout_ptr.dtype.element_ty),
                )
            y = c0 // 32 + offs_g
            sw = base_sw + (y // 8) * 256 + (y % 4) * 64 + ((y % 8) // 4) * 2
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
        base_sw = (
            (dest_p // 32) * (SCALEN_PAD * 32) + (dest_p % 16) * 4 + (dest_p % 32) // 16
        )
        for cc in tl.static_range(N_COLS // QCHUNK):
            c0 = cc * QCHUNK
            x = tl.load(qx_ptr + token_p * N_COLS + c0 + offs_q).to(tl.float32)
            x2 = tl.reshape(x, (QCHUNK // 32, 32))
            amax = tl.maximum(tl.max(tl.abs(x2), axis=1), 1e-10)
            sf = amax * (1.0 / 448.0)
            bits = sf.to(tl.int32, bitcast=True)
            exp = (bits >> 23) & 0xFF
            exp = tl.where((bits & 0x7FFFFF) != 0, exp + 1, exp)
            scale = (exp << 23).to(tl.float32, bitcast=True)
            if p % TOPK == 0:
                # one fp8 out row per token (pairs are token-major)
                q = tl.clamp(x2 / scale[:, None], -448.0, 448.0)
                tl.store(
                    qout_ptr + token_p * N_COLS + c0 + offs_q,
                    tl.reshape(q, (QCHUNK,)).to(qout_ptr.dtype.element_ty),
                )
            y = c0 // 32 + offs_g
            sw = base_sw + (y // 8) * 256 + (y % 4) * 64 + ((y % 8) // 4) * 2
            tl.store(qscale_ptr + sw, exp.to(tl.uint8))


def _small_sort_supported(topk_ids, block_size, expert_mask, num_local_tokens):
    m, topk = topk_ids.shape
    return (
        expert_mask is None
        and num_local_tokens is None
        and m * topk <= 256
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
    num_buf = triton.cdiv(max(moe_buf.numel(), 1), buf_block)
    if p <= 64 and p <= 2 * block_size:
        # compact variant: one sort CTA does the P x P rank compare
        num_quant = (p * (n_cols // min(2048, n_cols))) if emit_mx else 0
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
            QCHUNK=min(2048, n_cols),
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
        QCHUNK=min(2048, n_cols),
        SCALEN_PAD=scalen_pad,
        num_warps=8,
    )
    if emit_mx:
        return qout, qscale.view(torch.float8_e8m0fnu)
    return None


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
        emit = (
            quant_type == fm.QuantType.per_1x32
            and w1.dtype in (dtypes.fp4x2, dtypes.fp8)
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
            and hidden_states.is_contiguous()
            and hidden_states.shape[-1] % 2048 == 0
            and topk_ids.numel() <= 256
        )
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
        # newer aiter passes a caller-owned moe_buf as output=; leave that to aiter
        if (
            not output_aux
            and not return_local_topk_ids
            and orig_kwargs.get("output") is None
            and _small_sort_supported(
                topk_ids, int(block_size), expert_mask, num_local_tokens
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
            if accumulate:
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
        return orig_mx_quant(input, sorted_ids, *args, **kwargs)

    fm.fused_moe = fused_moe_wrapper
    fm._moe_sorting_impl = sorting_impl_wrapper
    fm.fused_dynamic_mxfp8_quant_moe_sort = mx_quant_wrapper
    logger.info("aiter small-batch MoE sorting patch applied")
