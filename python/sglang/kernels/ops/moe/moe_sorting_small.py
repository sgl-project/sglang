# SPDX-License-Identifier: Apache-2.0
"""Single-launch MoE sorting (+ optional stage1 mxfp8 activation quant) for
decode-sized batches on the aiter MoE runner, applied to a stock aiter install
via runtime patches.

aiter's opus sorting kernel is built for thousands of tokens; at M<=8 it costs
~7us/layer of launch/ramp overhead per launch, and the separate
fused_dynamic_mxfp8_quant_moe_sort stage1 activation quant costs another ~4us.
With P = M*topk pairs (<=64) the whole sorting job — stable sort-by-expert,
per-expert block padding, expert-id table, num_valid, moe_buf zero-fill — fits
one small Triton program using P x P rank compares, and the quant rides the
same launch on dedicated CTAs that recompute the (cheap) pair math instead of
cross-CTA synchronizing (a serial pid-0 quant tail measured -3.8% e2e).

Output layouts match the aiter kernels bit-for-bit (MiniMax-M3 MI350X: sorting
verified over 160 random M<=8 cases; quant a1 bytes + swizzled e8m0 scale bytes
verified vs the HIP kernel):
  sorted_ids[i]        = (topk_slot << 24) | token   (padding: (topk << 24) | M)
  sorted_weights       = pair weight                 (padding: 0)
  sorted_expert_ids[b] = expert of block b
  num_valid_ids        = [num_blocks * block_size, M]
  quant a1             = per-token fp8 rows; scale byte per (sorted_row, group)
                         at aiter's mx_scale_shuffle_idx address, e8m0 RoundUp
                         (fp32 exponent bits of amax * float32(1/448))

Three patch points on the aiter.fused_moe module namespace, all falling back
to the original functions when the fast path does not apply:
  * ``fused_moe``            — stashes hidden_states for the sort-time quant
  * ``_moe_sorting_impl``    — replaces the opus sort at M*topk <= 64
  * ``fused_dynamic_mxfp8_quant_moe_sort`` — consumes the pre-emitted quant
"""

from __future__ import annotations

import functools
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
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
    NUM_BUF: tl.constexpr,  # buf-zero CTAs occupy pids [1, NUM_BUF]
    EMIT_MX: tl.constexpr,  # also emit the mxfp8 quant of qx (group_size 32)
    N_COLS: tl.constexpr,
    QCHUNK: tl.constexpr,  # columns per quant iteration (multiple of 32)
    SCALEN_PAD: tl.constexpr,  # ceil(N_COLS/32 / 8) * 8
):
    pid = tl.program_id(0)
    if pid > 0 and pid <= NUM_BUF:
        # Zero-fill the MoE accumulation buffer.
        offs = (pid - 1) * BUF_BLOCK + tl.arange(0, BUF_BLOCK)
        tl.store(
            moe_buf_ptr + offs,
            tl.zeros((BUF_BLOCK,), moe_buf_ptr.dtype.element_ty),
            mask=offs < moe_buf_numel,
        )
        return

    P = M * TOPK
    offs_p = tl.arange(0, P_POW2)
    # Sort math is a few P x P vector ops — cheap enough that the quant CTAs
    # recompute it independently instead of waiting on pid 0.
    mask_p = offs_p < P
    # Sentinel expert (larger than any real id) keeps inactive lanes out of
    # every "smaller expert" count below.
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

    # Blocks owned by experts with a smaller id (leaders carry their expert's
    # block count).
    smaller = e[None, :] < e[:, None]
    blocks_before = tl.sum(
        tl.where(smaller & is_leader[None, :], blocks_of_e[None, :], 0), axis=1
    )
    dest = blocks_before * BLOCK_SIZE + rank

    if EMIT_MX and pid > NUM_BUF:
        # Quant CTA: one (pair, column-chunk) slice each. Mirrors
        # fused_dynamic_mxfp8_quant_moe_sort (group_size=32, e8m0 RoundUp
        # scale via fp32 bit manipulation, per-token fp8 rows, scale byte
        # per (sorted_row, group) at the mx_scale_shuffle_idx address).
        q_id = pid - NUM_BUF - 1
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

    # pid 0: sort outputs.
    # Pass 1: padding over the whole used region; pass 2: scatter real pairs.
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

    # Expert id per used block: leaders write their expert into each of their
    # blocks. With P <= 2 * BLOCK_SIZE an expert owns at most 2 blocks.
    for j in tl.static_range(2):
        bm = is_leader & (j < blocks_of_e)
        tl.store(sorted_expert_ids_ptr + blocks_before + j, e, mask=bm)

    tl.store(
        num_valid_ids_ptr + tl.arange(0, 2),
        tl.where(tl.arange(0, 2) == 0, num_valid, M),
    )


def _small_sort_supported(topk_ids, block_size, expert_mask, num_local_tokens):
    m, topk = topk_ids.shape
    return (
        expert_mask is None
        and num_local_tokens is None
        and m * topk <= 64
        and m * topk <= 2 * block_size
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
        NUM_BUF=num_buf,
        EMIT_MX=emit_mx,
        N_COLS=n_cols,
        QCHUNK=min(2048, n_cols),
        SCALEN_PAD=scalen_pad,
        num_warps=4,
    )
    if emit_mx:
        return qout, qscale.view(torch.float8_e8m0fnu)
    return None


# hidden_states of the in-flight aiter fused_moe call, when its dtypes make
# the stage1 mxfp8 quant path certain (single-threaded forward; cleared in
# the fused_moe wrapper's finally).
_pending_quant_input: torch.Tensor | None = None

_patched = False


def apply_aiter_small_moe_sort_patch() -> None:
    """Patch a stock aiter so decode-sized MoE sorting (+ stage1 mxfp8 quant)
    runs as one sglang Triton launch. Idempotent."""
    global _patched
    if _patched:
        return

    import aiter.fused_moe as fm
    from aiter import dtypes
    from aiter.jit.utils.chip_info import get_gfx  # noqa: F401  (import check)

    orig_fused_moe = fm.fused_moe
    orig_sorting_impl = fm._moe_sorting_impl
    orig_mx_quant = fm.fused_dynamic_mxfp8_quant_moe_sort

    @functools.wraps(orig_fused_moe)
    def fused_moe_wrapper(
        hidden_states, w1, w2, topk_weight, topk_ids, *args, **kwargs
    ):
        global _pending_quant_input
        quant_type = kwargs.get("quant_type", fm.QuantType.No)
        emit = (
            quant_type == fm.QuantType.per_1x32
            and w1.dtype in (dtypes.fp4x2, dtypes.fp8)
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
            and hidden_states.is_contiguous()
            and hidden_states.shape[-1] % 2048 == 0
            and topk_ids.numel() <= 64
        )
        _pending_quant_input = hidden_states if emit else None
        try:
            return orig_fused_moe(
                hidden_states, w1, w2, topk_weight, topk_ids, *args, **kwargs
            )
        finally:
            _pending_quant_input = None

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
    ):
        if (
            not output_aux
            and not return_local_topk_ids
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
            if (expert_mask is not None) or accumulate:
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
                _pending_quant_input,
            )
            if quant_ret is not None:
                sorted_ids._premx_quant = quant_ret
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
        )

    @functools.wraps(orig_mx_quant)
    def mx_quant_wrapper(input, sorted_ids, *args, **kwargs):
        pre = getattr(sorted_ids, "_premx_quant", None)
        if pre is not None and pre[0].shape == input.shape:
            del sorted_ids._premx_quant
            return pre
        return orig_mx_quant(input, sorted_ids, *args, **kwargs)

    fm.fused_moe = fused_moe_wrapper
    fm._moe_sorting_impl = sorting_impl_wrapper
    fm.fused_dynamic_mxfp8_quant_moe_sort = mx_quant_wrapper
    _patched = True
    logger.info("aiter small-batch MoE sorting patch applied")
