# Copyright 2025 XunhaoLai. All rights reserved.

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import (
    SPARSE_KV_FP8_DTYPES,
    check_sparse_kv_fp8,
    get_cu_seqblocks,
    robust_allocator,
)

# Prefill M-bucket split for per-length autotune pinning (SILOTIGER-762). The
# observed prefill passes cluster into a small partial-chunk tail and a large
# ~16k chunk; the threshold sits in the empty gap between them so the autotuner
# caches/pins one winning occupancy config per prefill-length bucket instead of
# reusing one winner across all M.
_PREFILL_M_BUCKET_THRESHOLD = 2048


@triton.heuristics(
    {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "BLOCK_SIZE_VD": lambda args: triton.next_power_of_2(args["v_head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(
            max(
                16 // args["BLOCK_SIZE_Q"],
                triton.next_power_of_2(args["gqa_group_size"]),
            )
        ),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"] * args["BLOCK_SIZE_H"],
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.autotune(
    # num_stages is capped at 3. With the bq8 prefill tile, num_stages=4 drives the
    # ROCm `tritonamdgpu-block-pingpong` pass to crash the compiler HARD on gfx950
    # (RuntimeError: PassManager::run failed) instead of being pruned -- that
    # exception is not caught by the autotuner and takes the whole scheduler down.
    # ns 2/3 are verified to compile and cover the useful pipeline depth for this
    # gather-bound sparse kernel; deeper stages don't help here (SILOTIGER-762).
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in (2, 4, 8)
        for ns in (2, 3)
    ],
    key=[
        "BLOCK_SIZE_Q",
        "BLOCK_SIZE_K",
        "qk_head_dim",
        "v_head_dim",
        "gqa_group_size",
        # M-bucket (0=small tail, 1=large ~16k chunk): the best occupancy config
        # differs by prefill-pass length, so cache/pin the autotune winner per
        # bucket instead of reusing one winner across all M (SILOTIGER-762).
        "m_bucket",
    ],
)
@triton.jit
def _gqa_share_sparse_fwd_kernel(
    q_ptr,  # Q: n x h x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    sink_ptr,  # Sink: h x d
    t_ptr,  # topk_idx: kh x n x k
    o_ptr,  # O: n x h x d
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    # seqlens
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_kv_heads,
    gqa_group_size,
    qk_head_dim,
    v_head_dim,
    max_topk,
    # q loop num
    num_q_loop,
    # M-bucket id: unused in the kernel body, present only so @triton.autotune
    # keys on it and caches the best config per prefill-length bucket.
    m_bucket,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_sh,
    stride_sd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_r2t_b,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    # has sink
    HAS_SINK: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_PV: tl.constexpr,
    FP8_QK: tl.constexpr,
    Q_IS_FP8: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # bf16/fp16 compute dtype for internal casts (e.g. widening the fp8 K/V
    # cache). Taken from the OUTPUT element type rather than q.dtype so the "Q
    # arrives as fp8" fast path (Q_IS_FP8) still widens K/V to bf16, not fp8.
    compute_dtype = o_ptr.dtype.element_ty
    # get batch id and head id
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (
        tl.load(slot_ids + pid_b).to(tl.int64) + max_slots
    ) % max_slots  # safety against negative
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    if HAS_SINK:
        sink_ptrs = tl.make_block_ptr(
            base=sink_ptr + pid_h * stride_sh,
            shape=(gqa_group_size, qk_head_dim),
            strides=(stride_sh, stride_sd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_KD),
            order=(1, 0),
        )
        sink = tl.load(sink_ptrs, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
    # offsets for paged K/V load
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_kd = tl.arange(0, BLOCK_SIZE_KD)
    off_vd = tl.arange(0, BLOCK_SIZE_VD)
    kd_mask = off_kd < qk_head_dim
    vd_mask = off_vd < v_head_dim
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        # init topk idx pointer
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        # we assume that the topk_idx is right padded with -1
        off_t = tl.arange(0, BLOCK_SIZE_T)
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < max_topk, other=-1)
        valid_idx = tl.where(topk_idx >= 0, off_t, -1)
        real_topk = tl.sum(valid_idx != -1, axis=0)
        # init qkv pointer
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, qk_head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_KD),
            order=(2, 1, 0),
        )
        # load q, shape: [BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D] -> [BLOCK_SIZE_QH, BLOCK_SIZE_D]
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        # init statistics
        off_q_k = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        if HAS_SINK:
            m_i = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H), dtype=tl.float32)
            lse_i = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H), dtype=tl.float32)
            qsink = (
                tl.sum(q.to(tl.float32) * sink[None, :, :], axis=2) * sm_scale_log2e
            )  # (BLOCK_SIZE_Q, BLOCK_SIZE_H)
            m_i += qsink
            lse_i += qsink
            m_i = tl.reshape(m_i, BLOCK_SIZE_QH)
            lse_i = tl.reshape(lse_i, BLOCK_SIZE_QH)
        else:
            m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
            lse_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        acc_o = tl.full((BLOCK_SIZE_QH, BLOCK_SIZE_VD), 0, dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_KD)
        # Select the Q operand + score scale for the Q@K dot (SILOTIGER-762):
        #   Q_IS_FP8 -- Q already fp8 (unit-scaled like the fp8 K cache): consume
        #     it natively, no in-kernel rounding (the lossless "no upcast" path).
        #   FP8_QK   -- Q arrives bf16: per-tensor quantize it so amax(|Q|) maps
        #     to the e4m3 finite max (448), then fold the inverse scale into
        #     qk_scale. Overflow-safe (max->448, no NaN); the amax+cast run once
        #     per q-block (not per K tile), so the cost is negligible.
        #   else     -- keep Q at the bf16 compute dtype (K is widened to match).
        # q_qk is kept separate from `q` so the sink dot still uses bf16.
        qk_scale = sm_scale_log2e
        if Q_IS_FP8:
            if IS_FP8:
                q_qk = q  # fp8 Q x fp8 K -> native fp8 MFMA, lossless in-kernel
            else:
                q_qk = q.to(compute_dtype)  # fp8 Q, bf16 K -> widen Q to match
        elif FP8_QK and IS_FP8:
            q_scale = 448.0 / tl.maximum(tl.max(tl.abs(q)), 1e-9)
            q_qk = (q * q_scale).to(tl.float8e4nv)
            qk_scale = sm_scale_log2e / q_scale
        else:
            q_qk = q
        # sparse attention
        for i in range(real_topk):
            # get current block start index (absolute K position)
            c = tl.load(t_ptr_j).to(tl.int32) * BLOCK_SIZE_K
            t_ptr_j = t_ptr_j + stride_tk
            # paged load K via req_to_token: pos -> slot -> k_cache
            pos = c + off_n
            pos_mask = pos < seq_len
            slots = tl.load(
                req_to_token_ptr + sid * stride_r2t_b + pos,
                mask=pos_mask,
                other=0,
            ).to(tl.int64)
            slots = (slots + max_slots) % max_slots  # safety against negative
            # k shape: [BLOCK_SIZE_KD, BLOCK_SIZE_K] (transposed for tl.dot)
            k = tl.load(
                k_cache_ptr
                + slots[None, :] * stride_ks
                + pid_kh * stride_kh
                + off_kd[:, None] * stride_kd,
                mask=kd_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if IS_FP8:
                if Q_IS_FP8:
                    # Q already fp8 -> native fp8 Q@K, keep K fp8 (no widen).
                    pass
                elif FP8_QK:
                    # in-kernel fp8 Q@K -> keep K fp8 (drops the widen cvt).
                    pass
                else:
                    # fp8 main K cache is unit-scaled; widen to the compute dtype
                    # before the tl.dot (compiled out when the cache is bf16).
                    k = k.to(compute_dtype)
            # compute qk
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            # causal mask
            qk += tl.where(off_q_k[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            # [BLOCK_SIZE_QH, qk_head_dim] @ [qk_head_dim, BLOCK_SIZE_K]
            #   -> [BLOCK_SIZE_QH, BLOCK_SIZE_K]. qk_scale folds the FP8_QK
            # Q-scale inverse into sm_scale (== sm_scale_log2e when FP8_QK is off).
            qk += tl.dot(q_qk, k) * qk_scale
            # K boundary mask: positions beyond seq_len contribute -inf
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            # paged load V
            v = tl.load(
                v_cache_ptr
                + slots[:, None] * stride_vs
                + pid_kh * stride_vh
                + off_vd[None, :] * stride_vd,
                mask=pos_mask[:, None] & vd_mask[None, :],
                other=0.0,
            )
            if IS_FP8:
                if FP8_PV:
                    # keep V in fp8 for a fp8 P@V MFMA (drops the widen cvt).
                    pass
                else:
                    # Widen V so `p.to(v.dtype)` casts P to the compute dtype
                    # rather than fp8 (which would wreck attn-weight precision).
                    v = v.to(compute_dtype)
            # online softmax: running max + per-tile acc_o rescale.
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            acc_o_scale = tl.exp2(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]
            if FP8_PV and IS_FP8:
                # cast P to fp8 for the P@V MFMA (V kept fp8 above). P in [0, 1]
                # here, so the clamp only guards a stray rounding overflow.
                p_cast = tl.minimum(p, 448.0).to(v.dtype)
            else:
                p_cast = p.to(v.dtype)
            acc_o += tl.dot(p_cast, v)
            # update statistics
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # final scale
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        # save output
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_VD)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, v_head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_VD),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


@torch.no_grad()
def flash_prefill_with_gqa_share_sparse(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size_q: int,
    block_size_k: int,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    sm_scale: Optional[float] = None,
    use_tma: bool = True,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
) -> torch.Tensor:
    triton.set_allocator(robust_allocator)
    is_fp8 = check_sparse_kv_fp8(q, k_cache, v_cache, label="prefill")
    # SILOTIGER-762: Q normally arrives bf16/fp16 (enforced by check_sparse_kv_fp8),
    # so this is False today and the native-fp8-Q "no rounding" path stays dormant
    # (Q_IS_FP8 is compiled out). Detect it anyway so a future pre-quantized-Q
    # caller lights up the lossless path without another kernel change.
    q_is_fp8 = q.dtype in SPARSE_KV_FP8_DTYPES
    assert block_size_q in {1, 2, 4, 8, 16, 32, 64}
    assert block_size_k in {16, 32, 64, 128}
    # shape
    total_q, num_q_heads, qk_head_dim = q.shape
    max_slots, num_k_heads, _ = k_cache.shape
    _, num_v_heads, v_head_dim = v_cache.shape
    batch_size = cu_seqlens.shape[0] - 1
    topk = topk_idx.shape[-1]
    assert topk_idx.shape[0] == num_k_heads
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    gqa_group_size = num_q_heads // num_k_heads
    assert gqa_group_size * block_size_q <= 128
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    if cu_seqblocks_q is None or max_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, _, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k
        )
    # output tensor. Output stays a real compute dtype even when Q is fp8-storage.
    out_dtype = (
        q.dtype if q.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    )
    o = torch.empty(
        total_q, num_q_heads, v_head_dim, device=q.device, dtype=out_dtype
    )
    # launch kernel
    num_q_loop = (
        max_seqblock_q // 131072 + 1
    )  # calculate multiple queries in one kernel if seqlence length is too long
    BLOCK_SIZE_Q = triton.next_power_of_2(block_size_q)
    BLOCK_SIZE_K = triton.next_power_of_2(block_size_k)
    # M-bucket for per-length autotune pinning (SILOTIGER-762): the best occupancy
    # config differs between the small partial-chunk tail and the large ~16k
    # prefill chunk. Two buckets; the threshold sits in the gap between them.
    m_bucket = 0 if max_seqlen_q <= _PREFILL_M_BUCKET_THRESHOLD else 1
    # SILOTIGER-762 fp8 attention datapath. Both flags consume the already-fp8 KV
    # cache natively instead of upcasting it to bf16, and default ON; disable
    # either with SGLANG_MINIMAX_SPARSE_FP8_PV=0 / SGLANG_MINIMAX_SPARSE_FP8_QK=0.
    # Both are no-ops unless the KV cache is fp8.
    fp8_pv = os.environ.get("SGLANG_MINIMAX_SPARSE_FP8_PV", "1") == "1"
    fp8_qk = os.environ.get("SGLANG_MINIMAX_SPARSE_FP8_QK", "1") == "1"
    grid = (
        triton.cdiv(triton.cdiv(max_seqlen_q, block_size_q), num_q_loop),
        num_k_heads,
        batch_size,
    )
    _gqa_share_sparse_fwd_kernel[grid](
        q,
        k_cache,
        v_cache,
        sink,
        topk_idx,
        o,
        req_to_token,
        cu_seqlens,
        cu_seqblocks_q,
        seq_lens,
        prefix_lens,
        slot_ids,
        max_slots,
        num_k_heads,
        gqa_group_size,
        qk_head_dim,
        v_head_dim,
        topk,
        num_q_loop,
        m_bucket,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        req_to_token.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_TMA=use_tma,
        IS_FP8=is_fp8,
        FP8_PV=fp8_pv,
        FP8_QK=fp8_qk,
        Q_IS_FP8=q_is_fp8,
    )
    return o
