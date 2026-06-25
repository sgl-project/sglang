# Copyright 2025 XunhaoLai. All rights reserved.

from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import (
    check_sparse_kv_fp8,
    get_cu_seqblocks,
    set_triton_allocator_if_available,
)


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
    # Configs that fail to compile on the target arch are skipped, so widening
    # the num_warps x num_stages grid only adds candidates, never a bad kernel.
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in (2, 4, 8)
        for ns in (2, 3, 4)
    ],
    key=[
        "BLOCK_SIZE_Q",
        "BLOCK_SIZE_K",
        "qk_head_dim",
        "v_head_dim",
        "gqa_group_size",
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
):
    sm_scale_log2e = sm_scale * 1.4426950409
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
                # fp8 main K cache is unit-scaled; widen to the Q compute dtype
                # before the tl.dot (compiled out when the cache is bf16).
                k = k.to(q.dtype)
            # compute qk
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            # causal mask
            qk += tl.where(off_q_k[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            # [BLOCK_SIZE_QH, qk_head_dim] @ [qk_head_dim, BLOCK_SIZE_K]
            #   -> [BLOCK_SIZE_QH, BLOCK_SIZE_K]
            qk += tl.dot(q, k) * sm_scale_log2e
            # K boundary mask: positions beyond seq_len contribute -inf
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            # compute m_ij and l_ij
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            # scale acc_o
            acc_o_scale = tl.exp2(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]
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
                # Widen V so `p.to(v.dtype)` casts P to the compute dtype rather
                # than to fp8 (which would wreck attention-weight precision).
                v = v.to(q.dtype)
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)
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
    set_triton_allocator_if_available()
    is_fp8 = check_sparse_kv_fp8(q, k_cache, v_cache, label="prefill")
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
    # output tensor
    o = torch.empty(total_q, num_q_heads, v_head_dim, device=q.device, dtype=q.dtype)
    # launch kernel
    num_q_loop = (
        max_seqblock_q // 131072 + 1
    )  # calculate multiple queries in one kernel if seqlence length is too long
    BLOCK_SIZE_Q = triton.next_power_of_2(block_size_q)
    BLOCK_SIZE_K = triton.next_power_of_2(block_size_k)
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
    )
    return o
