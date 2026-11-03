"""Bias-aware block-sparse GQA attention for DASH (Triton).

Adds a per-(query, slot) additive ``log(prior)/sigma`` bias to REMOTE blocks only;
local-window blocks stay unbiased, matching training. Scores live in the log2
domain (``exp2`` softmax), so the natural-log bias is scaled by log2e first.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.unlimited_ocr_dash.common import (
    _FP8_DTYPES,
    SPARSE_BLOCK_SIZE,
    _arch_supports_pdl,
    _merge_topk_attn_out_kernel,
)

# Kernels inline log2(e) as 1.4426950409: @triton.jit bodies read only
# constexpr module globals.

# Split-K target program count for the decode attend.
TARGET_GRID = 256

# Cache keyed by (device, dtype, graph_owned); grows monotonically to max C/Q
# geometry so captured graphs never see a reallocated address.
_DEC_PARTIALS: dict = {}


def _max_topk_chunks(chunk_width: int, num_kv_heads: int) -> int:
    """Largest ``num_topk_chunks`` any batch size can ask for (total_q == 1)."""
    target = max(1, min(int(chunk_width), TARGET_GRID // max(1, num_kv_heads)))
    return 1 << (target.bit_length() - 1)


def _decode_partials(
    chunks: int,
    total_q: int,
    num_heads: int,
    head_dim: int,
    dtype,
    device,
    alloc_chunks: int = 0,
    alloc_rows: int = 0,
):
    # Graph-owned storage must never be replaced by an oversized eager fallback.
    graph_owned = alloc_rows > 0
    key = (str(device), dtype, graph_owned)
    cur = _DEC_PARTIALS.get(key)
    need_c = max(chunks, alloc_chunks)
    need_q = max(total_q, alloc_rows)
    if (
        cur is None
        or cur[0].shape[0] < need_c
        or cur[0].shape[1] < need_q
        or cur[0].shape[2] < num_heads
        or cur[0].shape[3] < head_dim
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "[Unlimited-OCR v2] decode split-K partials would be (re)allocated during "
                "cuda graph capture. They must be pre-allocated at the maximum "
                "geometry on an eager forward by passing cg_rows= to "
                "dash_sparse_attn_decode."
            )
        c = max(need_c, 0 if cur is None else cur[0].shape[0])
        q_ = max(need_q, 0 if cur is None else cur[0].shape[1])
        h = max(num_heads, 0 if cur is None else cur[0].shape[2])
        dd = max(head_dim, 0 if cur is None else cur[0].shape[3])
        cur = (
            torch.empty(c, q_, h, dd, dtype=dtype, device=device),
            torch.empty(c, q_, h, dtype=torch.float32, device=device),
        )
        _DEC_PARTIALS[key] = cur
    o, lse = cur
    return (
        o[:chunks, :total_q, :num_heads, :head_dim],
        lse[:chunks, :total_q, :num_heads],
    )


# Prefill: one program per (query token, kv head, request); BLOCK_SIZE_Q == 1
# so topk/bias are indexed per query TOKEN.
@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_QH": lambda args: (
            args["BLOCK_SIZE_Q"] * triton.next_power_of_2(args["gqa_group_size"])
        ),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _dash_sparse_fwd_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    k_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_dim]
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    b_ptr,  # blk_bias:  [num_kv_heads, total_q, topk] fp32 (log(prior)/sigma)
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    num_q_loop,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_pos,
    stride_kv_h,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_bh,
    stride_bn,
    stride_bk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        off_q = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        pid_h = pid_kh * gqa_group_size
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        b_ptr_j = b_ptr + (q_block_start + pid_q_j) * stride_bn + pid_kh * stride_bh
        off_t = tl.arange(0, BLOCK_SIZE_T)
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < max_topk, other=-1)
        real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        for _ in range(real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)
            pos = c + off_n
            pos_mask = pos < seq_len
            k = tl.load(
                k_cache_ptr
                + page * stride_kv_blk
                + off_n[None, :] * stride_kv_pos
                + pid_kh * stride_kv_h
                + off_d[:, None] * stride_kv_d,
                mask=d_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                k = k.to(q.dtype)
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            # remote blocks contribute only keys strictly BELOW the local window
            # [q - LW + 1, q]; the window itself is attended in the pass below.
            qk += tl.where(
                off_q[:, None, :] >= c + LOCAL_BLOCKS * BLOCK_SIZE_K,
                0,
                float("-inf"),
            )
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            qk += tl.dot(q, k) * sm_scale_log2e
            # DASH: per-(query, slot) log(prior)/sigma bias, log2 domain. Masked
            # lanes stay -inf (-inf + finite == -inf).
            if HAS_BIAS:
                blk_bias = tl.load(b_ptr_j).to(tl.float32)
                qk += blk_bias * 1.4426950409
            b_ptr_j = b_ptr_j + stride_bk
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
            v = tl.load(
                v_cache_ptr
                + page * stride_kv_blk
                + off_n[:, None] * stride_kv_pos
                + pid_kh * stride_kv_h
                + off_d[None, :] * stride_kv_d,
                mask=pos_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                v = v.to(q.dtype)
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # --- token-level local window pass (NO bias, train-parity) ---
        if LOCAL_BLOCKS > 0:
            q_pos_scalar = pid_q_j * BLOCK_SIZE_Q + prefix_len
            b_q_scalar = q_pos_scalar // BLOCK_SIZE_K
            for local_offset in tl.static_range(LOCAL_BLOCKS + 1):
                local_blk = b_q_scalar - local_offset
                if local_blk >= 0:
                    c_local = local_blk * BLOCK_SIZE_K
                    page_local = tl.load(bt_row + local_blk).to(tl.int64)
                    pos_local = c_local + off_n
                    pos_mask_local = pos_local < seq_len
                    k_local = tl.load(
                        k_cache_ptr
                        + page_local * stride_kv_blk
                        + off_n[None, :] * stride_kv_pos
                        + pid_kh * stride_kv_h
                        + off_d[:, None] * stride_kv_d,
                        mask=d_mask[:, None] & pos_mask_local[None, :],
                        other=0.0,
                    )
                    if USE_FP8:
                        k_local = k_local.to(q.dtype)
                    qk_local = tl.zeros(
                        (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32
                    )
                    qk_local += tl.where(
                        (off_q[:, None, :] >= c_local)
                        & (
                            off_q[:, None, :]
                            <= c_local + LOCAL_BLOCKS * BLOCK_SIZE_K - 1
                        ),
                        0,
                        float("-inf"),
                    )
                    qk_local = tl.reshape(qk_local, BLOCK_SIZE_QH, BLOCK_SIZE_K)
                    qk_local += tl.dot(q, k_local) * sm_scale_log2e
                    qk_local += tl.where(pos_mask_local[None, :], 0, float("-inf"))
                    m_ij = tl.maximum(m_i, tl.max(qk_local, axis=1))
                    p = tl.exp2(qk_local - m_ij[:, None])
                    l_ij = tl.sum(p, axis=1)
                    acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
                    v_local = tl.load(
                        v_cache_ptr
                        + page_local * stride_kv_blk
                        + off_n[:, None] * stride_kv_pos
                        + pid_kh * stride_kv_h
                        + off_d[None, :] * stride_kv_d,
                        mask=pos_mask_local[:, None] & d_mask[None, :],
                        other=0.0,
                    )
                    if USE_FP8:
                        v_local = v_local.to(q.dtype)
                    acc_o += tl.dot(p.to(v_local.dtype), v_local)
                    m_i = m_ij
                    lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


# Decode: split-K over the selected blocks (flash-decoding), merged by
# _merge_topk_attn_out_kernel.
@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _dash_sparse_decode_kernel(
    q_ptr,
    k_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_dim]
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    b_ptr,  # blk_bias:  [num_kv_heads, total_q, topk] fp32
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,
    seq_lens,
    total_q,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_pos,
    stride_kv_h,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_bh,
    stride_bn,
    stride_bk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_FP8: tl.constexpr,
    USE_PDL: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_bc = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    local_win_start = tl.maximum(0, query_pos - LOCAL_BLOCKS * BLOCK_SIZE_K + 1)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    pid_h = pid_kh * gqa_group_size
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    bias_base = b_ptr + pid_kh * stride_bh + pid_b * stride_bn
    topk_idx = tl.load(idx_base + off_t * stride_tk, mask=off_t < max_topk, other=-1)
    real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
    # Split over the slots the router filled rather than the stable buffer width;
    # this balances the work across chunks while retaining full coverage.
    chunk_size_topk = (real_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_topk = tl.minimum(chunk_start_topk + chunk_size_topk, real_topk)

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    cur_bias_ptr = bias_base + chunk_start_topk * stride_bk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = (pos < kv_len) & (pos < local_win_start)
        k = tl.load(
            k_cache_ptr
            + page * stride_kv_blk
            + off_n[None, :] * stride_kv_pos
            + pid_kh * stride_kv_h
            + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            k = k.to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        # DASH: per-(query, slot) log(prior)/sigma bias, log2 domain.
        if HAS_BIAS:
            blk_bias = tl.load(cur_bias_ptr).to(tl.float32)
            qk += blk_bias * 1.4426950409
        cur_bias_ptr = cur_bias_ptr + stride_bk
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = tl.load(
            v_cache_ptr
            + page * stride_kv_blk
            + off_n[:, None] * stride_kv_pos
            + pid_kh * stride_kv_h
            + off_d[None, :] * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            v = v.to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    # --- local window pass (chunk 0 only, NO bias) ---
    if LOCAL_BLOCKS > 0 and pid_c == 0:
        b_q_scalar = query_pos // BLOCK_SIZE_K
        for local_offset in tl.static_range(LOCAL_BLOCKS + 1):
            local_blk = b_q_scalar - local_offset
            if local_blk >= 0:
                c_local = local_blk * BLOCK_SIZE_K
                page_local = tl.load(bt_row + local_blk).to(tl.int64)
                pos_local = c_local + off_n
                pos_mask_local = (
                    (pos_local >= local_win_start)
                    & (pos_local <= query_pos)
                    & (pos_local < kv_len)
                )
                k_local = tl.load(
                    k_cache_ptr
                    + page_local * stride_kv_blk
                    + off_n[None, :] * stride_kv_pos
                    + pid_kh * stride_kv_h
                    + off_d[:, None] * stride_kv_d,
                    mask=d_mask[:, None] & pos_mask_local[None, :],
                    other=0.0,
                )
                if USE_FP8:
                    k_local = k_local.to(q.dtype)
                qk_local = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
                qk_local += tl.where(pos_mask_local[None, :], 0, float("-inf"))
                qk_local += tl.dot(q, k_local) * sm_scale_log2e
                m_ij = tl.maximum(m_i, tl.max(qk_local, axis=1))
                p = tl.exp2(qk_local - m_ij[:, None])
                l_ij = tl.sum(p, axis=1)
                acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
                v_local = tl.load(
                    v_cache_ptr
                    + page_local * stride_kv_blk
                    + off_n[:, None] * stride_kv_pos
                    + pid_kh * stride_kv_h
                    + off_d[None, :] * stride_kv_d,
                    mask=pos_mask_local[:, None] & d_mask[None, :],
                    other=0.0,
                )
                if USE_FP8:
                    v_local = v_local.to(q.dtype)
                acc_o += tl.dot(p.to(v_local.dtype), v_local)
                m_i = m_ij
                lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i, boundary_check=(0,))

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# Wrappers; same signature as the sparse ones plus ``blk_bias``.
@torch.no_grad()
def dash_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_cache: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk] int32
    blk_bias: torch.Tensor | None,  # [num_kv_heads, total_q, topk] fp32
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    local_blocks: int = 0,
    sparse_block_size: int = SPARSE_BLOCK_SIZE,
) -> None:
    """Bias-aware block-sparse prefill attention. block_size_q == 1."""
    _, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = k_cache.dtype in _FP8_DTYPES
    has_bias = blk_bias is not None
    bias_t, b_strides = _bias_operand(blk_bias, topk_idx)
    grid = (max_query_len, num_kv_heads, batch)
    _dash_sparse_fwd_kernel[grid](
        q,
        k_cache,
        v_cache,
        topk_idx,
        bias_t,
        output,
        block_table,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        seq_lens,
        prefix_lens,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        topk,
        1,  # num_q_loop
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        b_strides[0],
        b_strides[1],
        b_strides[2],
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=1,
        BLOCK_SIZE_K=sparse_block_size,
        USE_FP8=use_fp8,
        LOCAL_BLOCKS=local_blocks,
        HAS_BIAS=has_bias,
    )


def _bias_operand(
    blk_bias: torch.Tensor | None, topk_idx: torch.Tensor
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Bias tensor and strides to launch with; a dummy keeps the signature stable."""
    if blk_bias is None:
        return topk_idx.new_zeros(1, dtype=torch.float32), (0, 0, 0)
    assert blk_bias.shape == topk_idx.shape, (blk_bias.shape, topk_idx.shape)
    assert blk_bias.dtype == torch.float32, blk_bias.dtype
    return blk_bias, blk_bias.stride()


def _num_topk_chunks(chunk_width: int, total_q: int, num_kv_heads: int) -> int:
    """Power-of-two split-K factor that fills ``TARGET_GRID`` programs."""
    target = max(1, min(chunk_width, TARGET_GRID // max(1, total_q * num_kv_heads)))
    return 1 << (target.bit_length() - 1)


def _launch_decode_partials(
    *,
    q,
    k_cache,
    v_cache,
    topk_idx,
    bias_t,
    b_strides,
    o_partial,
    lse_partial,
    block_table,
    seq_lens,
    num_kv_heads: int,
    sm_scale: float,
    decode_query_len: int,
    num_topk_chunks: int,
    local_blocks: int,
    sparse_block_size: int,
    has_bias: bool,
    use_pdl: bool,
) -> None:
    total_q, num_heads, head_dim = q.shape
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    grid = (total_q * num_topk_chunks, num_kv_heads)
    _dash_sparse_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        topk_idx,
        bias_t,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        total_q,
        num_kv_heads,
        num_heads // num_kv_heads,
        head_dim,
        topk_idx.shape[-1],
        sm_scale,
        decode_query_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        b_strides[0],
        b_strides[1],
        b_strides[2],
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=sparse_block_size,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_FP8=k_cache.dtype in _FP8_DTYPES,
        USE_PDL=use_pdl,
        LOCAL_BLOCKS=local_blocks,
        HAS_BIAS=has_bias,
        **pdl_launch,
    )


def _launch_merge_partials(
    *, o_partial, lse_partial, output, num_topk_chunks: int, use_pdl: bool
) -> None:
    total_q, num_heads, head_dim = output.shape
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    _merge_topk_attn_out_kernel[(total_q, num_heads)](
        o_partial,
        lse_partial,
        output,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )


@torch.no_grad()
def dash_sparse_attn_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,
    blk_bias: torch.Tensor | None,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    local_blocks: int = 0,
    sparse_block_size: int = SPARSE_BLOCK_SIZE,
    chunk_width: int | None = None,
    cg_rows: int | None = None,
) -> None:
    """Bias-aware block-sparse decode with split-K merging.

    ``chunk_width`` keeps the unnarrowed chunk count; ``cg_rows`` preallocates
    the max CUDA-graph scratch geometry (rows x peak chunks).
    """
    total_q, num_heads, head_dim = q.shape
    assert total_q == seq_lens.shape[0] * decode_query_len
    use_pdl = _arch_supports_pdl()
    bias_t, b_strides = _bias_operand(blk_bias, topk_idx)
    cw = topk_idx.shape[-1] if chunk_width is None else int(chunk_width)
    num_topk_chunks = _num_topk_chunks(cw, total_q, num_kv_heads)
    o_partial, lse_partial = _decode_partials(
        num_topk_chunks,
        total_q,
        num_heads,
        head_dim,
        q.dtype,
        q.device,
        # Max geometry of the capture sweep: chunks peak at total_q == 1, rows
        # at the largest captured batch.
        alloc_chunks=0 if not cg_rows else _max_topk_chunks(cw, num_kv_heads),
        alloc_rows=0 if not cg_rows else int(cg_rows),
    )
    _launch_decode_partials(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        topk_idx=topk_idx,
        bias_t=bias_t,
        b_strides=b_strides,
        o_partial=o_partial,
        lse_partial=lse_partial,
        block_table=block_table,
        seq_lens=seq_lens,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
        decode_query_len=decode_query_len,
        num_topk_chunks=num_topk_chunks,
        local_blocks=local_blocks,
        sparse_block_size=sparse_block_size,
        has_bias=blk_bias is not None,
        use_pdl=use_pdl,
    )
    _launch_merge_partials(
        o_partial=o_partial,
        lse_partial=lse_partial,
        output=output,
        num_topk_chunks=num_topk_chunks,
        use_pdl=use_pdl,
    )
