"""
Split-K Attention Kernel for Large TopK Cases

This module implements a split-K version of the attention kernel that:
1. Splits the K (topk) dimension across multiple kernel instances
2. Each instance computes partial results with its own m_i, l_i, and accumulators
3. A combine kernel merges the partial results using online softmax

This reduces register pressure by processing fewer K tokens per kernel instance,
improving occupancy and overall performance for large topk cases.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .triton_mla_kernels_decode_common import _bucket_total_tokens


# ============================================================================
# Split-K Attention Kernel
# ============================================================================
@triton.autotune(
    configs=[
        # num_stages=2 configs for software pipelining
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=2
        ),
        # num_stages=1 baseline configs
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 8, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 8, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
    ],
    key=["total_tokens_bucket", "h_q", "topk_per_split", "d_qk"],
)
@triton.jit
def _splitk_attention_kernel(
    Q,
    KV,
    Mask,
    PartialOutput,
    PartialLSE,
    PartialM,
    sm_scale,
    total_tokens,
    total_tokens_bucket,
    h_q,
    total_topk,
    d_qk,
    d_v,
    topk_per_split,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_t,
    stride_kv_k,
    stride_kv_d,
    stride_mask_t,
    stride_mask_k,
    stride_po_s,
    stride_po_t,
    stride_po_h,
    stride_po_d,
    stride_plse_s,
    stride_plse_t,
    stride_plse_h,
    stride_pm_s,
    stride_pm_t,
    stride_pm_h,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Split-K attention kernel that processes a subset of K tokens."""
    LOG2E: tl.constexpr = 1.4426950408889634

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)
    pid_t_64 = pid_t.to(tl.int64)

    NEG_INF = float("-inf")

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    # Compute K range for this split
    k_start = pid_k * topk_per_split
    k_end = tl.minimum(k_start + topk_per_split, total_topk)

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    stride_q_t_64 = tl.cast(stride_q_t, tl.int64)
    stride_kv_t_64 = tl.cast(stride_kv_t, tl.int64)
    stride_mask_t_64 = tl.cast(stride_mask_t, tl.int64)
    q_base = Q + pid_t_64 * stride_q_t_64
    kv_base = KV + pid_t_64 * stride_kv_t_64
    mask_base = Mask + pid_t_64 * stride_mask_t_64

    for n_start in range(k_start, k_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < k_end

        mask_ptrs = mask_base + offs_n * stride_mask_k
        invalid = tl.load(mask_ptrs, mask=mask_n, other=True)
        valid = mask_n & ~invalid

        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)

        for d_start in range(0, d_qk, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < d_qk

            q_ptrs = (
                q_base + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
            )
            q_chunk = tl.load(
                q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0
            ).to(tl.bfloat16)

            k_ptrs = (
                kv_base + offs_n[:, None] * stride_kv_k + offs_d[None, :] * stride_kv_d
            )
            k_chunk = tl.load(
                k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0
            ).to(tl.bfloat16)

            qk += tl.dot(q_chunk, tl.trans(k_chunk))

        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, NEG_INF)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
        p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
        l_new = alpha * l_i + tl.sum(p, axis=1)
        p_bf16 = p.to(tl.bfloat16)

        offs_v = tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, v)

        offs_v = BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(
            v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0
        ).to(tl.bfloat16)
        acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, v)

        offs_v = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(
            v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0
        ).to(tl.bfloat16)
        acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, v)

        offs_v = 3 * BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(
            v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0
        ).to(tl.bfloat16)
        acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, v)

        m_i = m_new
        l_i = l_new

    # Store partial results
    stride_po_s_64 = tl.cast(stride_po_s, tl.int64)
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64)
    po_base = PartialOutput + pid_k * stride_po_s_64 + pid_t_64 * stride_po_t_64

    offs_h_2d = offs_h[:, None]
    mask_h_2d = mask_h[:, None]
    offs_v_0 = tl.arange(0, BLOCK_D)
    offs_v_1 = BLOCK_D + tl.arange(0, BLOCK_D)
    offs_v_2 = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_v_3 = 3 * BLOCK_D + tl.arange(0, BLOCK_D)

    tl.store(
        po_base + offs_h_2d * stride_po_h + offs_v_0[None, :] * stride_po_d,
        acc_0,
        mask=mask_h_2d,
    )
    tl.store(
        po_base + offs_h_2d * stride_po_h + offs_v_1[None, :] * stride_po_d,
        acc_1,
        mask=mask_h_2d & (offs_v_1[None, :] < d_v),
    )
    tl.store(
        po_base + offs_h_2d * stride_po_h + offs_v_2[None, :] * stride_po_d,
        acc_2,
        mask=mask_h_2d & (offs_v_2[None, :] < d_v),
    )
    tl.store(
        po_base + offs_h_2d * stride_po_h + offs_v_3[None, :] * stride_po_d,
        acc_3,
        mask=mask_h_2d & (offs_v_3[None, :] < d_v),
    )

    stride_plse_s_64 = tl.cast(stride_plse_s, tl.int64)
    stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64)
    plse_ptrs = (
        PartialLSE
        + pid_k * stride_plse_s_64
        + pid_t_64 * stride_plse_t_64
        + offs_h * stride_plse_h
    )
    tl.store(plse_ptrs, l_i, mask=mask_h)

    stride_pm_s_64 = tl.cast(stride_pm_s, tl.int64)
    stride_pm_t_64 = tl.cast(stride_pm_t, tl.int64)
    pm_ptrs = (
        PartialM
        + pid_k * stride_pm_s_64
        + pid_t_64 * stride_pm_t_64
        + offs_h * stride_pm_h
    )
    tl.store(pm_ptrs, m_i, mask=mask_h)


# ============================================================================
# Combine Kernel for Split-K
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 32, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 8, "BLOCK_D": 128}, num_warps=4, num_stages=1),
    ],
    key=["total_tokens_bucket", "h_q", "split_k"],
)
@triton.jit
def _combine_splitk_attention_kernel(
    PartialOutput,
    PartialLSE,
    PartialM,
    AttnSink,
    Output,
    LSE,
    total_tokens,
    total_tokens_bucket,
    h_q,
    d_v,
    split_k,
    stride_po_s,
    stride_po_t,
    stride_po_h,
    stride_po_d,
    stride_plse_s,
    stride_plse_t,
    stride_plse_h,
    stride_pm_s,
    stride_pm_t,
    stride_pm_h,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_t,
    stride_lse_h,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Combine partial results from split-K attention kernel."""
    LOG2E: tl.constexpr = 1.4426950408889634
    NEG_INF = float("-inf")
    POS_INF = float("+inf")

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    m_acc = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    stride_po_s_64 = tl.cast(stride_po_s, tl.int64)
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64)
    stride_plse_s_64 = tl.cast(stride_plse_s, tl.int64)
    stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64)
    stride_pm_s_64 = tl.cast(stride_pm_s, tl.int64)
    stride_pm_t_64 = tl.cast(stride_pm_t, tl.int64)

    offs_h_2d = offs_h[:, None]
    mask_h_2d = mask_h[:, None]
    offs_v_0 = tl.arange(0, BLOCK_D)
    offs_v_1 = BLOCK_D + tl.arange(0, BLOCK_D)
    offs_v_2 = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_v_3 = 3 * BLOCK_D + tl.arange(0, BLOCK_D)

    for k in range(split_k):
        k_64 = tl.cast(k, tl.int64)
        po_base = PartialOutput + k_64 * stride_po_s_64 + pid_t_64 * stride_po_t_64

        p_acc_0 = tl.load(
            po_base + offs_h_2d * stride_po_h + offs_v_0[None, :] * stride_po_d,
            mask=mask_h_2d,
            other=0.0,
        )
        p_acc_1 = tl.load(
            po_base + offs_h_2d * stride_po_h + offs_v_1[None, :] * stride_po_d,
            mask=mask_h_2d & (offs_v_1[None, :] < d_v),
            other=0.0,
        )
        p_acc_2 = tl.load(
            po_base + offs_h_2d * stride_po_h + offs_v_2[None, :] * stride_po_d,
            mask=mask_h_2d & (offs_v_2[None, :] < d_v),
            other=0.0,
        )
        p_acc_3 = tl.load(
            po_base + offs_h_2d * stride_po_h + offs_v_3[None, :] * stride_po_d,
            mask=mask_h_2d & (offs_v_3[None, :] < d_v),
            other=0.0,
        )

        plse_ptrs = (
            PartialLSE
            + k_64 * stride_plse_s_64
            + pid_t_64 * stride_plse_t_64
            + offs_h * stride_plse_h
        )
        p_l = tl.load(plse_ptrs, mask=mask_h, other=0.0)

        pm_ptrs = (
            PartialM
            + k_64 * stride_pm_s_64
            + pid_t_64 * stride_pm_t_64
            + offs_h * stride_pm_h
        )
        p_m = tl.load(pm_ptrs, mask=mask_h, other=NEG_INF)

        m_new = tl.maximum(m_acc, p_m)
        alpha_acc = tl.where(
            m_acc == NEG_INF, 0.0, tl.math.exp2((m_acc - m_new) * LOG2E)
        )
        alpha_p = tl.where(p_m == NEG_INF, 0.0, tl.math.exp2((p_m - m_new) * LOG2E))
        l_new = alpha_acc * l_acc + alpha_p * p_l

        acc_0 = acc_0 * alpha_acc[:, None] + p_acc_0 * alpha_p[:, None]
        acc_1 = acc_1 * alpha_acc[:, None] + p_acc_1 * alpha_p[:, None]
        acc_2 = acc_2 * alpha_acc[:, None] + p_acc_2 * alpha_p[:, None]
        acc_3 = acc_3 * alpha_acc[:, None] + p_acc_3 * alpha_p[:, None]

        m_acc = m_new
        l_acc = l_new

    lse = m_acc + tl.math.log2(tl.where(l_acc == 0.0, 1.0, l_acc)) / LOG2E
    is_lonely_q = l_acc == 0.0

    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_acc) * LOG2E)
        denominator = l_acc + exp_attn_sink_minus_m
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        output_scale = 1.0 / denominator
    else:
        output_scale = tl.where(l_acc == 0.0, 0.0, 1.0 / l_acc)

    is_lonely_q_2d = is_lonely_q[:, None]
    output_scale_2d = output_scale[:, None]
    acc_0 = tl.where(is_lonely_q_2d, 0.0, acc_0 * output_scale_2d)
    acc_1 = tl.where(is_lonely_q_2d, 0.0, acc_1 * output_scale_2d)
    acc_2 = tl.where(is_lonely_q_2d, 0.0, acc_2 * output_scale_2d)
    acc_3 = tl.where(is_lonely_q_2d, 0.0, acc_3 * output_scale_2d)
    lse = tl.where(is_lonely_q, POS_INF, lse)

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64

    tl.store(
        o_base + offs_h_2d * stride_o_h + offs_v_0[None, :] * stride_o_d,
        acc_0.to(tl.bfloat16),
        mask=mask_h_2d,
    )
    tl.store(
        o_base + offs_h_2d * stride_o_h + offs_v_1[None, :] * stride_o_d,
        acc_1.to(tl.bfloat16),
        mask=mask_h_2d & (offs_v_1[None, :] < d_v),
    )
    tl.store(
        o_base + offs_h_2d * stride_o_h + offs_v_2[None, :] * stride_o_d,
        acc_2.to(tl.bfloat16),
        mask=mask_h_2d & (offs_v_2[None, :] < d_v),
    )
    tl.store(
        o_base + offs_h_2d * stride_o_h + offs_v_3[None, :] * stride_o_d,
        acc_3.to(tl.bfloat16),
        mask=mask_h_2d & (offs_v_3[None, :] < d_v),
    )

    stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64)
    tl.store(LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h, lse, mask=mask_h)


# ============================================================================
# Runner Function
# ============================================================================
def run_splitk_attention(
    q_reshaped: torch.Tensor,
    gathered_kv: torch.Tensor,
    invalid_mask: torch.Tensor,
    d_v: int,
    sm_scale: float,
    total_tokens: int,
    h_q: int,
    total_topk: int,
    d_qk: int,
    attn_sink: Optional[torch.Tensor] = None,
    split_k: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run split-K attention kernel."""
    device = q_reshaped.device

    topk_per_split = (total_topk + split_k - 1) // split_k

    partial_output = torch.empty(
        split_k, total_tokens, h_q, d_v, dtype=torch.float32, device=device
    )
    partial_lse = torch.empty(
        split_k, total_tokens, h_q, dtype=torch.float32, device=device
    )
    partial_m = torch.empty(
        split_k, total_tokens, h_q, dtype=torch.float32, device=device
    )

    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

    grid_splitk = lambda meta: (
        total_tokens,
        triton.cdiv(h_q, meta["BLOCK_H"]),
        split_k,
    )
    _splitk_attention_kernel[grid_splitk](
        q_reshaped,
        gathered_kv,
        invalid_mask,
        partial_output,
        partial_lse,
        partial_m,
        sm_scale,
        total_tokens,
        _bucket_total_tokens(total_tokens),
        h_q,
        total_topk,
        d_qk,
        d_v,
        topk_per_split,
        q_reshaped.stride(0),
        q_reshaped.stride(1),
        q_reshaped.stride(2),
        gathered_kv.stride(0),
        gathered_kv.stride(1),
        gathered_kv.stride(2),
        invalid_mask.stride(0),
        invalid_mask.stride(1),
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
    )

    HAS_ATTN_SINK = attn_sink is not None
    attn_sink_tensor = attn_sink if HAS_ATTN_SINK else lse[:1]

    grid_combine = lambda meta: (total_tokens, triton.cdiv(h_q, meta["BLOCK_H"]))
    _combine_splitk_attention_kernel[grid_combine](
        partial_output,
        partial_lse,
        partial_m,
        attn_sink_tensor,
        output,
        lse,
        total_tokens,
        _bucket_total_tokens(total_tokens),
        h_q,
        d_v,
        split_k,
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        HAS_ATTN_SINK=HAS_ATTN_SINK,
    )

    return output, lse
