"""Fused axis-stacked GDN prefill prologue: split + gating + q/k L2 norm in ONE launch.

Fuses the GDN extend prologue chain ``fused_qkv_split_gdn_prefill`` ->
``fused_gdn_gating`` -> ``l2norm_fwd_qk`` into a single kernel whose 1D grid is
range-partitioned by program id into four roles (q-norm rows, k-norm rows,
v-copy, gating). Each role body is a clone of the corresponding standalone
kernel, with q/k loads addressed directly into the packed ``mixed_qkv`` — the
raw (un-normalized) q/k intermediates never reach global memory.

Clone discipline (do NOT "clean up" without re-proving the 0-ULP matrix in
test_gdn_prefill_flashinfer_opts.py):

1. The gating body must use ``tl.exp``/``tl.log`` (fast paths) for softplus and
   ``-exp(A_log)``, then ``libdevice.exp`` for FlashInfer's alpha — swapping
   either exponential silently breaks bit-parity with the standalone kernels.
2. The norm role's 0-ULP equality under manual-pointer loads (vs the original
   block-ptr loads) is a Triton lowering property pinned by CI, not a language
   contract; a Triton upgrade that breaks it must fail the 0-ULP matrix test.
3. If the causal conv1d is ever fused in, q/k MUST be rounded to
   bf16 in-register before the L2 reduction: today the norm reads the conv's
   already-rounded bf16 output, and norming pre-round fp32 values diverges on
   ~26% of elements.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _norm_body_packed(
    mixed_qkv,
    y,
    eps,
    i_t,
    base_col,
    stride_t,
    t_rows,
    NUM_QK_HEADS: tl.constexpr,
    HEAD_QK_DIM: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    # Clone of fla.l2norm._l2norm_row_block (fp32 reduce, DIVISION form, eps
    # inside sqrt, bf16 round on store) with the load addressed into packed
    # mixed_qkv: row r -> token r // H, head r % H, col c ->
    # mixed_qkv[tok, base_col + head * HEAD_QK_DIM + c].
    rows = i_t * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    tok = rows // NUM_QK_HEADS
    head = rows % NUM_QK_HEADS
    m = (rows < t_rows)[:, None] & (cols < HEAD_QK_DIM)[None, :]
    p_x = (
        mixed_qkv
        + tok[:, None] * stride_t
        + (base_col + head[:, None] * HEAD_QK_DIM + cols[None, :])
    )
    b_x = tl.load(p_x, mask=m, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = y + rows[:, None] * HEAD_QK_DIM + cols[None, :]
    tl.store(p_y, b_y.to(y.dtype.element_ty), mask=m)


@triton.jit
def _vcopy_body(
    mixed_qkv,
    v,
    i_t,
    stride_t,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Split-kernel body specialized to the v region: a pure bf16 bit-copy
    # (zero casts), one program per token.
    offsets = tl.arange(0, BLOCK_V)
    mask = offsets < V_DIM
    values = tl.load(mixed_qkv + i_t * stride_t + QK_DIM + offsets, mask=mask)
    tl.store(v + i_t * V_DIM + offsets, values, mask=mask)


@triton.jit
def _gating_body_matched(
    alpha,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    i_g,
    seq_len,
    stride_a,
    stride_b,
    NUM_V_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BT_G: tl.constexpr,
):
    # fused_gdn_gating_kernel math on a matched BT_G x NUM_V_HEADS tile
    # (elementwise, lane-remap-invariant => bit-identical to the nw=1 original;
    # the literal nw=1-shaped clone floods 96.9%-idle CTAs and erases the win).
    t_off = i_g * BT_G + tl.arange(0, BT_G)[:, None]
    head_off = tl.arange(0, NUM_V_HEADS)[None, :]
    mask = t_off < seq_len
    off = t_off * NUM_V_HEADS + head_off
    blk_A_log = tl.load(A_log + head_off)
    blk_a = tl.load(a + t_off * stride_a + head_off, mask=mask)
    blk_b = tl.load(b + t_off * stride_b + head_off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    blk_alpha = libdevice.exp(blk_g)
    tl.store(alpha + off, blk_alpha.to(alpha.dtype.element_ty), mask=mask)
    # beta quirk kept verbatim from fused_gdn_gating_kernel: the sigmoid is
    # rounded to b's dtype (bf16) before the fp32 store.
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


@triton.jit
def gdn_prefill_fused_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    q_norm,
    k_norm,
    v,
    alpha,
    beta_out,
    eps,
    seq_len,
    t_rows,
    nbn,
    stride_t,
    stride_a,
    stride_b,
    NUM_QK_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_QK_DIM: tl.constexpr,
    HEAD_V_DIM: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BT_G: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
):
    # 1D grid range-partitioned by pid (zero dead programs):
    #   [0, nbn)                q-norm  (BT=16 row-blocks over (Hk*T, 128))
    #   [nbn, 2*nbn)            k-norm
    #   [2*nbn, 2*nbn+T)        v-copy  (1 program/token, split-native)
    #   [2*nbn+T, +cdiv(T,BT_G)) gating (matched BT_G x Hv tile)
    # Sizes (seq_len/t_rows/nbn) and strides are runtime scalars so distinct
    # prefill lengths reuse one compiled specialization.
    pid = tl.program_id(0)
    Q_DIM: tl.constexpr = NUM_QK_HEADS * HEAD_QK_DIM
    K_DIM: tl.constexpr = NUM_QK_HEADS * HEAD_QK_DIM
    V_DIM: tl.constexpr = NUM_V_HEADS * HEAD_V_DIM
    if pid < nbn:
        _norm_body_packed(
            mixed_qkv,
            q_norm,
            eps,
            pid,
            0,
            stride_t,
            t_rows,
            NUM_QK_HEADS,
            HEAD_QK_DIM,
            BT,
            BD,
        )
    elif pid < 2 * nbn:
        _norm_body_packed(
            mixed_qkv,
            k_norm,
            eps,
            pid - nbn,
            Q_DIM,
            stride_t,
            t_rows,
            NUM_QK_HEADS,
            HEAD_QK_DIM,
            BT,
            BD,
        )
    elif pid < 2 * nbn + seq_len:
        _vcopy_body(
            mixed_qkv, v, pid - 2 * nbn, stride_t, Q_DIM + K_DIM, V_DIM, BLOCK_V
        )
    else:
        _gating_body_matched(
            alpha,
            beta_out,
            A_log,
            a,
            b,
            dt_bias,
            pid - 2 * nbn - seq_len,
            seq_len,
            stride_a,
            stride_b,
            NUM_V_HEADS,
            beta,
            threshold,
            BT_G,
        )


def gdn_prefill_fused(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    num_qk_heads: int,
    num_v_heads: int,
    head_qk_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-launch GDN prefill prologue.

    Args:
        mixed_qkv: packed conv output ``(T, q_dim + k_dim + v_dim)``,
            column layout ``[q | k | v]``, last dim contiguous.
        a, b: gating projections ``(T, num_v_heads)`` (rows may be strided).
        A_log, dt_bias: per-head gating parameters ``(num_v_heads,)``.

    Returns:
        ``(q_norm, k_norm, v, alpha, beta)`` with shapes ``(1, T, Hk, Dqk)`` bf16,
        ``(1, T, Hk, Dqk)`` bf16, ``(1, T, Hv, Dv)`` bf16, ``(1, T, Hv)`` fp32,
        ``(1, T, Hv)`` fp32 — bit-identical to the three-kernel chain.
    """
    seq_len, qkv_dim = mixed_qkv.shape
    assert qkv_dim == 2 * num_qk_heads * head_qk_dim + num_v_heads * head_v_dim
    assert mixed_qkv.stride(1) == 1, "fused path requires a row-contiguous mixed_qkv"
    assert head_qk_dim == 128, "norm body is the verified BD=128 clone shape"
    assert (
        num_v_heads & (num_v_heads - 1) == 0
    ), "matched gating tile requires power-of-two num_v_heads (tl.arange)"
    assert A_log.shape == dt_bias.shape == (num_v_heads,)
    assert a.shape == b.shape == (seq_len, num_v_heads)
    assert a.stride(1) == 1 and b.stride(1) == 1

    dtype = mixed_qkv.dtype
    device = mixed_qkv.device
    q_norm = torch.empty(
        (1, seq_len, num_qk_heads, head_qk_dim), dtype=dtype, device=device
    )
    k_norm = torch.empty(
        (1, seq_len, num_qk_heads, head_qk_dim), dtype=dtype, device=device
    )
    v = torch.empty((1, seq_len, num_v_heads, head_v_dim), dtype=dtype, device=device)
    alpha = torch.empty((1, seq_len, num_v_heads), dtype=torch.float32, device=device)
    beta_out = torch.empty(
        (1, seq_len, num_v_heads), dtype=torch.float32, device=device
    )

    BT = 16  # l2norm's native row-block; also the matched gating token tile
    t_rows = seq_len * num_qk_heads
    nbn = triton.cdiv(t_rows, BT)
    grid = (2 * nbn + seq_len + triton.cdiv(seq_len, BT),)
    gdn_prefill_fused_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        q_norm,
        k_norm,
        v,
        alpha,
        beta_out,
        eps,
        seq_len,
        t_rows,
        nbn,
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        NUM_QK_HEADS=num_qk_heads,
        NUM_V_HEADS=num_v_heads,
        HEAD_QK_DIM=head_qk_dim,
        HEAD_V_DIM=head_v_dim,
        BT=BT,
        BD=head_qk_dim,
        BLOCK_V=triton.next_power_of_2(num_v_heads * head_v_dim),
        BT_G=BT,
        beta=1.0,
        threshold=20.0,
        num_warps=8,
        num_stages=3,
    )
    return q_norm, k_norm, v, alpha, beta_out
