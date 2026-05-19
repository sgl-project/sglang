"""Fused triton kernels for Gemma4 decoder layer operations.

Fuses standard RMSNorm + residual-add (+ optional scalar multiply) into
a single kernel pass to reduce kernel launch overhead.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_rmsnorm_residual_kernel(
    X_ptr,
    W_ptr,
    Residual_ptr,
    Scalar_ptr,
    Out_ptr,
    stride_x,
    stride_r,
    stride_o,
    N,
    eps,
    HAS_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: out = rmsnorm(x, w) + residual [* scalar]

    When HAS_SCALAR is True, also multiplies by a scalar loaded from Scalar_ptr.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + row * stride_r + cols, mask=mask, other=0.0).to(
        tl.float32
    )

    var = tl.sum(x * x, axis=0) / N
    rrms = tl.rsqrt(var + eps)
    out = x * rrms * w + r

    if HAS_SCALAR:
        scalar = tl.load(Scalar_ptr).to(tl.float32)
        out = out * scalar

    tl.store(Out_ptr + row * stride_o + cols, out.to(x.dtype), mask=mask)


def gemma_rmsnorm_residual_scalar(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused (rmsnorm(x) + residual) * scalar."""
    assert x.dim() == 2 and x.stride(-1) == 1, "Expected contiguous 2D input"
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x)

    _gemma_rmsnorm_residual_kernel[(M,)](
        x,
        weight,
        residual,
        scalar,
        out,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        N,
        eps,
        HAS_SCALAR=True,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def _gemma_dual_rmsnorm_residual_kernel(
    X1_ptr,
    W1_ptr,
    X2_ptr,
    W2_ptr,
    W3_ptr,
    Residual_ptr,
    Scalar_ptr,
    Out_ptr,
    stride_x1,
    stride_x2,
    stride_r,
    stride_o,
    N,
    eps1,
    eps2,
    eps3,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: out = (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual) * scalar"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x1 = tl.load(X1_ptr + row * stride_x1 + cols, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X2_ptr + row * stride_x2 + cols, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(W3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + row * stride_r + cols, mask=mask, other=0.0).to(
        tl.float32
    )

    var1 = tl.sum(x1 * x1, axis=0) / N
    norm1 = x1 * tl.rsqrt(var1 + eps1) * w1

    var2 = tl.sum(x2 * x2, axis=0) / N
    norm2 = x2 * tl.rsqrt(var2 + eps2) * w2

    combined = norm1 + norm2

    var3 = tl.sum(combined * combined, axis=0) / N
    norm3 = combined * tl.rsqrt(var3 + eps3) * w3

    scalar = tl.load(Scalar_ptr).to(tl.float32)
    out = (norm3 + r) * scalar

    tl.store(Out_ptr + row * stride_o + cols, out.to(x1.dtype), mask=mask)


@triton.jit
def _gemma_qkv_rmsnorm_kernel(
    Q_ptr,
    K_in_ptr,
    V_in_ptr,
    K_out_ptr,
    V_out_ptr,
    Q_w_ptr,
    K_w_ptr,
    stride_q_m,
    stride_kin_m,
    stride_vin_m,
    stride_kout_m,
    stride_vout_m,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    eps,
    HAS_KV: tl.constexpr,
    K_EQ_V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-token fused RMSNorm of Q (with q_w), K (with k_w), V (no scale).

    Three modes, selected via the ``HAS_KV`` / ``K_EQ_V`` constexpr toggles:

    * **Q-only** (``HAS_KV=False``): normalises Q in-place from ``Q_ptr``.
      K/V pointers are unused.
    * **QKV** (``HAS_KV=True, K_EQ_V=False``): normalises Q, K, V in-place.
      ``K_in_ptr == K_out_ptr`` and ``V_in_ptr == V_out_ptr`` (the launcher
      passes the same tensor for input and output).
    * **K=V (a.k.a. ``attention_k_eq_v``)** (``HAS_KV=True, K_EQ_V=True``):
      normalises Q in-place. ``K_in_ptr`` is the shared raw K/V projection;
      ``V_in_ptr`` is unused. ``K_out_ptr`` receives ``norm(KV) * k_weight``
      and ``V_out_ptr`` receives ``norm(KV)``. One rrms per (token, head) is
      shared between K and V.
    """
    m = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < HEAD_DIM

    qw = tl.load(Q_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Q heads — in-place
    for h in tl.static_range(NUM_Q_HEADS):
        off = m * stride_q_m + h * HEAD_DIM + cols
        x = tl.load(Q_ptr + off, mask=mask, other=0.0).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
        out = x * rrms * qw
        tl.store(Q_ptr + off, out.to(Q_ptr.dtype.element_ty), mask=mask)

    if HAS_KV:
        kw = tl.load(K_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        if K_EQ_V:
            # Shared KV input: one read of KV per head, two writes.
            for h in tl.static_range(NUM_KV_HEADS):
                in_off = m * stride_kin_m + h * HEAD_DIM + cols
                x = tl.load(K_in_ptr + in_off, mask=mask, other=0.0).to(tl.float32)
                rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
                v_out = x * rrms
                k_out = v_out * kw
                k_off = m * stride_kout_m + h * HEAD_DIM + cols
                v_off = m * stride_vout_m + h * HEAD_DIM + cols
                tl.store(
                    K_out_ptr + k_off,
                    k_out.to(K_out_ptr.dtype.element_ty),
                    mask=mask,
                )
                tl.store(
                    V_out_ptr + v_off,
                    v_out.to(V_out_ptr.dtype.element_ty),
                    mask=mask,
                )
        else:
            # Separate K and V inputs, normalised in-place.
            for h in tl.static_range(NUM_KV_HEADS):
                off = m * stride_kin_m + h * HEAD_DIM + cols
                x = tl.load(K_in_ptr + off, mask=mask, other=0.0).to(tl.float32)
                rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
                out = x * rrms * kw
                tl.store(K_in_ptr + off, out.to(K_in_ptr.dtype.element_ty), mask=mask)

            # V heads (no scaling: V-norm uses weight=ones)
            for h in tl.static_range(NUM_KV_HEADS):
                off = m * stride_vin_m + h * HEAD_DIM + cols
                x = tl.load(V_in_ptr + off, mask=mask, other=0.0).to(tl.float32)
                rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
                out = x * rrms
                tl.store(V_in_ptr + off, out.to(V_in_ptr.dtype.element_ty), mask=mask)


def gemma_qkv_rmsnorm(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    q_weight: torch.Tensor,
    k_weight: Optional[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    *,
    k_eq_v: bool = False,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Fused RMSNorm on Q, K, V (or any subset) for Gemma4 attention.

    All norms compute `x * rsqrt(mean(x^2) + eps)` independently per head.
    Q is scaled by `q_weight`, K by `k_weight`, V by 1 (Gemma4's V-norm has
    `with_scale=False`).

    Three call modes:

    * **Q-only** (``k is None and v is None``): KV-shared / Q-only layers.
      Q is normalised in-place. Returns ``None``.
    * **QKV** (``k is not None and v is not None and not k_eq_v``):
      standard separate K/V. Q, K, V normalised in-place. Returns ``None``.
    * **K=V** (``k is not None and v is None and k_eq_v=True``):
      attention_k_eq_v layers. ``k`` is the shared raw K/V projection
      input. Q is normalised in-place; K and V are allocated as new
      tensors with the same shape and strides as ``k``. Returns
      ``(k_out, v_out)``.
    """
    assert q.is_cuda
    assert q.stride(-1) == 1, "Q's last dim must be contiguous"
    assert q_weight.shape[-1] == head_dim
    M = q.shape[0] if q.dim() >= 2 else 1
    BLOCK = triton.next_power_of_2(head_dim)

    # Resolve the mode + allocate outputs if needed.
    if k_eq_v:
        assert (
            k is not None and v is None
        ), "k_eq_v=True expects k=<shared KV input>, v=None"
        assert k.is_cuda and k.stride(-1) == 1
        assert k_weight is not None and k_weight.shape[-1] == head_dim
        assert (
            q.shape[0] == k.shape[0]
        ), f"M mismatch: q.shape[0]={q.shape[0]} vs kv.shape[0]={k.shape[0]}"
        has_kv = True
        k_in = k
        v_in = q  # unused; just need a valid pointer for triton.
        k_out = torch.empty_like(k)
        v_out = torch.empty_like(k)
        stride_kin_m = k.stride(0)
        stride_vin_m = 0
        stride_kout_m = k_out.stride(0)
        stride_vout_m = v_out.stride(0)
    elif k is not None and v is not None:
        assert k.is_cuda and v.is_cuda
        assert k.stride(-1) == 1 and v.stride(-1) == 1
        assert k_weight is not None and k_weight.shape[-1] == head_dim
        has_kv = True
        k_in = k
        v_in = v
        # In-place: outputs == inputs.
        k_out = k
        v_out = v
        stride_kin_m = k.stride(0)
        stride_vin_m = v.stride(0)
        stride_kout_m = k.stride(0)
        stride_vout_m = v.stride(0)
    else:
        assert k is None and v is None, "Q-only mode requires both k and v to be None"
        has_kv = False
        # Unused pointers; pass q for safety.
        k_in = v_in = k_out = v_out = q
        stride_kin_m = stride_vin_m = stride_kout_m = stride_vout_m = 0

    _gemma_qkv_rmsnorm_kernel[(M,)](
        q,
        k_in,
        v_in,
        k_out,
        v_out,
        q_weight,
        k_weight if has_kv else q_weight,
        q.stride(0),
        stride_kin_m,
        stride_vin_m,
        stride_kout_m,
        stride_vout_m,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads if has_kv else 0,
        HEAD_DIM=head_dim,
        eps=eps,
        HAS_KV=has_kv,
        K_EQ_V=k_eq_v,
        BLOCK=BLOCK,
    )

    if k_eq_v:
        return k_out, v_out
    return None


def gemma_dual_rmsnorm_residual_scalar(
    x1: torch.Tensor,
    weight1: torch.Tensor,
    x2: torch.Tensor,
    weight2: torch.Tensor,
    weight3: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps1: float = 1e-6,
    eps2: float = 1e-6,
    eps3: float = 1e-6,
) -> torch.Tensor:
    """Fused (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual) * scalar."""
    assert x1.dim() == 2 and x1.stride(-1) == 1
    M, N = x1.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x1)

    _gemma_dual_rmsnorm_residual_kernel[(M,)](
        x1,
        weight1,
        x2,
        weight2,
        weight3,
        residual,
        scalar,
        out,
        x1.stride(0),
        x2.stride(0),
        residual.stride(0),
        out.stride(0),
        N,
        eps1,
        eps2,
        eps3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
