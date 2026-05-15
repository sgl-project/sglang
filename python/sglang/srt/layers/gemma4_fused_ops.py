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
    K_ptr,
    V_ptr,
    Q_w_ptr,
    K_w_ptr,
    stride_q_m,
    stride_k_m,
    stride_v_m,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    eps,
    HAS_KV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-token fused RMSNorm of Q (with q_w), K (with k_w), V (no scale).

    Layout assumption: each tensor's last dim packs (num_heads, head_dim) contiguously
    so per-head offset is `h * HEAD_DIM`. The token (M) stride is taken from
    stride_*_m so the kernel works on strided views (e.g. slices of a larger
    qkv buffer produced by `qkv.split`) without requiring `.contiguous()` copies.
    V uses `weight=ones` semantics so the multiply-by-weight is omitted.
    """
    m = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < HEAD_DIM

    qw = tl.load(Q_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Q heads
    for h in tl.static_range(NUM_Q_HEADS):
        off = m * stride_q_m + h * HEAD_DIM + cols
        x = tl.load(Q_ptr + off, mask=mask, other=0.0).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
        out = x * rrms * qw
        tl.store(Q_ptr + off, out.to(Q_ptr.dtype.element_ty), mask=mask)

    if HAS_KV:
        kw = tl.load(K_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # K heads
        for h in tl.static_range(NUM_KV_HEADS):
            off = m * stride_k_m + h * HEAD_DIM + cols
            x = tl.load(K_ptr + off, mask=mask, other=0.0).to(tl.float32)
            rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
            out = x * rrms * kw
            tl.store(K_ptr + off, out.to(K_ptr.dtype.element_ty), mask=mask)

        # V heads (no scaling: V-norm uses weight=ones)
        for h in tl.static_range(NUM_KV_HEADS):
            off = m * stride_v_m + h * HEAD_DIM + cols
            x = tl.load(V_ptr + off, mask=mask, other=0.0).to(tl.float32)
            rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
            out = x * rrms
            tl.store(V_ptr + off, out.to(V_ptr.dtype.element_ty), mask=mask)


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
) -> None:
    """In-place fused RMSNorm on Q, K, V for Gemma4 attention.

    All three norms compute `x * rsqrt(mean(x^2) + eps)` independently per head.
    Q is scaled by `q_weight`, K by `k_weight`, V by 1 (Gemma4's V-norm has
    `with_scale=False`).

    Inputs may be 2D `(M, num_heads * head_dim)` or strided views of a larger
    buffer (such as q/k/v slices from `qkv.split`). The kernel uses the actual
    `stride(0)` so no `.contiguous()` copy is required. Within a token, the
    last dim must be contiguous so heads pack as `h * head_dim` offsets.

    If k and v are both None (KV-shared layer), only Q is normalized.
    """
    assert q.is_cuda
    assert q.stride(-1) == 1, "Q's last dim must be contiguous"
    assert q_weight.shape[-1] == head_dim
    M = q.shape[0] if q.dim() >= 2 else 1
    BLOCK = triton.next_power_of_2(head_dim)

    has_kv = k is not None and v is not None
    if has_kv:
        assert k.is_cuda and v.is_cuda
        assert k.stride(-1) == 1 and v.stride(-1) == 1
        assert k_weight is not None and k_weight.shape[-1] == head_dim

    _gemma_qkv_rmsnorm_kernel[(M,)](
        q,
        k if has_kv else q,
        v if has_kv else q,
        q_weight,
        k_weight if has_kv else q_weight,
        q.stride(0),
        k.stride(0) if has_kv else 0,
        v.stride(0) if has_kv else 0,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads if has_kv else 0,
        HEAD_DIM=head_dim,
        eps=eps,
        HAS_KV=has_kv,
        BLOCK=BLOCK,
    )


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


@triton.jit
def _fused_kv_norm_kernel(
    X_ptr,  # input: shared K/V raw projection, shape [M, N]
    K_weight_ptr,  # k_norm weight, shape [N]
    K_out_ptr,  # output: normalised K, shape [M, N]
    V_out_ptr,  # output: normalised V, shape [M, N]
    stride_x,
    stride_k,
    stride_v,
    N,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: reads x once, writes k = rmsnorm(x, k_weight) and v = rmsnorm(x).

    For attention_k_eq_v layers where K and V share the same raw projection:
      - K = x * rrms * k_weight   (standard RMSNorm with learned scale)
      - V = x * rrms              (RMSNorm without scale, i.e. unit normalisation)

    Both share the same rrms = rsqrt(mean(x^2) + eps), so we compute it once.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load input once
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # Load k_norm weights
    k_w = tl.load(K_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Shared RMS computation
    var = tl.sum(x * x, axis=0) / N
    rrms = tl.rsqrt(var + eps)

    # V = x * rrms (no learned scale)
    v_out = x * rrms

    # K = x * rrms * k_weight
    k_out = v_out * k_w

    # Store both outputs
    tl.store(K_out_ptr + row * stride_k + cols, k_out.to(x.dtype), mask=mask)
    tl.store(V_out_ptr + row * stride_v + cols, v_out.to(x.dtype), mask=mask)


@triton.jit
def _gemma_q_keqv_rmsnorm_kernel(
    Q_ptr,  # in/out: per-token Q heads, shape [M, NUM_Q_HEADS * HEAD_DIM]
    KV_ptr,  # input: per-token K=V raw projection, shape [M, NUM_KV_HEADS * HEAD_DIM]
    K_out_ptr,  # output: per-token K, shape [M, NUM_KV_HEADS * HEAD_DIM]
    V_out_ptr,  # output: per-token V, shape [M, NUM_KV_HEADS * HEAD_DIM]
    Q_w_ptr,
    K_w_ptr,
    stride_q_m,
    stride_kv_m,
    stride_kout_m,
    stride_vout_m,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    """Per-token fused RMSNorm of Q (in-place, with q_w) and K=V (out-of-place).

    For the ``attention_k_eq_v`` path, the K and V projections share the same
    raw input. We compute a single rrms per (token, head) over that shared
    input and write:
        K = x * rrms * k_weight   (standard RMSNorm with learned scale)
        V = x * rrms              (Gemma4's V-norm has weight=ones)

    Q is normalised in-place exactly as in :func:`_gemma_qkv_rmsnorm_kernel`.
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

    kw = tl.load(K_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # K = norm(KV) * kw, V = norm(KV)  — shared rrms per (token, head)
    for h in tl.static_range(NUM_KV_HEADS):
        in_off = m * stride_kv_m + h * HEAD_DIM + cols
        x = tl.load(KV_ptr + in_off, mask=mask, other=0.0).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
        v_out = x * rrms
        k_out = v_out * kw
        k_off = m * stride_kout_m + h * HEAD_DIM + cols
        v_off = m * stride_vout_m + h * HEAD_DIM + cols
        tl.store(K_out_ptr + k_off, k_out.to(K_out_ptr.dtype.element_ty), mask=mask)
        tl.store(V_out_ptr + v_off, v_out.to(V_out_ptr.dtype.element_ty), mask=mask)


def gemma_q_keqv_rmsnorm(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused per-token RMSNorm for Q + (K = V) attention layers.

    Inputs:
      q: [M, num_q_heads * head_dim] (CUDA, contiguous last dim).
         Normalised in-place: ``q = norm(q) * q_weight``.
      kv: [M, num_kv_heads * head_dim] (CUDA, contiguous last dim) — the
         shared raw K/V projection output.
      q_weight, k_weight: [head_dim] learned RMSNorm scales.

    Returns:
      (k, v) — newly-allocated tensors with the same shape and stride
      pattern as ``kv``. ``k = norm(kv) * k_weight``, ``v = norm(kv)``.

    Replaces the (q_norm + fused_kv_norm) two-launch sequence with a single
    Triton launch.
    """
    assert q.is_cuda and kv.is_cuda
    assert q.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.shape[-1] == head_dim and k_weight.shape[-1] == head_dim
    assert q.shape[0] == kv.shape[0], f"M mismatch: {q.shape[0]} vs {kv.shape[0]}"

    M = q.shape[0]
    BLOCK = triton.next_power_of_2(head_dim)

    k_out = torch.empty_like(kv)
    v_out = torch.empty_like(kv)

    _gemma_q_keqv_rmsnorm_kernel[(M,)](
        q,
        kv,
        k_out,
        v_out,
        q_weight,
        k_weight,
        q.stride(0),
        kv.stride(0),
        k_out.stride(0),
        v_out.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        eps=eps,
        BLOCK=BLOCK,
    )
    return k_out, v_out


def fused_kv_norm(
    x: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused k_norm + v_derive for attention_k_eq_v layers.

    Given input x (the shared K/V projection output), computes:
      k = rmsnorm(x, k_weight)    — standard RMSNorm with learned scale
      v = rmsnorm(x)              — RMSNorm with unit scale (no learned weights)

    Both norms share the same RMS denominator, so we read x once and compute
    rsqrt(mean(x^2) + eps) once.

    Args:
        x: Input tensor of shape [*, head_dim].  Will be reshaped to 2D
           internally; the last dimension is the normalisation dimension.
        k_weight: Learned scale for k_norm, shape [head_dim].
        eps: Epsilon for numerical stability.

    Returns:
        (k, v) tuple of tensors with the same shape as x.
    """
    needs_reshape = x.dim() != 2
    if needs_reshape:
        original_shape = x.shape
        x = x.contiguous().reshape(-1, original_shape[-1])

    assert x.stride(-1) == 1, "Expected contiguous last dimension"
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)

    k_out = torch.empty_like(x)
    v_out = torch.empty_like(x)

    _fused_kv_norm_kernel[(M,)](
        x,
        k_weight,
        k_out,
        v_out,
        x.stride(0),
        k_out.stride(0),
        v_out.stride(0),
        N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if needs_reshape:
        k_out = k_out.reshape(original_shape)
        v_out = v_out.reshape(original_shape)

    return k_out, v_out
