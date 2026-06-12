"""Fused triton kernels for Gemma4 decoder layer operations.

Fuses standard RMSNorm + residual-add (+ optional scalar multiply) into
a single kernel pass to reduce kernel launch overhead.
"""

from enum import Enum
from typing import Optional

import torch
import triton
import triton.language as tl


class ProjAndNormMode(Enum):
    """Projection + RMSNorm layout for a Gemma4 attention layer.

    Q_ONLY    KV-sharing layer; only Q is projected and normalised.
    QK_ONLY   attention_k_eq_v layer; Q and a shared K/V are projected,
              the fused norm derives K and V from one K projection.
    QKV_FULL  Standard layer; Q, K, V are projected and normalised
              independently.
    """

    Q_ONLY = "q"
    QK_ONLY = "qk"
    QKV_FULL = "qkv"


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
def _gemma_qkv_rmsnorm_store(
    X_ptr,
    W_ptr,
    stride_m,
    m,
    h,
    cols,
    mask,
    HEAD_DIM: tl.constexpr,
    eps,
    HAS_WEIGHT: tl.constexpr,
):
    off = m * stride_m + h * HEAD_DIM + cols
    x = tl.load(X_ptr + off, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
    out = x * rrms
    if HAS_WEIGHT:
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = out * w
    tl.store(X_ptr + off, out.to(X_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gemma_qkv_rmsnorm_keqv_store(
    KV_in_ptr,
    K_out_ptr,
    V_out_ptr,
    K_w_ptr,
    stride_kin_m,
    stride_kout_m,
    stride_vout_m,
    m,
    h,
    cols,
    mask,
    HEAD_DIM: tl.constexpr,
    eps,
):
    """K=V store for one (token, head): read shared KV once, write
    ``K_out = norm(KV) * k_weight`` and ``V_out = norm(KV)`` (shared rrms).
    """
    in_off = m * stride_kin_m + h * HEAD_DIM + cols
    x = tl.load(KV_in_ptr + in_off, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
    v_out = x * rrms
    kw = tl.load(K_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    k_out = v_out * kw
    k_off = m * stride_kout_m + h * HEAD_DIM + cols
    v_off = m * stride_vout_m + h * HEAD_DIM + cols
    tl.store(K_out_ptr + k_off, k_out.to(K_out_ptr.dtype.element_ty), mask=mask)
    tl.store(V_out_ptr + v_off, v_out.to(V_out_ptr.dtype.element_ty), mask=mask)


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
    BY_HEAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused per-head RMSNorm: Q (q_w), K (k_w), V (weight=ones, so the
    multiply-by-weight is omitted).

    Modes (``HAS_KV`` / ``K_EQ_V``):
    * Q-only (``HAS_KV=False``): norm Q in place; K/V pointers unused.
    * QKV (``K_EQ_V=False``): norm Q, K, V in place (``*_in == *_out``).
    * K=V (``K_EQ_V=True``, ``attention_k_eq_v``): norm Q in place; ``K_in``
      is the shared K/V projection -> ``K_out = norm*k_w``, ``V_out = norm``.

    Launch shapes (``BY_HEAD``): ``(M,)`` one program per token (looping
    heads), or ``(M, total_heads)`` one program per (token, head) for better
    occupancy at M <= 256. The launcher sizes ``total_heads`` per mode
    (Q: ``NUM_Q_HEADS``; K=V: ``+ NUM_KV_HEADS``; QKV: ``+ 2 * NUM_KV_HEADS``).

    Each tensor's last dim packs (num_heads, head_dim) contiguously, so the
    per-head offset is ``h * HEAD_DIM`` and the M stride is read from
    ``stride_*_m`` -- strided views (e.g. ``qkv.split`` slices) need no copy.
    """
    m = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < HEAD_DIM

    if BY_HEAD:
        # One program per (token, head).
        h_all = tl.program_id(1)
        if h_all < NUM_Q_HEADS:
            # Q head.
            _gemma_qkv_rmsnorm_store(
                Q_ptr, Q_w_ptr, stride_q_m, m, h_all, cols, mask, HEAD_DIM, eps, True
            )
        elif HAS_KV:
            h = h_all - NUM_Q_HEADS
            if K_EQ_V:
                _gemma_qkv_rmsnorm_keqv_store(
                    K_in_ptr,
                    K_out_ptr,
                    V_out_ptr,
                    K_w_ptr,
                    stride_kin_m,
                    stride_kout_m,
                    stride_vout_m,
                    m,
                    h,
                    cols,
                    mask,
                    HEAD_DIM,
                    eps,
                )
            elif h < NUM_KV_HEADS:
                # K head.
                _gemma_qkv_rmsnorm_store(
                    K_in_ptr,
                    K_w_ptr,
                    stride_kin_m,
                    m,
                    h,
                    cols,
                    mask,
                    HEAD_DIM,
                    eps,
                    True,
                )
            else:
                # V head (no scale).
                hv = h - NUM_KV_HEADS
                _gemma_qkv_rmsnorm_store(
                    V_in_ptr,
                    Q_w_ptr,
                    stride_vin_m,
                    m,
                    hv,
                    cols,
                    mask,
                    HEAD_DIM,
                    eps,
                    False,
                )
    else:
        # One program per token, looping over heads.
        for h in tl.static_range(NUM_Q_HEADS):  # Q heads
            _gemma_qkv_rmsnorm_store(
                Q_ptr, Q_w_ptr, stride_q_m, m, h, cols, mask, HEAD_DIM, eps, True
            )

        if HAS_KV:
            if K_EQ_V:
                for h in tl.static_range(NUM_KV_HEADS):
                    _gemma_qkv_rmsnorm_keqv_store(
                        K_in_ptr,
                        K_out_ptr,
                        V_out_ptr,
                        K_w_ptr,
                        stride_kin_m,
                        stride_kout_m,
                        stride_vout_m,
                        m,
                        h,
                        cols,
                        mask,
                        HEAD_DIM,
                        eps,
                    )
            else:
                for h in tl.static_range(NUM_KV_HEADS):  # K heads
                    _gemma_qkv_rmsnorm_store(
                        K_in_ptr,
                        K_w_ptr,
                        stride_kin_m,
                        m,
                        h,
                        cols,
                        mask,
                        HEAD_DIM,
                        eps,
                        True,
                    )

                for h in tl.static_range(NUM_KV_HEADS):  # V heads (no scale)
                    _gemma_qkv_rmsnorm_store(
                        V_in_ptr,
                        Q_w_ptr,
                        stride_vin_m,
                        m,
                        h,
                        cols,
                        mask,
                        HEAD_DIM,
                        eps,
                        False,
                    )


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
    mode: ProjAndNormMode = ProjAndNormMode.QKV_FULL,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Fused per-head RMSNorm on Q, K, V (or any subset) for Gemma4.

    Each head computes ``x * rsqrt(mean(x^2) + eps)``: Q scaled by q_weight,
    K by k_weight, V by 1 (Gemma4 V-norm uses with_scale=False). Inputs may be
    2D ``(M, num_heads * head_dim)`` or strided ``qkv.split`` slices; the
    actual ``stride(0)`` is used so no ``.contiguous()`` copy is needed (the
    last dim must stay contiguous). The caller picks the layout via ``mode``:

    Q_ONLY    k=None, v=None. Q normalised in place. Returns None.
    QKV_FULL  k and v non-None. Q, K, V normalised in place. Returns
              None.
    QK_ONLY   k is the shared K/V projection, v=None. Q normalised in
              place; fresh K and V tensors are allocated. Returns
              (k_out, v_out).
    """
    assert q.is_cuda or q.is_xpu
    assert q.stride(-1) == 1, "Q's last dim must be contiguous"
    assert q_weight.shape[-1] == head_dim
    M = q.shape[0] if q.dim() >= 2 else 1
    BLOCK = triton.next_power_of_2(head_dim)

    # Resolve the mode + allocate outputs if needed.
    if mode is ProjAndNormMode.QK_ONLY:
        assert (
            k is not None and v is None
        ), "QK_ONLY expects k=<shared KV input>, v=None"
        assert k.is_cuda and k.stride(-1) == 1
        assert k_weight is not None and k_weight.shape[-1] == head_dim
        assert (
            q.shape[0] == k.shape[0]
        ), f"M mismatch: q.shape[0]={q.shape[0]} vs kv.shape[0]={k.shape[0]}"
        has_kv = True
        k_eq_v = True
        k_in = k
        v_in = q  # unused; valid pointer for triton.
        k_out = torch.empty_like(k)
        v_out = torch.empty_like(k)
        stride_kin_m = k.stride(0)
        stride_vin_m = 0
        stride_kout_m = k_out.stride(0)
        stride_vout_m = v_out.stride(0)
    elif mode is ProjAndNormMode.QKV_FULL:
        assert k is not None and v is not None, "QKV_FULL expects non-None k and v"
        assert (k.is_cuda and v.is_cuda) or (k.is_xpu and v.is_xpu)
        assert k.stride(-1) == 1 and v.stride(-1) == 1
        assert k_weight is not None and k_weight.shape[-1] == head_dim
        has_kv = True
        k_eq_v = False
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
        assert mode is ProjAndNormMode.Q_ONLY
        assert k is None and v is None, "Q_ONLY requires both k and v to be None"
        has_kv = False
        k_eq_v = False
        # Unused pointers; pass q for safety.
        k_in = v_in = k_out = v_out = q
        stride_kin_m = stride_vin_m = stride_kout_m = stride_vout_m = 0

    kv_heads = num_kv_heads if has_kv else 0

    # BY_HEAD for M <= 256: one program per (token, head) for better
    # occupancy. K=V uses one KV program per head; QKV uses two (K and V).
    by_head = M <= 256
    if by_head:
        if not has_kv:
            total_heads = num_q_heads
        elif k_eq_v:
            total_heads = num_q_heads + kv_heads
        else:
            total_heads = num_q_heads + 2 * kv_heads
        grid = (M, total_heads)
    else:
        grid = (M,)

    _gemma_qkv_rmsnorm_kernel[grid](
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
        NUM_KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        eps=eps,
        HAS_KV=has_kv,
        K_EQ_V=k_eq_v,
        BY_HEAD=by_head,
        BLOCK=BLOCK,
    )

    if mode is ProjAndNormMode.QK_ONLY:
        return k_out, v_out
    return None


@triton.jit
def _gemma_routing_post_topk_kernel(
    Logits_ptr,
    Ids_ptr,
    Scale_ptr,
    Out_weights_ptr,
    Out_ids_ptr,
    stride_l,
    stride_ow,
    stride_oi,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused: softmax(topk_logits) * per_expert_scale[topk_ids] → float32 weights, int32 ids.

    One program per token. K is the number of top-k experts (e.g. 8).
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K

    logits = tl.load(
        Logits_ptr + row * stride_l + cols, mask=mask, other=float("-inf")
    ).to(tl.float32)
    ids_i64 = tl.load(Ids_ptr + row * stride_l + cols, mask=mask, other=0)

    # Stable softmax
    max_val = tl.max(logits, axis=0)
    exp_val = tl.exp(logits - max_val)
    sum_exp = tl.sum(exp_val, axis=0)
    weights = exp_val / sum_exp

    # Gather per_expert_scale and multiply
    scale = tl.load(Scale_ptr + ids_i64, mask=mask, other=1.0).to(tl.float32)
    weights = weights * scale

    tl.store(Out_weights_ptr + row * stride_ow + cols, weights, mask=mask)
    tl.store(Out_ids_ptr + row * stride_oi + cols, ids_i64.to(tl.int32), mask=mask)


def gemma_routing_post_topk(
    topk_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax + scale-gather + casts for Gemma4 routing.

    Replaces: softmax(topk_logits) * per_expert_scale[topk_ids] → (f32, i32).
    """
    B, K = topk_logits.shape
    BLOCK_K = triton.next_power_of_2(K)
    out_weights = torch.empty((B, K), dtype=torch.float32, device=topk_logits.device)
    out_ids = torch.empty((B, K), dtype=torch.int32, device=topk_logits.device)

    _gemma_routing_post_topk_kernel[(B,)](
        topk_logits,
        topk_ids,
        per_expert_scale,
        out_weights,
        out_ids,
        topk_logits.stride(0),
        out_weights.stride(0),
        out_ids.stride(0),
        K=K,
        BLOCK_K=BLOCK_K,
    )
    return out_weights, out_ids


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
def _gemma4_routing_kernel(
    gating_ptr,  # [T, E] router logits, any float dtype
    per_expert_scale_ptr,  # [E] per-expert scale (any float dtype)
    topk_weights_ptr,  # [T, K] fp32 out
    topk_ids_ptr,  # [T, K] int32 out
    stride_g_t,  # stride of gating in the token dim
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < E

    logits = tl.load(
        gating_ptr + pid * stride_g_t + offs_e,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)

    # Pack (sort_key, expert_id) into one int64 so a single signed-ascending
    # tl.sort yields logits in descending float order. The key bijection is
    # anti-monotone on the float value, and the <<32 shift moves its high bit
    # into the int64 sign bit. Ties break by expert id ascending. Invalid
    # lanes use a max key so they sort last.
    MIN32 = -2147483648
    logit_bits = logits.to(tl.int32, bitcast=True)
    sign = logit_bits >> 31
    key = tl.where(sign == 0, logit_bits ^ -1, logit_bits ^ MIN32)
    key = tl.where(valid, key, 0x7FFFFFFF)
    sk64 = key.to(tl.int64) & 0x00000000FFFFFFFF
    packed = (sk64 << 32) | offs_e.to(tl.int64)

    sorted_p = tl.sort(packed, descending=False)
    all_keys = ((sorted_p >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    all_ids = (sorted_p & 0x00000000FFFFFFFF).to(tl.int32)

    # Invert the key bijection to recover the original logit value.
    sign_k = all_keys >> 31
    all_bits = tl.where(sign_k < 0, all_keys ^ -1, all_keys ^ MIN32)
    all_logits = all_bits.to(tl.float32, bitcast=True)

    # softmax over the top-K logits; max sits at index 0 (sorted descending).
    top_mask = offs_e < K
    max_l = tl.max(tl.where(top_mask, all_logits, -float("inf")), axis=0)
    raw_exp = tl.where(top_mask, tl.exp(all_logits - max_l), 0.0)

    denom = tl.sum(raw_exp, axis=0)
    denom = tl.where(denom > 0.0, denom, 1.0)
    weights = raw_exp / denom

    scales = tl.load(
        per_expert_scale_ptr + all_ids.to(tl.int64),
        mask=top_mask,
        other=1.0,
    ).to(tl.float32)
    weights = weights * scales

    base_off = pid * K + offs_e
    tl.store(topk_weights_ptr + base_off, weights, mask=top_mask)
    tl.store(topk_ids_ptr + base_off, all_ids, mask=top_mask)


def gemma4_fused_routing(
    gating_output: torch.Tensor,
    per_expert_scale: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-launch Gemma4 router.

    Args:
        gating_output: [T, E] router logits in any floating dtype; will be
            cast to fp32 inside the kernel.
        per_expert_scale: [E] per-expert scale, any floating dtype.
        topk: number of experts to keep per token.

    Returns:
        topk_weights: [T, topk] fp32 (matches SGLang TopK contract).
        topk_ids: [T, topk] int32 (matches SGLang TopK contract).
    """
    assert gating_output.dim() == 2, "expected [T, E] router logits"
    assert per_expert_scale.dim() == 1
    assert per_expert_scale.shape[0] == gating_output.shape[1]
    T, E = gating_output.shape
    assert topk <= E, f"topk ({topk}) must be <= E ({E})"
    assert E <= 1024, f"gemma4_fused_routing only supports E<=1024, got E={E}"

    gating_output = gating_output.contiguous()
    per_expert_scale = per_expert_scale.contiguous()

    BLOCK_E = triton.next_power_of_2(E)
    topk_weights = torch.empty(
        (T, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty((T, topk), dtype=torch.int32, device=gating_output.device)

    if T == 0:
        return topk_weights, topk_ids

    _gemma4_routing_kernel[(T,)](
        gating_output,
        per_expert_scale,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        E,
        topk,
        BLOCK_E,
        num_warps=1,
    )
    return topk_weights, topk_ids
