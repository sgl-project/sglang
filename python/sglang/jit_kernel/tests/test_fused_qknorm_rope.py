"""
Correctness tests for the fused_qknorm_rope JIT kernel.

Validates fused_qk_norm_rope against a pure-PyTorch reference and (when
available) the sgl_kernel AOT implementation.
"""

import os

import pytest
import torch

from sglang.jit_kernel.fused_qknorm_rope import fused_qk_norm_rope

try:
    from sgl_kernel import fused_qk_norm_rope as fused_qk_norm_rope_aot

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# CI / full-range helpers
# ---------------------------------------------------------------------------

_is_ci = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

HEAD_DIMS_FULL = [64, 128, 256]
HEAD_DIMS_CI = [128]

NUM_TOKENS_FULL = [1, 16, 128]
NUM_TOKENS_CI = [1, 64]

HEAD_DIMS = HEAD_DIMS_CI if _is_ci else HEAD_DIMS_FULL
NUM_TOKENS = NUM_TOKENS_CI if _is_ci else NUM_TOKENS_FULL


# ---------------------------------------------------------------------------
# Pure-PyTorch reference
# ---------------------------------------------------------------------------


def _compute_inv_freq_yarn(base, rotary_dim, factor, low, high, device):
    """Compute YaRN-adjusted inverse frequencies for rotary_dim//2 positions."""
    half_dims = torch.arange(rotary_dim // 2, dtype=torch.float32, device=device)
    inv_freq = base ** (-2.0 * half_dims / rotary_dim)

    if factor != 1.0:
        inv_freq_interp = inv_freq / factor
        inv_freq_extrap = inv_freq
        high_adj = high if abs(high - low) > 1e-6 else high + 0.001
        linear = (half_dims - low) / (high_adj - low)
        ramp = linear.clamp(0.0, 1.0)
        extrap_factor = 1.0 - ramp
        inv_freq = (
            inv_freq_interp * (1 - extrap_factor) + inv_freq_extrap * extrap_factor
        )

    return inv_freq


def fused_qk_norm_rope_ref(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    eps,
    q_weight,
    k_weight,
    base,
    is_neox,
    position_ids,
    factor,
    low,
    high,
    attention_factor,
    rotary_dim,
):
    """
    Pure-PyTorch reference: RMSNorm per head, then RoPE on Q and K.

    Returns a new tensor (same shape as qkv) with the transformation applied.
    """
    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v

    qkv_f = qkv.float()
    qw = q_weight.float()
    kw = k_weight.float()

    # Reshape to [num_tokens, total_heads, head_dim]
    qkv_3d = qkv_f.view(num_tokens, total_heads, head_dim)
    q = qkv_3d[:, :num_heads_q].clone()  # [num_tokens, nq, head_dim]
    k = qkv_3d[:, num_heads_q : num_heads_q + num_heads_k].clone()

    # RMSNorm per head
    def rms_norm_heads(x, w):
        # x: [num_tokens, n_heads, head_dim], w: [head_dim]
        rms = (x**2).mean(-1, keepdim=True)
        return x * torch.rsqrt(rms + eps) * w

    q = rms_norm_heads(q, qw)
    k = rms_norm_heads(k, kw)

    # Compute frequencies
    inv_freq = _compute_inv_freq_yarn(base, rotary_dim, factor, low, high, qkv.device)
    # theta: [num_tokens, rotary_dim//2]
    theta = position_ids.float().unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = torch.cos(theta)  # [num_tokens, rotary_dim//2]
    sin = torch.sin(theta)
    # Broadcast across heads: [num_tokens, 1, rotary_dim//2]
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)

    if not is_neox:
        # Interleave (GPT-J) style: rotate pairs (x[2i], x[2i+1])
        def apply_interleave(x):
            # x: [num_tokens, n_heads, head_dim]
            x_rot = x[:, :, :rotary_dim]  # [num_tokens, n_heads, rotary_dim]
            x_pairs = x_rot.view(num_tokens, -1, rotary_dim // 2, 2)
            x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
            x0_new = x0 * c - x1 * s
            x1_new = x1 * c + x0 * s
            x_rot_new = torch.stack([x0_new, x1_new], dim=-1).view(
                num_tokens, -1, rotary_dim
            )
            result = x.clone()
            result[:, :, :rotary_dim] = x_rot_new * attention_factor
            return result

        q = apply_interleave(q)
        k = apply_interleave(k)
    else:
        # NeoX style: first half × cos − second half × sin (and vice versa)
        def apply_neox(x):
            # x: [num_tokens, n_heads, head_dim]
            x1 = x[:, :, : rotary_dim // 2]
            x2 = x[:, :, rotary_dim // 2 : rotary_dim]
            x1_new = x1 * c - x2 * s
            x2_new = x2 * c + x1 * s
            result = x.clone()
            result[:, :, : rotary_dim // 2] = x1_new * attention_factor
            result[:, :, rotary_dim // 2 : rotary_dim] = x2_new * attention_factor
            return result

        q = apply_neox(q)
        k = apply_neox(k)

    # Write back into a copy of the full QKV
    result_3d = qkv_f.view(num_tokens, total_heads, head_dim).clone()
    result_3d[:, :num_heads_q] = q
    result_3d[:, num_heads_q : num_heads_q + num_heads_k] = k
    return result_3d.view(num_tokens, -1).bfloat16()


# ---------------------------------------------------------------------------
# Tests: correctness vs PyTorch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("is_neox", [False, True])
def test_fused_qknorm_rope_vs_ref(head_dim, num_tokens, is_neox):
    torch.manual_seed(head_dim * num_tokens + int(is_neox))
    device = "cuda"
    num_heads_q, num_heads_k, num_heads_v = 4, 2, 2
    total_heads = num_heads_q + num_heads_k + num_heads_v
    rotary_dim = head_dim  # full rotary

    qkv = torch.randn(
        (num_tokens, total_heads * head_dim), dtype=torch.bfloat16, device=device
    )
    q_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    eps = 1e-5
    base = 10000.0
    factor = 1.0  # no YaRN
    low, high = 1.0, 32.0
    attention_factor = 1.0

    ref = fused_qk_norm_rope_ref(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    qkv_jit = qkv.clone()
    fused_qk_norm_rope(
        qkv_jit,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    assert torch.allclose(qkv_jit.float(), ref.float(), atol=5e-3, rtol=1e-2), (
        f"mismatch: head_dim={head_dim}, num_tokens={num_tokens}, "
        f"is_neox={is_neox}, "
        f"max_err={( qkv_jit.float() - ref.float()).abs().max().item():.4e}"
    )


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("is_neox", [False, True])
def test_fused_qknorm_rope_partial_rotary(head_dim, is_neox):
    """Test with rotary_dim < head_dim: non-rotary elements should be RMSNorm-only."""
    torch.manual_seed(42 + head_dim + int(is_neox))
    device = "cuda"
    num_tokens = 16
    num_heads_q, num_heads_k, num_heads_v = 2, 2, 2
    total_heads = num_heads_q + num_heads_k + num_heads_v
    rotary_dim = head_dim // 2  # half of head_dim

    # NeoX requires half_rotary_lanes to be power of 2.
    # half_rotary_lanes = rotary_dim / (head_dim / 32) / 2 = (head_dim//2) / (head_dim/32) / 2
    # = 16 / 2 = 8 → power of 2, OK for all supported head_dims.

    qkv = torch.randn(
        (num_tokens, total_heads * head_dim), dtype=torch.bfloat16, device=device
    )
    q_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    ref = fused_qk_norm_rope_ref(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        1e-5,
        q_weight,
        k_weight,
        10000.0,
        is_neox,
        position_ids,
        1.0,
        1.0,
        32.0,
        1.0,
        rotary_dim,
    )

    qkv_jit = qkv.clone()
    fused_qk_norm_rope(
        qkv_jit,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        1e-5,
        q_weight,
        k_weight,
        10000.0,
        is_neox,
        position_ids,
        1.0,
        1.0,
        32.0,
        1.0,
        rotary_dim,
    )

    assert torch.allclose(qkv_jit.float(), ref.float(), atol=5e-3, rtol=1e-2), (
        f"partial rotary mismatch: head_dim={head_dim}, is_neox={is_neox}, "
        f"max_err={(qkv_jit.float() - ref.float()).abs().max().item():.4e}"
    )


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_fused_qknorm_rope_yarn_scaling(head_dim):
    """Test with YaRN scaling (factor != 1.0)."""
    torch.manual_seed(99 + head_dim)
    device = "cuda"
    num_tokens = 32
    num_heads_q, num_heads_k, num_heads_v = 2, 2, 2
    total_heads = num_heads_q + num_heads_k + num_heads_v
    rotary_dim = head_dim

    qkv = torch.randn(
        (num_tokens, total_heads * head_dim), dtype=torch.bfloat16, device=device
    )
    q_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    factor = 2.5
    low, high = 4.0, 32.0
    attention_factor = 0.9
    is_neox = False  # test with interleave; NeoX also tested in other tests

    ref = fused_qk_norm_rope_ref(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        1e-5,
        q_weight,
        k_weight,
        500000.0,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    qkv_jit = qkv.clone()
    fused_qk_norm_rope(
        qkv_jit,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        1e-5,
        q_weight,
        k_weight,
        500000.0,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    assert torch.allclose(qkv_jit.float(), ref.float(), atol=5e-3, rtol=1e-2), (
        f"YaRN mismatch: head_dim={head_dim}, "
        f"max_err={(qkv_jit.float() - ref.float()).abs().max().item():.4e}"
    )


def test_fused_qknorm_rope_default_rotary_dim():
    """rotary_dim=None should default to head_dim."""
    device = "cuda"
    num_tokens = 8
    num_heads_q, num_heads_k, num_heads_v = 2, 2, 2
    head_dim = 128
    total_heads = num_heads_q + num_heads_k + num_heads_v

    torch.manual_seed(0)
    qkv1 = torch.randn(
        (num_tokens, total_heads * head_dim), dtype=torch.bfloat16, device=device
    )
    qkv2 = qkv1.clone()
    q_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    position_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    common_kwargs = dict(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        eps=1e-5,
        q_weight=q_weight,
        k_weight=k_weight,
        base=10000.0,
        is_neox=False,
        position_ids=position_ids,
        factor=1.0,
        low=1.0,
        high=32.0,
        attention_factor=1.0,
    )

    fused_qk_norm_rope(qkv1, **common_kwargs, rotary_dim=None)
    fused_qk_norm_rope(qkv2, **common_kwargs, rotary_dim=head_dim)

    assert torch.equal(qkv1, qkv2), "rotary_dim=None must equal rotary_dim=head_dim"


# ---------------------------------------------------------------------------
# Cross-validation against AOT sgl_kernel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not AOT_AVAILABLE, reason="sgl_kernel not available")
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("is_neox", [False, True])
def test_fused_qknorm_rope_vs_aot(head_dim, is_neox):
    torch.manual_seed(head_dim * 7 + int(is_neox))
    device = "cuda"
    num_tokens = 32
    num_heads_q, num_heads_k, num_heads_v = 4, 2, 2
    total_heads = num_heads_q + num_heads_k + num_heads_v

    qkv = torch.randn(
        (num_tokens, total_heads * head_dim), dtype=torch.bfloat16, device=device
    )
    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device).abs() + 0.5
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device).abs() + 0.5
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    common = dict(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        eps=1e-5,
        q_weight=q_weight,
        k_weight=k_weight,
        base=10000.0,
        is_neox=is_neox,
        position_ids=position_ids,
        factor=1.0,
        low=1.0,
        high=32.0,
        attention_factor=1.0,
        rotary_dim=head_dim,
    )

    qkv_jit = qkv.clone()
    fused_qk_norm_rope(qkv_jit, **common)

    qkv_aot = qkv.clone()
    fused_qk_norm_rope_aot(qkv_aot, **common)

    assert torch.allclose(qkv_jit.float(), qkv_aot.float(), atol=1e-2, rtol=1e-2), (
        f"JIT vs AOT mismatch: head_dim={head_dim}, is_neox={is_neox}, "
        f"max_err={(qkv_jit.float() - qkv_aot.float()).abs().max().item():.4e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
