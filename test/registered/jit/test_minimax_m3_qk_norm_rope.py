# SPDX-License-Identifier: Apache-2.0
"""Reference tests for MiniMax-M3 fused Q/K Gemma RMSNorm + RoPE."""

import pytest
import torch

from sglang.srt.utils import is_hip

if not is_hip():
    pytest.skip(
        "MiniMax-M3 fused Q/K norm + RoPE kernel is ROCm-only.",
        allow_module_level=True,
    )
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from sglang.jit_kernel.minimax_m3.qk_norm_rope import (  # noqa: E402
    qk_gemma_rmsnorm_rope,
    sparse_qk_index_gemma_rmsnorm_rope,
    sparse_qk_index_gemma_rmsnorm_rope_cache,
)
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

# ROCm-only fused kernel; runs in the AMD jit-kernel unit suite.
register_amd_ci(est_time=30, suite="jit-kernel-unit-test-amd")

DEVICE = "cuda"
EPS = 1e-6


def _gemma_norm_by_head(x: torch.Tensor, weight: torch.Tensor, head_dim: int):
    orig_shape = x.shape
    orig_dtype = x.dtype
    xh = x.view(x.shape[0], -1, head_dim).float()
    var = xh.pow(2).mean(dim=-1, keepdim=True)
    out = xh * torch.rsqrt(var + EPS) * (1.0 + weight.float())
    return out.to(orig_dtype).reshape(orig_shape)


def _apply_rope_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool,
):
    orig_shape = x.shape
    xh = x.view(x.shape[0], -1, head_dim)
    x_rot = xh[..., :rotary_dim].float()
    x_pass = xh[..., rotary_dim:]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos[:, None, :].float()
    sin = sin[:, None, :].float()

    if is_neox_style:
        x1, x2 = x_rot.chunk(2, dim=-1)
        y_rot = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1 = x_rot[..., ::2]
        x2 = x_rot[..., 1::2]
        y_rot = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        y_rot = y_rot.flatten(-2)

    return torch.cat((y_rot.to(x.dtype), x_pass), dim=-1).reshape(orig_shape)


def _reference(
    q,
    k,
    q_weight,
    k_weight,
    positions,
    cos_sin_cache,
    head_dim,
    rotary_dim,
    is_neox_style,
):
    q_norm = _gemma_norm_by_head(q, q_weight, head_dim)
    k_norm = _gemma_norm_by_head(k, k_weight, head_dim)
    q_ref = _apply_rope_ref(
        q_norm, positions, cos_sin_cache, head_dim, rotary_dim, is_neox_style
    )
    k_ref = _apply_rope_ref(
        k_norm, positions, cos_sin_cache, head_dim, rotary_dim, is_neox_style
    )
    return q_ref, k_ref


def _sparse_reference(
    q,
    k,
    idx_q,
    idx_k,
    q_weight,
    k_weight,
    idx_q_weight,
    idx_k_weight,
    positions,
    cos_sin_cache,
    head_dim,
    rotary_dim,
    is_neox_style,
):
    q_ref, k_ref = _reference(
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    idx_q_norm = _gemma_norm_by_head(idx_q, idx_q_weight, head_dim)
    idx_k_norm = _gemma_norm_by_head(idx_k, idx_k_weight, head_dim)
    idx_q_ref = _apply_rope_ref(
        idx_q_norm, positions, cos_sin_cache, head_dim, rotary_dim, is_neox_style
    )
    idx_k_ref = _apply_rope_ref(
        idx_k_norm, positions, cos_sin_cache, head_dim, rotary_dim, is_neox_style
    )
    return q_ref, k_ref, idx_q_ref, idx_k_ref


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize(
    "num_tokens,q_heads,k_heads,head_dim,rotary_dim",
    [(1, 16, 1, 128, 64), (17, 16, 1, 128, 64), (64, 4, 1, 128, 64)],
)
@torch.inference_mode()
def test_qk_gemma_rmsnorm_rope_matches_reference(
    dtype, is_neox_style, num_tokens, q_heads, k_heads, head_dim, rotary_dim
):
    torch.manual_seed(0)
    q_dim = q_heads * head_dim
    k_dim = k_heads * head_dim
    padding_dim = 37
    qkv = torch.randn(
        num_tokens, q_dim + k_dim + padding_dim, device=DEVICE, dtype=dtype
    )
    q, k, _ = qkv.split([q_dim, k_dim, padding_dim], dim=-1)
    if num_tokens > 1:
        assert not q.is_contiguous()
        assert not k.is_contiguous()

    q_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    positions = torch.randint(0, 512, (num_tokens,), device=DEVICE, dtype=torch.long)
    cos_sin_cache = torch.randn(512, rotary_dim, device=DEVICE, dtype=dtype)

    got_q, got_k = qk_gemma_rmsnorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        EPS,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    ref_q, ref_k = _reference(
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        head_dim,
        rotary_dim,
        is_neox_style,
    )

    torch.testing.assert_close(got_q, ref_q, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_k, ref_k, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize(
    "num_tokens,q_heads,k_heads,idx_q_heads,head_dim,rotary_dim",
    [(1, 16, 1, 16, 128, 64), (19, 16, 1, 16, 128, 64)],
)
@torch.inference_mode()
def test_sparse_qk_index_gemma_rmsnorm_rope_matches_reference(
    dtype,
    is_neox_style,
    num_tokens,
    q_heads,
    k_heads,
    idx_q_heads,
    head_dim,
    rotary_dim,
):
    torch.manual_seed(1)
    q = torch.randn(num_tokens, q_heads * head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(num_tokens, k_heads * head_dim, device=DEVICE, dtype=dtype)
    idx_q = torch.randn(num_tokens, idx_q_heads * head_dim, device=DEVICE, dtype=dtype)
    idx_k = torch.randn(num_tokens, head_dim, device=DEVICE, dtype=dtype)
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    idx_q_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    idx_k_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    positions = torch.randint(0, 512, (num_tokens,), device=DEVICE, dtype=torch.long)
    cos_sin_cache = torch.randn(512, rotary_dim, device=DEVICE, dtype=dtype)

    got = sparse_qk_index_gemma_rmsnorm_rope(
        q,
        k,
        idx_q,
        idx_k,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        positions,
        cos_sin_cache,
        EPS,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    ref = _sparse_reference(
        q,
        k,
        idx_q,
        idx_k,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        positions,
        cos_sin_cache,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    for got_tensor, ref_tensor in zip(got, ref):
        torch.testing.assert_close(got_tensor, ref_tensor, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_neox_style", [True, False])
@torch.inference_mode()
def test_sparse_qk_index_gemma_rmsnorm_rope_cache_matches_reference(
    dtype, is_neox_style
):
    torch.manual_seed(2)
    num_tokens, q_heads, k_heads, idx_q_heads = 11, 16, 1, 16
    head_dim, rotary_dim = 128, 64
    q = torch.randn(num_tokens, q_heads * head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(num_tokens, k_heads * head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(num_tokens, k_heads * head_dim, device=DEVICE, dtype=dtype)
    idx_q = torch.randn(num_tokens, idx_q_heads * head_dim, device=DEVICE, dtype=dtype)
    idx_k = torch.randn(num_tokens, head_dim, device=DEVICE, dtype=dtype)
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    idx_q_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    idx_k_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    positions = torch.randint(0, 512, (num_tokens,), device=DEVICE, dtype=torch.long)
    cos_sin_cache = torch.randn(512, rotary_dim, device=DEVICE, dtype=dtype)
    out_cache_loc = torch.randperm(64, device=DEVICE, dtype=torch.int64)[:num_tokens]
    k_cache = torch.empty(64, k_heads, head_dim, device=DEVICE, dtype=dtype)
    v_cache = torch.empty(64, k_heads, head_dim, device=DEVICE, dtype=dtype)
    idx_k_cache = torch.empty(64, 1, head_dim, device=DEVICE, dtype=dtype)

    got = sparse_qk_index_gemma_rmsnorm_rope_cache(
        q,
        k,
        v,
        idx_q,
        idx_k,
        k_cache,
        v_cache,
        idx_k_cache,
        out_cache_loc,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        positions,
        cos_sin_cache,
        EPS,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    ref = _sparse_reference(
        q,
        k,
        idx_q,
        idx_k,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        positions,
        cos_sin_cache,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    for got_tensor, ref_tensor in zip(got, ref):
        torch.testing.assert_close(got_tensor, ref_tensor, atol=3e-2, rtol=3e-2)

    torch.testing.assert_close(
        k_cache.index_select(0, out_cache_loc),
        ref[1].view(num_tokens, k_heads, head_dim),
        atol=3e-2,
        rtol=3e-2,
    )
    torch.testing.assert_close(
        v_cache.index_select(0, out_cache_loc),
        v.view(num_tokens, k_heads, head_dim),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        idx_k_cache.index_select(0, out_cache_loc),
        ref[3].view(num_tokens, 1, head_dim),
        atol=3e-2,
        rtol=3e-2,
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
