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
)

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
