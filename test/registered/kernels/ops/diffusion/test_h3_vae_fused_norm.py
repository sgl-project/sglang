"""MiniMax-H3 video VAE fused RMSNorm and Q/K RoPE."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    h3_vae_qk_rmsnorm_rope,
    h3_vae_rmsnorm,
    h3_vae_scale_add_rmsnorm,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    _apply_rotary_pos_emb_impl,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@torch.inference_mode()
def test_rmsnorm_matches_pytorch_fp32_accumulation():
    rows, dim = 128, 2048
    hidden = torch.randn(4, rows // 4, dim, device="cuda")
    weight = torch.randn(dim, device="cuda")
    actual = h3_vae_rmsnorm(hidden, weight, 1e-5)
    expected = F.rms_norm(hidden, (dim,), weight, 1e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    half = torch.randn(rows, dim, device="cuda", dtype=torch.float16)
    actual_half = h3_vae_rmsnorm(half, weight, 1e-5)
    expected_half = F.rms_norm(half.float(), (dim,), weight, 1e-5).half()
    torch.testing.assert_close(actual_half, expected_half, rtol=1e-3, atol=1e-2)


@torch.inference_mode()
def test_scale_add_residual_matches_eager_product():
    rows, dim = 64, 256
    residual = torch.randn(2, rows // 2, dim, device="cuda")
    update = torch.randn_like(residual, dtype=torch.float16)
    scale = torch.randn(dim, device="cuda")
    weight = torch.randn(dim, device="cuda")
    original = residual.clone()
    fused_residual, fused_norm = h3_vae_scale_add_rmsnorm(
        residual, update, scale, weight, 1e-5
    )
    expected_residual = original + update.float() * scale
    expected_norm = F.rms_norm(expected_residual, (dim,), weight, 1e-5)
    torch.testing.assert_close(fused_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(fused_norm, expected_norm, rtol=1e-4, atol=1e-5)
    assert fused_residual.data_ptr() == residual.data_ptr()


@torch.inference_mode()
def test_qk_rmsnorm_rope_matches_eager_neox():
    batch, seq_len, heads = 2, 17, 4
    packed = torch.randn(batch, seq_len, heads, 192, device="cuda", dtype=torch.float16)
    query, key, _ = packed.chunk(3, dim=-1)
    half_cos = torch.randn(seq_len, 24, device="cuda", dtype=torch.float16)
    half_sin = torch.randn(seq_len, 24, device="cuda", dtype=torch.float16)
    cos = torch.cat((half_cos, half_cos), dim=-1)
    sin = torch.cat((half_sin, half_sin), dim=-1)
    actual_q, actual_k = h3_vae_qk_rmsnorm_rope(query, key, 1e-5, cos, sin)
    norm = torch.nn.RMSNorm(64, eps=1e-5, elementwise_affine=False).cuda()
    rotary = (
        cos.view(1, seq_len, 1, 48).expand(batch, -1, -1, -1),
        sin.view(1, seq_len, 1, 48).expand(batch, -1, -1, -1),
    )
    with torch.autocast("cuda", enabled=False):
        expected_q = _apply_rotary_pos_emb_impl(norm(query), rotary)
        expected_k = _apply_rotary_pos_emb_impl(norm(key), rotary)
    torch.testing.assert_close(actual_q, expected_q, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-3, atol=1e-2)
    assert h3_vae_qk_rmsnorm_rope(query.float(), key.float(), 1e-5, cos, sin) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
