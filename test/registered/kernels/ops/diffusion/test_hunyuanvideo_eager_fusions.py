"""HunyuanVideo eager QKV/RoPE and quality-gated QKNorm tests."""

import sys
from unittest.mock import patch

import pytest
import torch

import sglang.kernels.ops.diffusion.hunyuan_qknorm as hunyuan_qknorm
from sglang.kernels.ops.diffusion.hunyuan_qknorm import (
    mark_hunyuan_qknorm_site,
    mount_hunyuan_qknorm,
    unmount_hunyuan_qknorm,
)
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.rotary_embedding.utils import (
    _apply_rotary_emb,
)
from sglang.multimodal_gen.runtime.models.dits.hunyuanvideo import (
    _hunyuan_pack_qkv,
    _hunyuan_qknorm,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("img_tokens,txt_tokens", [(257, 31), (4096, 256)])
def test_hunyuan_qkv_rope_pack_is_bit_exact(img_tokens, txt_tokens):
    torch.manual_seed(0)
    shape_img = (1, img_tokens, 24, 128)
    shape_txt = (1, txt_tokens, 24, 128)
    img_q, img_k, img_v = (
        torch.randn(shape_img, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    txt_q, txt_k, txt_v = (
        torch.randn(shape_txt, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    cos = torch.randn(img_tokens, 64, device="cuda")
    sin = torch.randn_like(cos)

    q, k, v = _hunyuan_pack_qkv(img_q, img_k, img_v, txt_q, txt_k, txt_v, cos, sin)
    q_ref = torch.cat(
        (_apply_rotary_emb(img_q, cos, sin, is_neox_style=False), txt_q), dim=1
    )
    k_ref = torch.cat(
        (_apply_rotary_emb(img_k, cos, sin, is_neox_style=False), txt_k), dim=1
    )
    v_ref = torch.cat((img_v, txt_v), dim=1)

    assert torch.equal(q, q_ref)
    assert torch.equal(k, k_ref)
    assert torch.equal(v, v_ref)


def test_hunyuan_quality_qknorm_matches_rmsnorm():
    torch.manual_seed(1)
    site = torch.nn.Module()
    mark_hunyuan_qknorm_site(site)
    q_norm = RMSNorm(128, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    k_norm = RMSNorm(128, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    packed = torch.randn(1, 257, 3, 24, 128, device="cuda", dtype=torch.bfloat16)
    q, k = packed[:, :, 0], packed[:, :, 1]
    q_ref = q_norm(q.contiguous()).to(q)
    k_ref = k_norm(k.contiguous()).to(k)

    q_unmounted, k_unmounted = _hunyuan_qknorm(site, q, k, q_norm, k_norm)
    assert torch.equal(q_unmounted, q_ref)
    assert torch.equal(k_unmounted, k_ref)

    assert mount_hunyuan_qknorm(site)
    q_out, k_out = _hunyuan_qknorm(site, q, k, q_norm, k_norm)
    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)

    unmount_hunyuan_qknorm(site)
    q_unmounted, k_unmounted = _hunyuan_qknorm(site, q, k, q_norm, k_norm)
    assert torch.equal(q_unmounted, q_ref)
    assert torch.equal(k_unmounted, k_ref)


def test_hunyuan_quality_qknorm_stays_unmounted_without_cute_kernel():
    site = torch.nn.Module()
    mark_hunyuan_qknorm_site(site)

    with patch.object(hunyuan_qknorm, "_get_qk_rmsnorm_cute", return_value=None):
        assert not mount_hunyuan_qknorm(site)

    assert not hunyuan_qknorm._FUSION.is_enabled(site)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
