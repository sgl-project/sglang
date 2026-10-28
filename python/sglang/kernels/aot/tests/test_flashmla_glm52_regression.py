"""Existing cache ABIs must remain independent of GLM-5.2's NVFP4 API."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires SM100/SM103",
)


@torch.inference_mode()
def test_v32_fp8_sparse_decode_reference():
    from sgl_kernel import flashmla_ops  # noqa: F401

    torch.manual_seed(41913)
    q = torch.randn(2, 1, 64, 576, device="cuda", dtype=torch.bfloat16) * 0.125
    cache = torch.empty(4, 64, 1, 656, device="cuda", dtype=torch.uint8)
    latent = (torch.randn(4, 64, 512, device="cuda") * 0.125).to(torch.float8_e4m3fn)
    rope = torch.randn(4, 64, 64, device="cuda", dtype=torch.bfloat16) * 0.125
    cache[..., :512] = latent.view(torch.uint8).unsqueeze(2)
    cache[..., 512:528] = torch.ones(4, 64, 1, 4, device="cuda").view(torch.uint8)
    cache[..., 528:] = rope.view(torch.uint8).unsqueeze(2)
    indices = torch.randint(0, 256, (2, 1, 2048), device="cuda", dtype=torch.int32)
    scale = 1 / math.sqrt(576)
    out, lse, _, _ = torch.ops.sgl_kernel.sparse_decode_fwd.default(
        q, cache, indices, None, None, None, None, None, None, None, 512, scale, "V32"
    )
    kv = torch.cat((latent.to(torch.bfloat16), rope), dim=-1).reshape(256, 576)
    selected = kv[indices[:, 0].long()].float()
    logits = torch.einsum("bhd,bkd->bhk", q[:, 0].float(), selected) * scale
    expected = torch.einsum("bhk,bkd->bhd", logits.softmax(-1), selected[..., :512])
    torch.testing.assert_close(out[:, 0].float(), expected, atol=8e-4, rtol=0.02)
    torch.testing.assert_close(lse[:, :, 0], logits.logsumexp(-1), atol=2e-4, rtol=2e-4)


@torch.inference_mode()
def test_v41_fp4_extra_cache_smoke():
    from sgl_kernel import flashmla_ops  # noqa: F401

    q = torch.randn(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.125
    # V4.1's rows and scales occupy separate page regions, not GLM52's inline row.
    primary = torch.zeros(2, 64, 1, 528, device="cuda", dtype=torch.uint8)
    extra = torch.zeros(2, 64, 1, 288, device="cuda", dtype=torch.uint8)
    indices = torch.randint(0, 128, (1, 1, 64), device="cuda", dtype=torch.int32)
    extra_indices = torch.randint(0, 128, (1, 1, 64), device="cuda", dtype=torch.int32)
    out, lse, _, _ = torch.ops.sgl_kernel.sparse_decode_fwd.default(
        q,
        primary,
        indices,
        None,
        None,
        None,
        None,
        extra,
        extra_indices,
        None,
        512,
        1 / math.sqrt(512),
        "V41",
    )
    assert out.shape == (1, 1, 64, 512)
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
    assert torch.isfinite(lse).all()
