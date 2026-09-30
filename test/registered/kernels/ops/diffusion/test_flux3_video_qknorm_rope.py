"""Bit-exact FLUX 3 video VAE QKV RMSNorm + rotate-half RoPE."""

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_flux3_video_qknorm_rope,
    flux3_video_qknorm_rope,
)
from sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae import (
    _VIDEO_QKV_GATE,
    Natten3D,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_EPS = 1e-5


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _reference(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q, k, v = qkv.unbind(4)
    q = F.rms_norm(q, (q.shape[-1],), eps=_EPS)
    k = F.rms_norm(k, (k.shape[-1],), eps=_EPS)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k, v


def _tables(t, h, w, device):
    # Same construction as RotaryPositionEmbedding3D, base 256, head dim 64.
    chunk = 16
    axis = 1.0 / (256.0 ** (torch.arange(0, chunk, 2, device=device).float() / chunk))
    inv = torch.stack([axis, axis, axis, torch.zeros(chunk // 2, device=device)])
    grids = torch.meshgrid(
        torch.arange(t, device=device, dtype=torch.float32),
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pos = torch.stack(grids + (torch.zeros_like(grids[0]),), dim=-1)
    freqs = torch.einsum("...a,af->...af", pos, inv.float()).reshape(1, t, h, w, 1, -1)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos().bfloat16().contiguous(), freqs.sin().bfloat16().contiguous()


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 8, 8, 4),
        (1, 1, 17, 20, 32),
        (1, 2, 8, 10, 8),
    ],
)
def test_flux3_video_qknorm_rope_matches_eager(shape):
    b, t, h, w, heads = shape
    torch.manual_seed(0)
    qkv = torch.randn(b, t, h, w, 3, heads, 64, device="cuda", dtype=torch.bfloat16)
    cos, sin = _tables(t, h, w, qkv.device)
    assert can_use_flux3_video_qknorm_rope(qkv, cos, sin)
    got = flux3_video_qknorm_rope(qkv, cos, sin, eps=_EPS)
    ref = _reference(qkv, cos, sin)
    for name, actual, expected in zip(("q", "k", "v"), got, ref):
        assert torch.equal(actual, expected), name


def test_natten_prepare_qkv_matches_eager_and_reuses_rope_cache():
    _VIDEO_QKV_GATE.disabled = False
    _VIDEO_QKV_GATE.verified = False
    torch.manual_seed(1)
    block = (
        Natten3D(256, [5, 5, 5], num_heads=4, causal=True, qk_norm=True)
        .cuda()
        .bfloat16()
        .eval()
    )
    x = torch.randn(1, 1, 8, 8, 256, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        qkv = block.qkv(x).reshape(1, 1, 8, 8, 3, 4, 64)
        eager = block._eager_qkv(qkv, 0)
        fused = block._prepare_qkv(x)
    assert _VIDEO_QKV_GATE.verified
    for actual, expected in zip(fused, eager):
        assert torch.equal(actual, expected)
    cos, sin = block.rope.cos_sin(1, 8, 8, x.dtype, x.device)
    again_cos, again_sin = block.rope.cos_sin(1, 8, 8, x.dtype, x.device)
    assert cos.data_ptr() == again_cos.data_ptr()
    assert sin.data_ptr() == again_sin.data_ptr()
