"""FLUX.1 fused LN+modulate fast path must stay bit-exact vs eager."""

import pytest
import torch

import sglang.multimodal_gen.runtime.models.dits.flux as flux
from sglang.kernels.ops.diffusion.fused_ln_modulate import (
    mark_fused_ln_modulate_site,
    mount_fused_ln_modulate,
)
from sglang.multimodal_gen.runtime.models.dits.flux import (
    _flux_fused_ln_modulate,
    _flux_norm_modulate,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _eager(norm, x, scale, shift):
    return norm(x) * (1 + scale[:, None]) + shift[:, None]


def _make_site_inputs(shape, chunks, seed):
    torch.manual_seed(seed)
    batch, seq, hidden = shape
    norm = torch.nn.LayerNorm(hidden, eps=1e-6, elementwise_affine=False).cuda()
    x = (torch.randn(batch, seq, hidden, device="cuda") * 8).bfloat16()
    emb = torch.randn(batch, chunks * hidden, device="cuda").bfloat16()
    parts = emb.chunk(chunks, dim=1)  # strided adaLN projection views
    return norm, x, parts[0], parts[1]


@pytest.mark.parametrize(
    "shape,chunks",
    [
        ((1, 4096, 3072), 6),  # dual-stream image tokens (1024^2), chunk(6)
        ((1, 512, 3072), 6),  # dual-stream text tokens
        ((1, 4608, 3072), 3),  # single-stream concat, chunk(3)
        ((2, 300, 3072), 6),  # CFG batch, odd seq
    ],
)
def test_flux_fused_ln_modulate_is_bit_exact(shape, chunks):
    # Every distinct (shape, stride, eps) signature the FLUX.1 sites emit
    # must verify torch.equal on first sight and stay enabled.
    norm, x, shift, scale = _make_site_inputs(shape, chunks, seed=0)
    out = _flux_fused_ln_modulate(norm, x, scale, shift)
    assert out is not None
    assert torch.equal(out, _eager(norm, x, scale, shift))
    assert not flux._FLUX_LN_MOD.disabled
    assert flux._FLUX_LN_MOD.verified


def test_flux_norm_modulate_bitexact_supersedes_high_fold():
    # With the quality="high" affine fold mounted, the bit-exact kernel
    # still takes priority, so the site output stays lossless.
    site = torch.nn.Module()
    mark_fused_ln_modulate_site(site)
    assert mount_fused_ln_modulate(site)
    norm, x, shift, scale = _make_site_inputs((1, 128, 3072), 6, seed=1)
    out = _flux_norm_modulate(site, norm, x, scale, shift)
    assert torch.equal(out, _eager(norm, x, scale, shift))


def test_flux_fused_ln_modulate_rejects_unsupported_hidden():
    # hidden % 4 != 0 is outside the kernel contract and must bail out.
    norm, x, shift, scale = _make_site_inputs((1, 64, 3070), 6, seed=2)
    assert _flux_fused_ln_modulate(norm, x, scale, shift) is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
