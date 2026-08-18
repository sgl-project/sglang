"""LTX-2 quality=high RMSNorm+modulate fusion: gated, close to eager."""

import sys

import pytest
import torch
from torch import nn

import sglang.multimodal_gen.runtime.models.dits.ltx_2 as ltx2_module
from sglang.kernels.ops.diffusion.ltx2_rmsnorm_modulate import (
    fused_ltx2_rms_norm_modulate,
    mark_ltx2_rms_norm_modulate_site,
    mount_ltx2_rms_norm_modulate,
    unmount_ltx2_rms_norm_modulate,
)
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNormNoWeight
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import _ltx2_rms_norm_modulate
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=8, suite="nightly-amd-kernel-1-gpu", nightly=True)


@pytest.fixture(autouse=True)
def _setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


def _eager(rms, x, scale, shift, eps):
    return rms(x, eps) * (1 + scale) + shift


def _inputs(hidden, batch=1, seq=4096):
    rms = RMSNormNoWeight()
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    shift = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    return rms, x, scale, shift


# hidden 4096 = LTX-2 video stream, 2048 = audio stream.
@pytest.mark.parametrize("hidden", [4096, 2048])
def test_lossless_default_is_bitexact(hidden):
    # A marked-but-unmounted site uses only the self-verified bit-exact
    # modulate fast path after the reference aten RMSNorm.
    block = nn.Module()
    mark_ltx2_rms_norm_modulate_site(block)
    rms, x, scale, shift = _inputs(hidden)
    out = _ltx2_rms_norm_modulate(block, rms, x, scale, shift, 1e-6)
    assert torch.equal(out, _eager(rms, x, scale, shift, 1e-6))


def test_lossless_compile_keeps_expression_visible_to_inductor(monkeypatch):
    block = nn.Module()
    mark_ltx2_rms_norm_modulate_site(block)
    rms, x, scale, shift = _inputs(2048, seq=126)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        ltx2_module,
        "_ltx2_modulate",
        lambda *_args: pytest.fail("compiled path must not call the opaque custom op"),
    )

    out = _ltx2_rms_norm_modulate(block, rms, x, scale, shift, 1e-6)
    assert torch.equal(out, _eager(rms, x, scale, shift, 1e-6))


@pytest.mark.parametrize("hidden", [4096, 2048])
def test_mounted_high_uses_fused_kernel(hidden):
    block = nn.Module()
    mark_ltx2_rms_norm_modulate_site(block)
    assert mount_ltx2_rms_norm_modulate(block)
    try:
        rms, x, scale, shift = _inputs(hidden)
        out = _ltx2_rms_norm_modulate(block, rms, x, scale, shift, 1e-6)
        # The mounted path routes through the fused kernel exactly.
        assert torch.equal(out, fused_ltx2_rms_norm_modulate(x, scale, shift, 1e-6))
        # And stays within half-precision rounding of the eager reference.
        ref = _eager(rms, x, scale, shift, 1e-6)
        assert torch.allclose(out.float(), ref.float(), atol=3e-2, rtol=1e-2)
    finally:
        unmount_ltx2_rms_norm_modulate(block)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
