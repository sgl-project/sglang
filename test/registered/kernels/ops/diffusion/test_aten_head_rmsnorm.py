"""``diffusion.aten_head_rmsnorm``: the bit-exact replica of ROCm aten ``rms_norm``.

ROCm-only: the oracle is ROCm's own ``F.rms_norm`` (and the bf16 NeoX
``rotary_embedding`` that follows it), so these cases run on the AMD lane and
nothing in ``test_norm.py`` does.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="replays ROCm aten rms_norm",
)

DEVICE = "cuda"


@torch.no_grad()
@pytest.mark.parametrize("shape", [(8256, 56), (37, 14), (2, 9, 3)])
@pytest.mark.parametrize("varying_exponent", [False, True])
def test_aten_head_rmsnorm_matches_rocm_rms_norm(shape, varying_exponent):
    from sglang.kernels.ops.diffusion import (
        aten_head_rmsnorm,
        can_use_aten_head_rmsnorm,
    )

    *tokens, heads = shape
    # q of a fused qkv projection: a strided view with packed heads
    qkv = torch.randn(*tokens, 3 * heads * 128, device=DEVICE, dtype=torch.bfloat16)
    if varying_exponent:
        exponents = torch.randint(-12, 12, qkv.shape, device=DEVICE).float()
        qkv = (qkv.float() * torch.exp2(exponents)).to(qkv.dtype)
    q = qkv[..., : heads * 128].unflatten(-1, (heads, 128))
    weight = torch.randn(128, device=DEVICE, dtype=q.dtype)
    assert can_use_aten_head_rmsnorm(q, weight)
    out = aten_head_rmsnorm(q, weight, 1e-6)
    assert out.is_contiguous()
    assert torch.equal(out, F.rms_norm(q, (128,), weight, 1e-6))


@torch.no_grad()
@pytest.mark.parametrize("rot_dim", [96, 128, 64])
def test_aten_head_rmsnorm_rope_matches_rms_norm_then_rotary_embedding(rot_dim):
    from sgl_kernel import rotary_embedding

    from sglang.kernels.ops.diffusion import (
        aten_head_rmsnorm,
        can_use_aten_head_rmsnorm_rope,
    )

    tokens, heads = 1031, 14
    qkv = torch.randn(tokens, 3 * heads * 128, device=DEVICE, dtype=torch.bfloat16)
    q = qkv[:, : heads * 128].view(tokens, heads, 128)
    weight = torch.randn(128, device=DEVICE, dtype=q.dtype)
    angles = torch.rand(2 * tokens, rot_dim // 2, device=DEVICE) * 100
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(q.dtype)
    positions = torch.randperm(2 * tokens, device=DEVICE)[:tokens]
    assert can_use_aten_head_rmsnorm_rope(q, cache, positions)
    expected = F.rms_norm(q, (128,), weight, 1e-5)
    key = expected.clone()
    rotary_embedding(
        positions, expected.view(tokens, -1), key.view(tokens, -1), 128, cache, True
    )
    out = aten_head_rmsnorm(q, weight, 1e-5, cache, positions)
    assert torch.equal(out, expected)


def test_aten_head_rmsnorm_guards():
    from sglang.kernels.ops.diffusion import can_use_aten_head_rmsnorm

    x = torch.ones(4, 3, 128, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.ones(128, device=DEVICE, dtype=x.dtype)
    for inputs, gamma in [
        (x.float(), weight.float()),
        (x, weight.float()),
        (x, None),
        (x.transpose(0, 1), weight),
        (x[..., ::2], weight[::2]),
        (x.flatten(0, 1), weight),
        (x[:0], weight),
        (x.cpu(), weight.cpu()),
    ]:
        assert not can_use_aten_head_rmsnorm(inputs, gamma)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
