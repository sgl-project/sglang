import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    ScaleResidualLayerNormScaleShift,
    null_modulation,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_fp32_layernorm_cache_reuse_and_invalidation():
    norm = FP32LayerNorm(16, eps=1e-5).cuda().to(torch.bfloat16)
    inputs = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        for updated in (False, True):
            if updated:
                previous = norm.__dict__["_weight_fp32_cache"]
                norm.weight.add_(1.0)
            actual = norm(inputs)
            expected = F.layer_norm(
                inputs.float(),
                norm.normalized_shape,
                norm.weight.float(),
                norm.bias.float(),
                norm.eps,
            ).to(inputs.dtype)
            torch.testing.assert_close(actual, expected)
            weight_cache = norm.__dict__["_weight_fp32_cache"]
            bias_cache = norm.__dict__["_bias_fp32_cache"]
            if updated:
                assert weight_cache[0] != previous[0]
                assert weight_cache[1] is not previous[1]
            norm(inputs)
            assert norm.__dict__["_weight_fp32_cache"][1] is weight_cache[1]
            assert norm.__dict__["_bias_fp32_cache"][1] is bias_cache[1]

    assert "_weight_fp32_cache" not in norm.state_dict()
    assert "_bias_fp32_cache" not in norm.state_dict()


def test_fp32_layernorm_grad_mode_preserves_autograd_path():
    norm = FP32LayerNorm(16, eps=1e-5).cuda().to(torch.bfloat16)
    inputs = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output = norm(inputs).float().sum()
    output.backward()

    assert inputs.grad is not None
    assert "_weight_fp32_cache" not in norm.__dict__
    assert "_bias_fp32_cache" not in norm.__dict__


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the ROCm dispatch needs a ROCm GPU"
)
def test_rocm_layernorm_paths_keep_the_eager_bits():
    """The ROCm fast paths skip the fp32 round trip and fuse the gated residual;
    every output must stay bit-identical to forward_native rounded to bf16."""
    torch.manual_seed(0)
    hidden, tokens = 512, 300
    bf16 = torch.bfloat16
    x = torch.randn(1, tokens, hidden, device="cuda", dtype=bf16)
    residual = (torch.randn(1, tokens, hidden, device="cuda") * 3).to(bf16)
    gate, scale, shift = (torch.randn(1, 1, hidden, device="cuda") for _ in range(3))
    null = null_modulation(x.device, bf16)

    norm = FP32LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
    assert torch.equal(norm.forward_hip(x), norm.forward_native(x))

    op = ScaleResidualLayerNormScaleShift(hidden, eps=1e-6, rocm_flydsl_high_only=True)
    for args in ((gate, null, null), (1, shift, scale)):
        fast = op.forward_hip(residual, x, *args)
        op.norm._forward_method = op.norm.forward_native
        eager = op.forward_native(residual, x, *args)
        op.norm._forward_method = op.norm.forward_hip
        for fast_out, eager_out in zip(fast, eager):
            assert torch.equal(fast_out.to(bf16), eager_out.to(bf16))
