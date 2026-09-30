import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.layernorm import FP32LayerNorm

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
