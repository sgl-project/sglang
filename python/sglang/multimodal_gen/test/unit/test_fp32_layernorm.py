import builtins
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers import layernorm
from sglang.multimodal_gen.runtime.layers.layernorm import FP32LayerNorm


@pytest.mark.parametrize("attentions_available", [True, False])
def test_fp32_layernorm_npu_dispatch(monkeypatch, attentions_available):
    original_import = builtins.__import__
    attentions_imports = []

    def import_with_optional_attentions(name, *args, **kwargs):
        if name == "attentions":
            attentions_imports.append(name)
            if not attentions_available:
                raise ImportError("attentions is not installed")
            return ModuleType("attentions")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(layernorm, "_is_npu", True)
    monkeypatch.setattr(
        FP32LayerNorm, "dispatch_forward", lambda self: self.forward_npu
    )
    monkeypatch.setattr(builtins, "__import__", import_with_optional_attentions)

    norm = FP32LayerNorm(16)

    assert attentions_imports == ["attentions"]
    assert norm._forward_method == (
        norm.forward_npu if attentions_available else norm.forward_native
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp32_layernorm_cache_matches_reference():
    norm = FP32LayerNorm(16, eps=1e-5).cuda().to(torch.bfloat16)
    inputs = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        actual = norm(inputs)
        expected = F.layer_norm(
            inputs.float(),
            norm.normalized_shape,
            norm.weight.float().to(device=inputs.device),
            norm.bias.float().to(device=inputs.device),
            norm.eps,
        ).to(inputs.dtype)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp32_layernorm_cache_reuses_converted_params():
    norm = FP32LayerNorm(16, eps=1e-5).cuda().to(torch.bfloat16)
    inputs = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        norm(inputs)
        weight_cache = norm.__dict__["_weight_fp32_cache"]
        bias_cache = norm.__dict__["_bias_fp32_cache"]

        norm(inputs)

    assert norm.__dict__["_weight_fp32_cache"][1] is weight_cache[1]
    assert norm.__dict__["_bias_fp32_cache"][1] is bias_cache[1]
    assert "_weight_fp32_cache" not in norm.state_dict()
    assert "_bias_fp32_cache" not in norm.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp32_layernorm_cache_invalidates_on_param_update():
    norm = FP32LayerNorm(16, eps=1e-5).cuda().to(torch.bfloat16)
    inputs = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        norm(inputs)
        first_key, first_weight = norm.__dict__["_weight_fp32_cache"]

        norm.weight.add_(1.0)
        norm(inputs)
        second_key, second_weight = norm.__dict__["_weight_fp32_cache"]

    assert second_key != first_key
    assert second_weight is not first_weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp32_layernorm_grad_mode_preserves_autograd_path():
    norm = FP32LayerNorm(16, eps=1e-5).cuda().to(torch.bfloat16)
    inputs = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output = norm(inputs).float().sum()
    output.backward()

    assert inputs.grad is not None
    assert "_weight_fp32_cache" not in norm.__dict__
    assert "_bias_fp32_cache" not in norm.__dict__
