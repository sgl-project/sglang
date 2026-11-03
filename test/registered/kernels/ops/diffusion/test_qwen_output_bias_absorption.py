import sys
from unittest.mock import patch

import pytest
import torch

import sglang.multimodal_gen.runtime.models.dits.qwen_image as qwen_image
from sglang.kernels.ops.diffusion import (
    try_fused_bias_mul_add,
    try_fused_bias_scale_residual_norm_scale_shift,
)
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    ScaleResidualLayerNormScaleShift,
)
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="Qwen-Image output-bias fusion is validated on SM103",
)


@pytest.fixture(autouse=True)
def _seed_cuda():
    torch.cuda.manual_seed(0)


@pytest.mark.parametrize("quant_name", ["modelopt_fp8", "modelopt_fp4"])
@pytest.mark.parametrize(
    "capability,expected",
    [
        (DeviceCapability(10, 0), False),
        (DeviceCapability(10, 3), True),
        (DeviceCapability(12, 0), False),
        (None, False),
    ],
)
def test_qwen_output_bias_absorption_is_sm103_only(quant_name, capability, expected):
    class _QuantConfig:
        def get_name(self):
            return quant_name

    assert (
        qwen_image._can_defer_modelopt_output_bias(_QuantConfig(), capability)
        is expected
    )


@requires_sm103
def test_qwen_output_bias_absorption_is_bit_exact():
    hidden = 3072
    shape = (1, 64, hidden)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    bias = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(1, 1, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn_like(gate)
    shift = torch.randn_like(gate)
    norm = ScaleResidualLayerNormScaleShift(
        hidden, eps=1e-6, elementwise_affine=False
    ).cuda()

    expected_norm, expected_residual = norm(
        residual=residual,
        x=x + bias,
        gate=gate,
        scale=scale,
        shift=shift,
    )
    actual_norm, actual_residual = try_fused_bias_scale_residual_norm_scale_shift(
        residual,
        x,
        bias,
        gate,
        None,
        None,
        scale,
        shift,
        "layer",
        1e-6,
    )
    expected_final = MulAdd().cuda()(x + bias, gate, residual)
    actual_final = try_fused_bias_mul_add(x, bias, gate, residual)

    assert torch.equal(actual_norm, expected_norm)
    assert torch.equal(actual_residual, expected_residual)
    assert torch.equal(actual_final, expected_final)

    # A small residual must still break a BF16 product-rounding tie. Casting
    # through FP32 before the final BF16 store loses this information at large
    # magnitudes, so keep this case as a guard for the native BF16 FMA.
    x.fill_(-24576.0)
    bias.fill_(-0.01055908203125)
    gate.fill_(206.0)
    residual.fill_(-0.2080078125)
    expected_tie = MulAdd().cuda()(x + bias, gate, residual)
    actual_tie = try_fused_bias_mul_add(x, bias, gate, residual)
    assert torch.equal(actual_tie, expected_tie)


def test_qwen_output_bias_absorption_rejects_unsupported_inputs():
    hidden = 3072
    x = torch.randn(2, 17, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    row = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    assert try_fused_bias_mul_add(x, row, row, residual) is None

    x = x[:1].contiguous()
    residual = residual[:1].contiguous()
    if torch.cuda.get_device_capability() != (10, 3):
        assert try_fused_bias_mul_add(x, row, row, residual) is None
    with patch("torch.compiler.is_compiling", return_value=True):
        assert try_fused_bias_mul_add(x, row, row, residual) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
