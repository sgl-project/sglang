import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    fused_norm_scale_shift_fp8,
    fused_scale_residual_norm_scale_shift_fp8,
)
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    ScaleResidualLayerNormScaleShift,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 3072
EPS = 1e-6


def _make_inputs(rows: int):
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260831 + rows)
    x = torch.randn((1, rows, HIDDEN), dtype=DTYPE, device=DEVICE, generator=generator)
    residual = torch.randn_like(x)
    gate = torch.randn((HIDDEN,), dtype=DTYPE, device=DEVICE, generator=generator)
    scale = torch.randn((HIDDEN,), dtype=DTYPE, device=DEVICE, generator=generator)
    shift = torch.randn((HIDDEN,), dtype=DTYPE, device=DEVICE, generator=generator)
    return x, residual, gate, scale, shift


@pytest.mark.parametrize("rows", [1, 127, 1024])
@pytest.mark.parametrize(
    "input_scale_value", [0.005, 0.03125, 0.4263392984867096, 0.4754464328289032, 1.0]
)
def test_norm_scale_shift_fp8_is_bit_exact(rows: int, input_scale_value: float) -> None:
    x, _, _, scale, shift = _make_inputs(rows)
    input_scale = torch.tensor(input_scale_value, dtype=torch.float32, device=DEVICE)
    layer = LayerNormScaleShift(
        HIDDEN, eps=EPS, elementwise_affine=False, dtype=DTYPE
    ).to(DEVICE)

    normalized = layer.forward_cuda(x, shift, scale)
    expected, _ = static_quant_fp8(normalized, input_scale)
    actual_normalized, actual = fused_norm_scale_shift_fp8(
        x, scale, shift, input_scale, EPS
    )

    assert torch.equal(actual_normalized, normalized)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("rows", [1, 127, 1024])
@pytest.mark.parametrize(
    "input_scale_value", [0.005, 0.03125, 0.4263392984867096, 0.4754464328289032, 1.0]
)
def test_residual_norm_scale_shift_fp8_is_bit_exact(
    rows: int, input_scale_value: float
) -> None:
    x, residual, gate, scale, shift = _make_inputs(rows)
    input_scale = torch.tensor(input_scale_value, dtype=torch.float32, device=DEVICE)
    layer = ScaleResidualLayerNormScaleShift(
        HIDDEN, eps=EPS, elementwise_affine=False, dtype=DTYPE
    ).to(DEVICE)

    normalized, expected_residual = layer.forward_cuda(residual, x, gate, shift, scale)
    expected, _ = static_quant_fp8(normalized, input_scale)
    actual_normalized, actual, actual_residual = (
        fused_scale_residual_norm_scale_shift_fp8(
            residual, x, gate, scale, shift, input_scale, EPS
        )
    )

    assert torch.equal(actual_normalized, normalized)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(actual_residual, expected_residual)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
