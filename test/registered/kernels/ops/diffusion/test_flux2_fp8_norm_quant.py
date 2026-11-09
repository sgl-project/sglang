import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    fused_layernorm_modulate_fp8_quant_raw,
    fused_layernorm_modulate_raw,
)
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 6144
EPS = 1e-6


def _make_inputs(rows: int):
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260831 + rows)
    x = torch.randn((1, rows, HIDDEN), dtype=DTYPE, device=DEVICE, generator=generator)
    scale = torch.randn((1, 1, HIDDEN), dtype=DTYPE, device=DEVICE, generator=generator)
    shift = torch.randn((1, 1, HIDDEN), dtype=DTYPE, device=DEVICE, generator=generator)
    return x, scale, shift


@pytest.mark.parametrize("rows", [1, 127, 512])
@pytest.mark.parametrize(
    "input_scale_value", [0.005, 0.03125, 0.25, 0.4754464328289032, 1.0]
)
def test_flux2_layernorm_modulate_fp8_is_bit_exact(
    rows: int, input_scale_value: float
) -> None:
    x, scale, shift = _make_inputs(rows)
    input_scale = torch.tensor(input_scale_value, dtype=torch.float32, device=DEVICE)

    normalized = fused_layernorm_modulate_raw(
        x, scale.squeeze(1), shift.squeeze(1), EPS
    )
    expected, _ = static_quant_fp8(normalized, input_scale)
    actual = fused_layernorm_modulate_fp8_quant_raw(
        x, scale.squeeze(1), shift.squeeze(1), input_scale, EPS
    )

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
