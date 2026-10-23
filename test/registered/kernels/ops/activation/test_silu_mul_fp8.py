# SPDX-License-Identifier: Apache-2.0
"""Guard BF16 stores and direct FP32 -> E4M3 rounding in fused SwiGLU."""

import sys

import pytest
import torch

from sglang.kernels.ops.activation.silu_mul_fp8 import silu_mul_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() < (8, 9),
    reason="direct FP32 to E4M3 conversion needs CUDA SM89+",
)


@pytest.mark.parametrize("shape", [(33, 128), (4096, 12288), (65, 12300)])
def test_silu_mul_fp8_preserves_rounding_and_scales(shape):
    # Replacing the direct FP32 cast with Triton's default cast fails on SM89:
    # an intermediate FP16 round moves values onto E4M3 halfway points.
    torch.manual_seed(42)
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    gate[0].zero_()
    gate[1] *= 16
    up[2] *= 32
    hidden = torch.nn.functional.silu(gate) * up
    # The FP64 scalar divisions express correctly-rounded FP32 operations,
    # independent of the reference quantizer's architecture-specific fast math.
    scales = (hidden.float().abs().amax(dim=-1, keepdim=True).double() / 448).float()
    inverse = torch.where(scales == 0, 0, scales.double().reciprocal()).float()
    expected = (hidden.float() * inverse).clamp(-448, 448).to(torch.float8_e4m3fn)
    quantized, actual_scales = silu_mul_fp8(gate, up)
    assert torch.equal(actual_scales, scales)
    assert torch.equal(quantized.view(torch.uint8), expected.view(torch.uint8))
    assert quantized.is_contiguous()


def test_silu_mul_fp8_rejects_unsupported_layouts():
    gate = torch.zeros((2, 16), device="cuda", dtype=torch.bfloat16)
    for first, second in (
        (gate.float(), gate),
        (gate, gate[:, :8]),
        (gate[:, ::2], gate[:, ::2]),
        (gate[:, :0], gate[:, :0]),
        (gate, gate.cpu()),
    ):
        with pytest.raises(ValueError, match="equal contiguous BF16"):
            silu_mul_fp8(first, second)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
