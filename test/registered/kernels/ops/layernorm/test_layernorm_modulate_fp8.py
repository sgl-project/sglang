# SPDX-License-Identifier: Apache-2.0
"""Protect Welford order, BF16 stores and direct FP32/E4M3 rounding."""

import sys

import pytest
import torch

from sglang.kernels.ops.layernorm.layernorm_modulate_fp8 import layernorm_modulate_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() < (8, 9),
    reason="requires CUDA SM89+",
)


@pytest.mark.parametrize(
    "shape", [(1, 4096, 4096), (2, 33, 2240), (2, 17, 4100), (1, 0, 128)]
)
def test_norm_modulate_fp8_exact(shape):
    torch.manual_seed(42)
    b, s, d = shape
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    modulation = torch.randn((b, d * 2), device="cuda", dtype=torch.bfloat16)[:, :d]
    if s:
        x[:, 0].zero_()
        if b > 1:
            modulation[1].fill_(-1)
    hidden = torch.nn.functional.layer_norm(x, (d,), eps=1e-6) * (
        1 + modulation[:, None]
    )
    flat = hidden.reshape(-1, d).float()
    scales = (flat.abs().amax(-1, keepdim=True).double() / 448).float()
    inverse = torch.where(scales == 0, 0, scales.double().reciprocal()).float()
    expected = (flat * inverse).clamp(-448, 448).to(torch.float8_e4m3fn)
    q, actual = layernorm_modulate_fp8(x, modulation)
    assert torch.equal(actual, scales)
    assert torch.equal(q.reshape(-1, d).view(torch.uint8), expected.view(torch.uint8))


def test_norm_modulate_fp8_validation():
    x = torch.zeros((1, 2, 128), device="cuda", dtype=torch.bfloat16)
    m = torch.zeros((1, 128), device="cuda", dtype=torch.bfloat16)
    for a, b, eps in [
        (x.float(), m, 1e-6),
        (x[:, :, ::2], m[:, :64], 1e-6),
        (x, m.cpu(), 1e-6),
        (x, m[:, :64], 1e-6),
        (x, m, 0),
    ]:
        with pytest.raises(ValueError, match="requires contiguous BF16"):
            layernorm_modulate_fp8(a, b, eps)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
