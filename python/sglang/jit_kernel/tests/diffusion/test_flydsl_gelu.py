import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="jit-kernel-unit-test-amd")

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not hasattr(torch.version, "hip") or not torch.version.hip:
        pytest.skip("ROCm/HIP required for FlyDSL kernels")
    torch.manual_seed(42)


SHAPES = [
    (1, 256, 13824),
    (1, 22500, 13824),
    (2, 16, 5120),
    (1, 1024, 5120),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_gelu_tanh_correctness(shape):
    from sglang.jit_kernel.diffusion.flydsl.gelu import flydsl_gelu_tanh

    gelu_ref = torch.nn.GELU(approximate="tanh")
    x = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)

    y_fly = flydsl_gelu_tanh(x)
    y_ref = gelu_ref(x)

    torch.testing.assert_close(y_fly, y_ref, atol=0.1, rtol=5e-2)


def test_gelu_tanh_non_bf16_skips():
    """Non-bf16 input should raise (kernel only supports bf16)."""
    from sglang.jit_kernel.diffusion.flydsl.gelu import flydsl_gelu_tanh

    x_fp32 = torch.randn(2, 16, 5120, device=DEVICE, dtype=torch.float32)
    with pytest.raises(AssertionError, match="Expected bf16"):
        flydsl_gelu_tanh(x_fp32)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
