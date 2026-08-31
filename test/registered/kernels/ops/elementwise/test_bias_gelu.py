import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.elementwise.bias_gelu import bias_gelu_tanh
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 128), (2, 7, 512), (1, 4096, 13824)])
def test_bias_gelu_tanh_is_bit_exact(dtype: torch.dtype, shape: tuple[int, ...]):
    torch.manual_seed(0)
    input = torch.randn(shape, device="cuda", dtype=dtype)
    bias = torch.randn(shape[-1], device="cuda", dtype=dtype)
    original_input = input.clone()

    expected = F.gelu(input + bias, approximate="tanh")
    actual = bias_gelu_tanh(input, bias)

    assert actual.shape == input.shape
    assert actual.data_ptr() != input.data_ptr()
    assert torch.equal(input, original_input)
    assert torch.equal(actual, expected)


def test_bias_gelu_tanh_rejects_unsupported_width():
    input = torch.randn(2, 127, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(127, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="hidden_dim"):
        bias_gelu_tanh(input, bias)


def test_bias_gelu_tanh_rejects_unsupported_dtype():
    input = torch.ones(2, 128, device="cuda", dtype=torch.int32)
    bias = torch.ones(128, device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match="does not support"):
        bias_gelu_tanh(input, bias)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
