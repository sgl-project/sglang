import itertools

import pytest
import torch

from sglang.jit_kernel.moe_sum_reduce import moe_sum_reduce


M_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]
TOPK_LIST = [2, 3, 4, 8, 9]
K_LIST = [512, 1024, 2048, 4096, 7168]
DTYPE_LIST = [torch.float16, torch.bfloat16, torch.float32]
SCALE_LIST = [0.5, 1.0, 2.0, 0.125]

configs = list(itertools.product(M_LIST, TOPK_LIST, K_LIST, DTYPE_LIST, SCALE_LIST))


@pytest.mark.parametrize("m, topk, k, dtype, scale", configs)
def test_moe_sum_reduce(m, topk, k, dtype, scale):
    input_tensor = torch.randn(m, topk, k, dtype=dtype, device="cuda")
    output = torch.empty(m, k, dtype=dtype, device="cuda")

    moe_sum_reduce(input_tensor, output, scale)

    expected = torch.sum(input_tensor.float(), dim=1) * scale
    expected = expected.to(dtype)

    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("topk", [5, 6, 7, 10])
def test_moe_sum_reduce_fallback(topk):
    m, k = 16, 1024
    dtype = torch.float16
    scale = 2.0
    input_tensor = torch.randn(m, topk, k, dtype=dtype, device="cuda")
    output = torch.empty(m, k, dtype=dtype, device="cuda")

    moe_sum_reduce(input_tensor, output, scale)

    expected = torch.sum(input_tensor.float(), dim=1) * scale
    expected = expected.to(dtype)

    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
