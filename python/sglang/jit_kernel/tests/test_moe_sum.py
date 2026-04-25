import pytest
import torch

from sglang.jit_kernel.moe_sum import moe_sum

M_LIST = [1, 33, 64, 222]
TOPK_LIST = [2, 3, 4, 6]
K_LIST = [128, 511, 1024]
DTYPE_LIST = [torch.float32, torch.float16, torch.bfloat16]


TOLERANCE = {
    torch.float32: (1e-5, 0),
    torch.float16: (2e-2, 0),
    torch.bfloat16: (1e-1, 0),
}


@pytest.mark.parametrize("m", M_LIST)
@pytest.mark.parametrize("topk", TOPK_LIST)
@pytest.mark.parametrize("k", K_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_moe_sum(m: int, topk: int, k: int, dtype: torch.dtype):
    input = torch.randn((m, topk, k), device="cuda", dtype=dtype)
    output = torch.empty((m, k), device="cuda", dtype=dtype)

    expected = input.sum(dim=1)
    moe_sum(input, output)

    atol, rtol = TOLERANCE[dtype]
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
