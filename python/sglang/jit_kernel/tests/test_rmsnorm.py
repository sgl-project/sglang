import itertools

import pytest
import torch
import triton


def sglang_jit_rmsnorm(input: torch.Tensor, weight: torch.Tensor) -> None:
    from sglang.jit_kernel.norm import rmsnorm

    rmsnorm(input, weight, output=input)


def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor) -> None:
    from flashinfer.norm import rmsnorm

    rmsnorm(input, weight, out=input)


BS_LIST = [2**n for n in range(0, 14)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
HIDDEN_SIZE_LIST = [512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
DEVICE = "cuda"
DTYPE = torch.bfloat16


@pytest.mark.parametrize(
    "batch_size,hidden_size", list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST))
)
def test_rmsnorm(batch_size: int, hidden_size: int) -> None:
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)
    input_sglang = input.clone()
    input_flashinfer = input.clone()
    sglang_jit_rmsnorm(input_sglang, weight)
    flashinfer_rmsnorm(input_flashinfer, weight)
    triton.testing.assert_close(input_sglang, input_flashinfer, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
