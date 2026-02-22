import itertools

import pytest
import torch


def sglang_jit_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    from sglang.jit_kernel.norm import fused_add_rmsnorm

    fused_add_rmsnorm(input, residual, weight, eps)


def sglang_aot_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    import sgl_kernel

    sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps)


def flashinfer_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    from flashinfer.norm import fused_add_rmsnorm

    fused_add_rmsnorm(input, residual, weight, eps=eps)


BS_LIST = [2**n for n in range(0, 14)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
HIDDEN_SIZE_LIST = [512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
DEVICE = "cuda"
DTYPE = torch.bfloat16


@pytest.mark.parametrize(
    "batch_size,hidden_size", list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST))
)
def test_fused_add_rmsnorm(batch_size: int, hidden_size: int) -> None:
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
    residual = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)

    eps = torch.finfo(torch.bfloat16).eps

    input_aot = input.clone()
    residual_aot = residual.clone()
    input_jit = input.clone()
    residual_jit = residual.clone()
    input_flashinfer = input.clone()
    residual_flashinfer = residual.clone()

    sglang_aot_fused_add_rmsnorm(input_aot, residual_aot, weight, eps)
    sglang_jit_fused_add_rmsnorm(input_jit, residual_jit, weight, eps)
    flashinfer_fused_add_rmsnorm(input_flashinfer, residual_flashinfer, weight, eps)

    torch.testing.assert_close(input_jit, input_aot, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(residual_jit, residual_aot, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(input_jit, input_flashinfer, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(residual_jit, residual_flashinfer, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
