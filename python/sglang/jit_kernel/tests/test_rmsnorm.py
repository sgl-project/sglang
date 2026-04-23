import itertools
import sys

import pytest
import torch

from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=240, suite="nightly-kernel-1-gpu", nightly=True)


EPS = 1e-6
DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16]


def sglang_jit_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    eps: float = EPS,
) -> None:
    from sglang.jit_kernel.norm import rmsnorm

    rmsnorm(input, weight, out=output, eps=eps)


def flashinfer_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: torch.Tensor,
    eps: float = EPS,
) -> None:
    from flashinfer.norm import rmsnorm

    rmsnorm(input, weight, out=output, eps=eps)


BS_LIST = [2**n for n in range(0, 14)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
BS_LIST = get_ci_test_range(BS_LIST, [1, 9, 256, 4109])
SUPPORTED_HIDDEN_SIZE_LIST = get_ci_test_range(
    [64, 128, 256, 512, *range(1024, 8192 + 1, 1024), 2304, 2560, 12288, 16384],
    [256, 1024, 16384],
)


@pytest.mark.parametrize(
    "batch_size,hidden_size",
    list(itertools.product(BS_LIST, SUPPORTED_HIDDEN_SIZE_LIST)),
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("specify_out", [True, False])
def test_rmsnorm(
    batch_size: int, hidden_size: int, dtype: torch.dtype, specify_out: bool
) -> None:
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)

    input_flashinfer = input.clone()
    output_flashinfer = torch.empty_like(input)
    flashinfer_rmsnorm(input_flashinfer, weight, output=output_flashinfer)

    if specify_out:
        output_sglang = torch.empty_like(input)
        sglang_jit_rmsnorm(input, weight, output=output_sglang)
    else:
        output_sglang = input.clone()
        sglang_jit_rmsnorm(output_sglang, weight, output=output_sglang)

    torch.testing.assert_close(output_sglang, output_flashinfer, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512, 8192, 8704, 16384])
def test_rmsnorm_hidden_size_support(hidden_size: int) -> None:
    from sglang.jit_kernel.norm import _is_supported_rmsnorm_hidden_size

    assert _is_supported_rmsnorm_hidden_size(hidden_size)


@pytest.mark.parametrize(
    ("hidden_size", "expected"),
    [
        (64, "RMSNormWarpKernel"),
        (128, "RMSNormWarpKernel"),
        (256, "RMSNormWarpKernel"),
        (512, "RMSNormKernel"),
        (1536, "RMSNormKernel"),
        (2048, "RMSNormHalfKernel"),
        (2304, "RMSNormKernel"),  # NOTE: not 512 aligned
        (8192, "RMSNormHalfKernel"),
        (8704, "RMSNormHalfKernel"),
        (16384, "RMSNormHalfKernel"),
    ],
)
def test_rmsnorm_kernel_dispatch(hidden_size: int, expected: str) -> None:
    from sglang.jit_kernel.norm import _rmsnorm_kernel_class

    assert _rmsnorm_kernel_class(hidden_size) == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
