import itertools
import sys

import pytest
import torch

from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


def sglang_jit_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    cast_x_before_out_mul: bool = False,
) -> None:
    from sglang.jit_kernel.norm import fused_add_rmsnorm

    fused_add_rmsnorm(
        input, residual, weight, eps, cast_x_before_out_mul=cast_x_before_out_mul
    )


def flashinfer_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    from flashinfer.norm import fused_add_rmsnorm

    fused_add_rmsnorm(input, residual, weight, eps=eps)


def forward_native_hf_reference(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    sum_fp32 = x.to(torch.float32) + residual.to(torch.float32)
    residual_out = sum_fp32.to(x.dtype)
    variance = sum_fp32.pow(2).mean(-1, keepdim=True)
    out = w * (sum_fp32 * torch.rsqrt(variance + eps)).to(x.dtype)
    return out, residual_out


BS_LIST = [2**n for n in range(0, 14)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
BS_LIST = get_ci_test_range(BS_LIST, [1, 9, 256, 4109])
HIDDEN_SIZE_LIST = get_ci_test_range(
    [512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192],
    [512, 2048, 8192],
)
DEVICE = "cuda"
DTYPE = torch.bfloat16
EPS = torch.finfo(torch.bfloat16).eps


@pytest.mark.parametrize(
    "batch_size,hidden_size,cast_x_before_out_mul",
    list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST, [False, True])),
)
def test_fused_add_rmsnorm(
    batch_size: int, hidden_size: int, cast_x_before_out_mul: bool
) -> None:
    torch.manual_seed(0)
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
    residual = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)

    input_sglang = input.clone()
    residual_sglang = residual.clone()
    sglang_jit_fused_add_rmsnorm(
        input_sglang,
        residual_sglang,
        weight,
        EPS,
        cast_x_before_out_mul=cast_x_before_out_mul,
    )

    if cast_x_before_out_mul:
        out_ref, residual_ref = forward_native_hf_reference(
            input, residual, weight, EPS
        )
    else:
        input_ref = input.clone()
        residual_ref_buf = residual.clone()
        flashinfer_fused_add_rmsnorm(input_ref, residual_ref_buf, weight, EPS)
        out_ref, residual_ref = input_ref, residual_ref_buf

    torch.testing.assert_close(input_sglang, out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(residual_sglang, residual_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
