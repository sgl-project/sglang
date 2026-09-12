import sys

import pytest
import torch

from sglang.kernels.ops.gemm.fp8_pertensor_gemm import fp8_pertensor_scaled_mm
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b",
    runner_config="1-gpu-small",
)


def baseline_scaled_mm(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return ((a.float() * scale_a) @ (b_nk.float().t() * scale_b)).to(out_dtype)


def _test_accuracy_once(M, N, K, device):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    b_nk_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    # Distinct scales: the epilogue reads two pointers, so equal values would
    # hide a kernel that applied one of them twice.
    scale_a = torch.tensor([0.001], device=device, dtype=torch.float32)
    scale_b = torch.tensor([0.002], device=device, dtype=torch.float32)
    o = baseline_scaled_mm(a_fp8, b_nk_fp8, scale_a, scale_b, torch.bfloat16)
    o1 = fp8_pertensor_scaled_mm(a_fp8, b_nk_fp8, scale_a, scale_b)
    rtol = 0.02
    atol = 1
    torch.testing.assert_close(o, o1, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not is_sm120_supported(), reason="fp8_pertensor_scaled_mm requires SM120 (>= 12.0)"
)
# M=16 and M=32 straddle the M=24 tile split, so both compiled tiles are covered.
@pytest.mark.parametrize("M", [16, 32])
@pytest.mark.parametrize("N,K", [(16384, 5120), (5120, 6144), (14336, 5120)])
def test_accuracy(M, N, K):
    _test_accuracy_once(M, N, K, "cuda")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
