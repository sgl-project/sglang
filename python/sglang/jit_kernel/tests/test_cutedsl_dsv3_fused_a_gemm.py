"""Tests for the CuTe DSL DeepSeek-V3 fused-A GEMM kernel."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

try:
    import cutlass  # noqa: F401
    from cutlass.cute.runtime import from_dlpack  # noqa: F401

    from sglang.jit_kernel import cutedsl_dsv3_fused_a_gemm as kmod

    CUTEDSL_AVAILABLE = True
except ImportError:
    CUTEDSL_AVAILABLE = False
    kmod = None

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

GEMM_M = 2112  # output features
GEMM_K = 7168  # hidden dim

_SM90_PLUS = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
requires = pytest.mark.skipif(
    not (CUTEDSL_AVAILABLE and _SM90_PLUS),
    reason="needs CuTe DSL and SM90+ (PDL / mbarrier / opt-in smem)",
)


def _make_inputs(M):
    weight = torch.randn(GEMM_M, GEMM_K, dtype=torch.bfloat16, device="cuda")
    mat_a = torch.randn(M, GEMM_K, dtype=torch.bfloat16, device="cuda")
    mat_b = weight.t()  # [GEMM_K, GEMM_M] column-major (stride(0) == 1)
    return mat_a, mat_b, weight


@requires
@pytest.mark.parametrize("M", [1, 2, 4, 7, 8, 13, 16])
def test_correctness(M):
    torch.manual_seed(M)
    mat_a, mat_b, weight = _make_inputs(M)
    out = kmod.dsv3_fused_a_gemm(mat_a, mat_b)
    assert out.shape == (M, GEMM_M)
    assert out.dtype == torch.bfloat16

    ref = (mat_a.float() @ weight.float().T).bfloat16()
    # bf16 accumulation over K=7168 -> compare against bf16-rounded reference.
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2.5)
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.999, f"cos={cos}"


@requires
def test_out_param():
    mat_a, mat_b, weight = _make_inputs(8)
    out = torch.empty(8, GEMM_M, dtype=torch.bfloat16, device="cuda")
    result = kmod.dsv3_fused_a_gemm(mat_a, mat_b, out=out)
    assert result is out
    torch.testing.assert_close(out, (mat_a.float() @ weight.float().T).bfloat16(), rtol=2e-2, atol=2.5)


@requires
def test_rejects_bad_layout():
    mat_a, mat_b, _ = _make_inputs(4)
    with pytest.raises(AssertionError):
        kmod.dsv3_fused_a_gemm(mat_a, mat_b.contiguous())  # row-major weight -> rejected
    with pytest.raises(AssertionError):
        kmod.dsv3_fused_a_gemm(torch.randn(17, GEMM_K, dtype=torch.bfloat16, device="cuda"), mat_b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
