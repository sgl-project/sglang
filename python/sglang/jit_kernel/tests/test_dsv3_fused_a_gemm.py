import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.dsv3_fused_a_gemm import (
    dsv3_fused_a_gemm as jit_dsv3_fused_a_gemm,
)

try:
    from sgl_kernel import dsv3_fused_a_gemm as aot_dsv3_fused_a_gemm

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

HdIn = 7168
HdOut = 2112


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="requires CUDA SM90+",
)
@pytest.mark.parametrize("num_tokens", [i + 1 for i in range(16)])
def test_jit_vs_torch_matmul(num_tokens):
    torch.manual_seed(42)
    mat_a = torch.randn(num_tokens, HdIn, dtype=torch.bfloat16, device="cuda")
    mat_b = (
        torch.randn(HdIn, HdOut, dtype=torch.bfloat16, device="cuda")
        .t()
        .contiguous()
        .t()
    )

    jit_output = jit_dsv3_fused_a_gemm(mat_a, mat_b)

    ref_output = F.linear(mat_a, mat_b.T)

    torch.testing.assert_close(jit_output, ref_output, rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(
    not AOT_AVAILABLE
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9,
    reason="requires sgl_kernel AOT + CUDA SM90+",
)
@pytest.mark.parametrize("num_tokens", [i + 1 for i in range(16)])
def test_jit_vs_aot(num_tokens):
    torch.manual_seed(42)
    mat_a = torch.randn(num_tokens, HdIn, dtype=torch.bfloat16, device="cuda")
    # mat_b must be column-major: shape [HdIn, HdOut] with stride(0)==1
    mat_b = (
        torch.randn(HdIn, HdOut, dtype=torch.bfloat16, device="cuda")
        .t()
        .contiguous()
        .t()
    )

    jit_output = jit_dsv3_fused_a_gemm(mat_a, mat_b)
    aot_output = aot_dsv3_fused_a_gemm(mat_a, mat_b)

    torch.testing.assert_close(jit_output, aot_output, rtol=0, atol=0)


if __name__ == "__main__":
    import subprocess

    subprocess.call(["pytest", "--tb=short", str(__file__)])
