import pytest
import torch

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


def _make_inputs(num_tokens):
    """Create test inputs: mat_a (row-major bf16), mat_b (column-major bf16)."""
    torch.manual_seed(42)
    mat_a = torch.randn(num_tokens, HdIn, dtype=torch.bfloat16, device="cuda")
    # mat_b must be column-major: shape [HdIn, HdOut] with stride(0)==1
    mat_b = (
        torch.randn(HdIn, HdOut, dtype=torch.bfloat16, device="cuda")
        .t()
        .contiguous()
        .t()
    )
    return mat_a, mat_b


@pytest.mark.parametrize("num_tokens", [i + 1 for i in range(16)])
def test_jit_vs_torch_matmul(num_tokens):
    """JIT kernel should produce approximately correct results vs torch.matmul."""
    mat_a, mat_b = _make_inputs(num_tokens)

    jit_output = jit_dsv3_fused_a_gemm(mat_a, mat_b)

    ref_output = (mat_a.float() @ mat_b.float()).bfloat16()

    cos_sim = torch.nn.functional.cosine_similarity(
        jit_output.float().flatten(), ref_output.float().flatten(), dim=0
    )
    assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim:.6f}"


@pytest.mark.skipif(not AOT_AVAILABLE, reason="sgl_kernel AOT not available")
@pytest.mark.parametrize("num_tokens", [i + 1 for i in range(16)])
def test_jit_vs_aot(num_tokens):
    """JIT kernel should produce bitwise identical results to AOT kernel."""
    mat_a, mat_b = _make_inputs(num_tokens)

    jit_output = jit_dsv3_fused_a_gemm(mat_a, mat_b)
    aot_output = aot_dsv3_fused_a_gemm(mat_a, mat_b)

    torch.testing.assert_close(jit_output, aot_output, rtol=0, atol=0)


if __name__ == "__main__":
    import subprocess

    subprocess.call(["pytest", "--tb=short", str(__file__)])
