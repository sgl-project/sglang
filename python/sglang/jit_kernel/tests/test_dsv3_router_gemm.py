import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.dsv3_router_gemm import dsv3_router_gemm as jit_dsv3_router_gemm

try:
    from sgl_kernel import dsv3_router_gemm as aot_dsv3_router_gemm

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

HIDDEN_DIM = 7168


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="requires CUDA SM90+",
)
@pytest.mark.parametrize("num_tokens", [i + 1 for i in range(16)])
@pytest.mark.parametrize("num_experts", [256, 384])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_jit_vs_torch_matmul(num_tokens, num_experts, out_dtype):
    torch.manual_seed(42)
    mat_a = torch.randn(num_tokens, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")
    mat_b = torch.randn(num_experts, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")

    jit_output = jit_dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype)

    ref_output = F.linear(mat_a, mat_b)
    if out_dtype == torch.float32:
        ref_output = ref_output.float()

    torch.testing.assert_close(jit_output, ref_output, rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(
    not AOT_AVAILABLE
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9,
    reason="requires sgl_kernel AOT + CUDA SM90+",
)
@pytest.mark.parametrize("num_tokens", [i + 1 for i in range(16)])
@pytest.mark.parametrize("num_experts", [256, 384])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_jit_vs_aot(num_tokens, num_experts, out_dtype):
    torch.manual_seed(42)
    mat_a = torch.randn(num_tokens, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")
    mat_b = torch.randn(num_experts, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")

    jit_output = jit_dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype)
    aot_output = aot_dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype)

    torch.testing.assert_close(jit_output, aot_output, rtol=0, atol=0)


if __name__ == "__main__":
    import subprocess

    subprocess.call(["pytest", "--tb=short", str(__file__)])
