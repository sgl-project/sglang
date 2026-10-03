"""Cake MoE router GEMMs (M in [1, 16]) through sglang.kernels.

Checks registry resolution of the three router ops, bitwise parity between the
facade and FlashInfer's direct call, and agreement with ``mat_a @ mat_b``
within BF16 tolerance. Skips (with the reason) when FlashInfer lacks the Cake
router module or the GPU is not SM100a / SM103a (FlashInfer serves other
architectures with its DSv3 CUDA kernel, which is not a Cake kernel).
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import gemm_router as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.gemm import cake as facade
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

ATOL = RTOL = 1e-2  # BF16 operands
# (op name, FlashInfer entry, K, N, out dtype)
CASES = [
    ("mm_m1_16_k7168_n128", "mm_M1_16_K7168_N128", 7168, 128, torch.bfloat16),
    ("mm_m1_16_k7168_n256", "mm_M1_16_K7168_N256", 7168, 256, torch.float32),
    ("mm_m1_16_k6144_n256", "mm_M1_16_K6144_N256", 6144, 256, torch.float32),
]


@pytest.mark.parametrize("name,entry,k,n,out_dtype", CASES)
def test_registry_resolves_flashinfer_backend(name, entry, k, n, out_dtype):
    spec = select_kernel(f"gemm.{name}", backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target == f"sglang.kernels.cake_kernels.gemm_router:{name}"
    assert spec.format_signature.in_place


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(cake.FI_MODULE, cake.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks flashinfer.jit.cake_router_gemm")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(f"Cake router GEMM is built for sm_100a/103a, device is {cc}")


@pytest.mark.parametrize("name,entry,k,n,out_dtype", CASES)
@pytest.mark.parametrize("num_tokens", [1, 7, 16])
@pytest.mark.parametrize("launch_with_pdl", [True, False])
def test_matches_flashinfer_and_reference(
    name, entry, k, n, out_dtype, num_tokens, launch_with_pdl
):
    _skip_unless_supported()
    torch.manual_seed(num_tokens)
    device = torch.device("cuda")
    mat_a = torch.randn(num_tokens, k, device=device, dtype=torch.bfloat16)
    mat_b = torch.randn(n, k, device=device, dtype=torch.bfloat16).t()  # column-major
    out = torch.empty(num_tokens, n, device=device, dtype=out_dtype)
    supports = getattr(cake, f"supports_{name}")
    assert supports(mat_a, mat_b, out)
    getattr(facade, f"cake_{name}")(mat_a, mat_b, out, launch_with_pdl=launch_with_pdl)
    import flashinfer.gemm as fi_gemm

    out_fi = torch.empty_like(out)
    getattr(fi_gemm, entry)(mat_a, mat_b, out_fi, launch_with_pdl=launch_with_pdl)
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    ref = mat_a.float() @ mat_b.float()
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


def test_supports_rejects_too_many_tokens_and_row_major_b():
    _skip_unless_supported()
    device = torch.device("cuda")
    mat_a = torch.randn(17, 7168, device=device, dtype=torch.bfloat16)
    mat_b = torch.randn(128, 7168, device=device, dtype=torch.bfloat16).t()
    out = torch.empty(17, 128, device=device, dtype=torch.bfloat16)
    assert not cake.supports_mm_m1_16_k7168_n128(mat_a, mat_b, out)
    mat_a = mat_a[:8]
    out = out[:8]
    assert cake.supports_mm_m1_16_k7168_n128(mat_a, mat_b, out)
    assert not cake.supports_mm_m1_16_k7168_n128(mat_a, mat_b.contiguous(), out)
    assert not cake.supports_mm_m1_16_k7168_n128(mat_a, mat_b, out.float())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
