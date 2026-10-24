"""Cake BF16 batched matmul (``bmm_bf16`` backend ``"cake"``) through sglang.kernels.

Checks registry resolution, bitwise parity between the facade and FlashInfer's
direct call, and agreement with ``torch.bmm`` within BF16 tolerance. Skips
(with the reason) when FlashInfer lacks the Cake BMM module or the GPU is not
SM100a / SM103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import gemm_bmm_bf16 as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.gemm.cake import cake_bmm_bf16
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OP = "gemm.bmm_bf16"
ATOL = RTOL = 1e-2  # BF16 operands


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target == "sglang.kernels.cake_kernels.gemm_bmm_bf16:bmm_bf16"


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(cake.FI_MODULE, cake.FI_JIT_MODULE):
        pytest.skip(
            "installed FlashInfer lacks flashinfer.jit.gemm.cake_blackwell_bf16_bmm"
        )
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(f"Cake BF16 BMM is built for sm_100a/103a, device is {cc}")


@pytest.mark.parametrize(
    "batch,m,n,k", [(4, 48, 80, 64), (2, 130, 1024, 256), (3, 7, 16, 1024)]
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_matches_flashinfer_and_reference(batch, m, n, k, out_dtype):
    _skip_unless_supported()
    torch.manual_seed(0)
    device = torch.device("cuda")
    A = torch.randn(batch, m, k, device=device, dtype=torch.bfloat16)
    W = torch.randn(batch, n, k, device=device, dtype=torch.bfloat16)
    B = W.transpose(-2, -1)  # exact column-major [batch, k, n] view
    assert cake.supports_bmm_bf16(A, B, out_dtype=out_dtype)
    out = cake_bmm_bf16(A, B, out_dtype=out_dtype)
    from flashinfer.gemm import bmm_bf16

    out_fi = bmm_bf16(A, B, out_dtype=out_dtype, backend="cake")
    torch.cuda.synchronize()
    assert out.dtype == out_dtype and tuple(out.shape) == (batch, m, n)
    assert torch.equal(out, out_fi)
    ref = torch.bmm(A.float(), B.float())
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


def test_supports_rejects_unsupported_k_and_row_major_b():
    _skip_unless_supported()
    device = torch.device("cuda")
    A = torch.randn(2, 16, 128, device=device, dtype=torch.bfloat16)
    W = torch.randn(2, 32, 128, device=device, dtype=torch.bfloat16)
    assert not cake.supports_bmm_bf16(A, W.transpose(-2, -1))  # K = 128
    A = torch.randn(2, 16, 256, device=device, dtype=torch.bfloat16)
    B_row_major = torch.randn(2, 256, 32, device=device, dtype=torch.bfloat16)
    assert not cake.supports_bmm_bf16(A, B_row_major)
    W = torch.randn(2, 36, 256, device=device, dtype=torch.bfloat16)
    assert not cake.supports_bmm_bf16(A, W.transpose(-2, -1))  # N % 8 != 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
