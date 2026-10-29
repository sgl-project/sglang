"""Cake SVDQuant NVFP4 GEMM (``mm_nvfp4_svdquant`` backend ``"cake"``) through sglang.kernels.

Checks registry resolution, bitwise parity between the facade and FlashInfer's
direct call on a catalogued route, and agreement with a dequantized-operand
torch reference within the FP4 block-scaled tolerance. Skips (with the reason)
when FlashInfer lacks the Cake SVDQuant modules, the CUDA toolkit is older
than 13.0, the route is not catalogued or the GPU is not SM100a / SM103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import gemm_svdquant as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.gemm.cake import cake_mm_nvfp4_svdquant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OP = "gemm.mm_nvfp4_svdquant"
ATOL, RTOL = 1.0, 0.1  # FP4 (e2m1) block-scaled


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target == "sglang.kernels.cake_kernels.gemm_svdquant:mm_nvfp4_svdquant"


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(cake.FI_MODULE, cake.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks flashinfer.jit.cake_nvfp4_svdquant")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(f"Cake SVDQuant is built for sm_100a/103a, device is {cc}")
    if not cake._cuda_13_or_newer():
        pytest.skip(
            f"Cake SVDQuant requires CUDA >= 13.0, torch has {torch.version.cuda}"
        )


def _dequantize(fp4, sf):
    from flashinfer.experimental.cake_nvfp4_per_token import cake_backend as cb

    fp4 = fp4.view(torch.uint8)
    rows, kh = fp4.shape
    k = 2 * kh
    e2m1 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=fp4.device,
    )
    nib = torch.stack([fp4 & 0xF, fp4 >> 4], dim=-1).reshape(rows, k).long()
    scales = cb.unswizzle_sf_128x4(sf.view(torch.uint8), rows, k).view(
        torch.float8_e4m3fn
    )
    return (e2m1[nib].view(rows, k // 16, 16) * scales.float()[:, :, None]).view(
        rows, k
    )


# Catalogued routes (both arches): N = 3072, K in {3072, 12288}, M = 129, rank 32.
@pytest.mark.parametrize("m,n,k,rank", [(129, 3072, 3072, 32), (129, 3072, 12288, 32)])
def test_matches_flashinfer_and_reference(m, n, k, rank):
    _skip_unless_supported()
    from flashinfer import SfLayout, nvfp4_quantize

    torch.manual_seed(k)
    device = torch.device("cuda")
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    a_gs = (448.0 * 6.0) / x.float().abs().max()
    b_gs = (448.0 * 6.0) / w.float().abs().max()
    a, a_sf = nvfp4_quantize(x, a_gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    b, b_sf = nvfp4_quantize(w, b_gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    a, b = a.view(torch.uint8), b.view(torch.uint8)
    a_sf, b_sf = (
        a_sf.view(torch.uint8).contiguous(),
        b_sf.view(torch.uint8).contiguous(),
    )
    alpha = (1.0 / (a_gs * b_gs)).reshape(1).float()
    d = (torch.randn(m, rank, device=device) * 0.1).to(torch.bfloat16)
    l1 = (torch.randn(n, rank, device=device) * 0.1).to(torch.bfloat16)
    if not cake.supports_mm_nvfp4_svdquant(a, b, a_sf, b_sf, alpha, d, l1):
        pytest.skip(
            "FlashInfer catalogues no unique Cake SVDQuant route for this problem"
        )
    out = cake_mm_nvfp4_svdquant(a, b, a_sf, b_sf, alpha, d, l1)
    from flashinfer.gemm import mm_nvfp4_svdquant

    out_fi = mm_nvfp4_svdquant(a, b, a_sf, b_sf, alpha, d, l1, backend="cake")
    torch.cuda.synchronize()
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (m, n)
    assert torch.equal(out, out_fi)
    ref = alpha * (
        _dequantize(a, a_sf) @ _dequantize(b, b_sf).T + d.float() @ l1.float().T
    )
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


def test_supports_rejects_uncatalogued_rank():
    _skip_unless_supported()
    device = torch.device("cuda")
    m, n, k = 129, 3072, 3072
    a = torch.zeros(m, k // 2, device=device, dtype=torch.uint8)
    b = torch.zeros(n, k // 2, device=device, dtype=torch.uint8)
    a_sf = torch.zeros(
        cake._swizzled_sf_size(m, k // 16), device=device, dtype=torch.uint8
    )
    b_sf = torch.zeros(
        cake._swizzled_sf_size(n, k // 16), device=device, dtype=torch.uint8
    )
    alpha = torch.ones(1, device=device)
    d = torch.zeros(m, 16, device=device, dtype=torch.bfloat16)  # rank % 32 != 0
    l1 = torch.zeros(n, 16, device=device, dtype=torch.bfloat16)
    assert not cake.supports_mm_nvfp4_svdquant(a, b, a_sf, b_sf, alpha, d, l1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
