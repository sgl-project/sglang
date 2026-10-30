"""Cake grouped MXFP8 quantization through sglang.kernels.

Checks for the Cake adapter distributed by FlashInfer: the registry resolves
the explicit FlashInfer backend; the facade result is bitwise identical to
calling FlashInfer's public ``mxfp8_grouped_quantize(backend="cake")``
directly; and dequantizing the valid rows of every group with their UE8M0
block scales reproduces the input within the FP8 tolerance (atol 0.1,
rtol 0.1). Skips (with the reason) when the installed FlashInfer lacks the
Cake module, the GPU is outside sm_100a / sm_103a, or the generated profile
for the input dtype is not installed.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import quantization as cake_quant
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.quantization.cake import cake_mxfp8_grouped_quantize
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OP = "quantization.mxfp8_grouped_quantize"
BLOCK = 32


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.quantization:")


def _skip_unless_supported(a: torch.Tensor, mask: torch.Tensor):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_quant.MXFP8_FI_MODULE, cake_quant.MXFP8_FI_JIT_MODULE
    ):
        pytest.skip(
            "installed FlashInfer lacks flashinfer.jit.cake_grouped_mxfp8_quantize"
        )
    cc = torch.cuda.get_device_capability()
    if cc not in cake_quant.ARCHS:
        pytest.skip(f"Cake grouped MXFP8 is built for sm_100a/103a, device is {cc}")
    if not cake_quant.supports_mxfp8_grouped_quantize(a, mask):
        pytest.skip(f"no generated grouped MXFP8 profile installed for {a.dtype}")


def _logical_views(x_q: torch.Tensor, sf: torch.Tensor, k: int):
    """``(q [B, M, K] e4m3, scales [B, padded_M, padded_K // 32] uint8)``."""
    q = x_q.permute(2, 0, 1)[:, :, :k].contiguous()
    b = sf.shape[-1]
    # Undo FlashInfer's permute(3, 4, 1, 5, 2, 0) -> [B, M/128, K/128, 32, 4, 4],
    # then the 128x4 swizzle (row = tile*128 + d4*32 + d32, col = ktile*4 + d4').
    grouped = sf.permute(5, 2, 4, 0, 1, 3)
    scales = grouped.transpose(2, 4).reshape(b, grouped.shape[1] * 128, -1)
    return q, scales


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("b,m,k", [(3, 256, 4096), (2, 200, 1024)])
def test_matches_flashinfer_and_reference(dtype, b, m, k):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(500 + b + m + k)
    a = torch.randn(b, m, k, device=device, dtype=dtype, generator=g)
    mask = torch.tensor([m, max(m // 2, 1), 0][:b], device=device, dtype=torch.int32)
    _skip_unless_supported(a, mask)

    x_q, sf = cake_mxfp8_grouped_quantize(a, mask)
    from flashinfer.quantization.fp8_quantization import mxfp8_grouped_quantize

    x_q_fi, sf_fi = mxfp8_grouped_quantize(a, mask, backend="cake")
    torch.cuda.synchronize()
    padded_k = (k + 127) // 128 * 128
    padded_m = (m + 127) // 128 * 128
    assert x_q.dtype == torch.float8_e4m3fn
    assert tuple(x_q.shape) == (m, padded_k, b)
    assert sf.dtype == torch.uint8
    assert tuple(sf.shape) == (32, 4, padded_m // 128, 4, padded_k // 128, b)
    q, scales = _logical_views(x_q, sf, k)
    q_fi, scales_fi = _logical_views(x_q_fi, sf_fi, k)
    for i in range(b):
        n = int(mask[i])
        if n == 0:
            continue
        # Only rows < mask[i] are defined; compare those bitwise.
        assert torch.equal(q[i, :n], q_fi[i, :n])
        assert torch.equal(scales[i, :n, : k // BLOCK], scales_fi[i, :n, : k // BLOCK])
        # UE8M0 dequant multiplier 2^(byte - 127) over each 32-wide block.
        mult = torch.exp2(scales[i, :n, : k // BLOCK].float() - 127.0)
        deq = (q[i, :n].float().view(n, k // BLOCK, BLOCK) * mult[:, :, None]).view(
            n, k
        )
        torch.testing.assert_close(deq, a[i, :n].float(), atol=0.1, rtol=0.1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
