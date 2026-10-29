"""Cake per-token NVFP4 GEMM (``mm_fp4`` backend ``"cake"``) through sglang.kernels.

Checks registry resolution of the four ops, bitwise parity between the facade
(one-shot, prepared runner and quantize + GEMM chain) and FlashInfer's direct
calls, and agreement with a dequantized-operand torch reference within the FP4
block-scaled tolerance. Skips (with the reason) when FlashInfer lacks the Cake
modules / generated programs or the GPU is not SM100a / SM103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import gemm_nvfp4_per_token as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.gemm.cake import (
    cake_allocate_nvfp4_per_token_quantize_outputs,
    cake_mm_fp4_per_token,
    cake_prepare_mm_fp4_per_token,
    cake_prepare_nvfp4_per_token_chain,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "gemm.mm_fp4_per_token",
    "gemm.prepare_mm_fp4_per_token",
    "gemm.allocate_nvfp4_per_token_quantize_outputs",
    "gemm.prepare_nvfp4_per_token_chain",
)
GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)
ATOL, RTOL = 1.0, 0.1  # FP4 (e2m1) block-scaled


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.gemm_nvfp4_per_token:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake.FI_MODULE, cake.FI_SUPPORT_MODULE, cake.FI_JIT_MODULE, cake.FI_GEMM_MODULE
    ):
        pytest.skip(
            "installed FlashInfer lacks flashinfer.experimental.cake_nvfp4_per_token"
        )
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(
            f"Cake per-token NVFP4 GEMM is built for sm_100a/103a, device is {cc}"
        )


def _dequantize(fp4, sf):
    """FP32 dequantisation of packed E2M1 + swizzled 128x4 E4M3 scales (rows x K)."""
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


def _operands(m, n, k, device, seed):
    from flashinfer import SfLayout, nvfp4_quantize

    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16, generator=g)
    w_global_sf = (448.0 * 6.0) / w.float().abs().max()
    w_fp4, w_sf = nvfp4_quantize(
        w, w_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
    w_scale = (1.0 / w_global_sf).reshape(1).float()
    a_fp4, a_sf, alpha = nvfp4_quantize(
        x, gs_inv, per_token_activation=True, backend="cake", out_scale=w_scale
    )
    return x, a_fp4, a_sf, alpha, w_fp4, w_sf, gs_inv, w_scale


# (K, N) = (7168, 2112) is a validated family; 8 runs the swapped orientation,
# 130 the m orientation with a tail.
@pytest.mark.parametrize("m", [8, 130])
def test_mm_fp4_matches_flashinfer_and_reference(m):
    _skip_unless_supported()
    device = torch.device("cuda")
    n, k = 2112, 7168
    _x, a_fp4, a_sf, alpha, w_fp4, w_sf, _gs, _ws = _operands(m, n, k, device, seed=m)
    if not cake.supports_mm_fp4_per_token(
        a_fp4, w_fp4.T, a_sf, w_sf.T, alpha, torch.bfloat16
    ):
        pytest.skip("FlashInfer registers no generated kernel for this shape/device")
    out = cake_mm_fp4_per_token(a_fp4, w_fp4.T, a_sf, w_sf.T, alpha, torch.bfloat16)
    from flashinfer import mm_fp4

    out_fi = mm_fp4(a_fp4, w_fp4.T, a_sf, w_sf.T, alpha, torch.bfloat16, backend="cake")
    torch.cuda.synchronize()
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (m, n)
    assert torch.equal(out, out_fi)
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(w_fp4, w_sf).T) * alpha[:, None]
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


def test_prepared_runner_and_chain_match_one_shot():
    _skip_unless_supported()
    device = torch.device("cuda")
    m, n, k = 130, 2112, 7168
    x, a_fp4, a_sf, alpha, w_fp4, w_sf, gs_inv, w_scale = _operands(m, n, k, device, 7)
    out_prepared = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    if not cake.supports_prepare_mm_fp4_per_token(
        a_fp4, a_sf, w_fp4, w_sf, alpha, out_prepared
    ):
        pytest.skip("FlashInfer registers no generated kernel for this shape/device")
    runner = cake_prepare_mm_fp4_per_token(
        a_fp4, a_sf, w_fp4, w_sf, alpha, out_prepared
    )
    assert isinstance(runner, cake.get_nvfp4_per_token_gemm_runner_class())
    assert runner.launch() is out_prepared
    from flashinfer import mm_fp4

    out_fi = mm_fp4(a_fp4, w_fp4.T, a_sf, w_sf.T, alpha, torch.bfloat16, backend="cake")
    torch.cuda.synchronize()
    assert torch.equal(out_prepared, out_fi)

    ws = cake_allocate_nvfp4_per_token_quantize_outputs(m, k, device)
    assert isinstance(ws, cake.get_per_token_quantize_outputs_class())
    out_chain = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    assert cake.supports_prepare_nvfp4_per_token_chain(
        x, gs_inv, w_fp4, w_sf, out_chain, ws, out_scale=w_scale
    )
    chain = cake_prepare_nvfp4_per_token_chain(
        x, gs_inv, w_fp4, w_sf, out_chain, ws, out_scale=w_scale
    )
    assert isinstance(chain, cake.get_nvfp4_per_token_chain_runner_class())
    assert chain.launch() is out_chain
    torch.cuda.synchronize()
    # Same quantizer program -> same packed activation -> same GEMM result.
    assert torch.equal(ws.fp4, a_fp4.view(torch.uint8))
    assert torch.equal(ws.scale, alpha)
    assert torch.equal(out_chain, out_fi)
    ref = (_dequantize(ws.fp4, ws.sf) @ _dequantize(w_fp4, w_sf).T) * ws.scale[:, None]
    torch.testing.assert_close(out_chain.float(), ref, atol=ATOL, rtol=RTOL)


def test_supports_rejects_scalar_alpha_and_row_major_weight():
    _skip_unless_supported()
    device = torch.device("cuda")
    m, n, k = 8, 2112, 7168
    _x, a_fp4, a_sf, alpha, w_fp4, w_sf, _gs, _ws = _operands(m, n, k, device, 3)
    assert not cake.supports_mm_fp4_per_token(
        a_fp4, w_fp4.T, a_sf, w_sf.T, alpha[:1], torch.bfloat16
    )
    assert not cake.supports_mm_fp4_per_token(
        a_fp4, w_fp4.T.contiguous(), a_sf, w_sf.T, alpha, torch.bfloat16
    )
    assert not cake.supports_mm_fp4_per_token(
        a_fp4, w_fp4.T, a_sf, w_sf.T, alpha, torch.bfloat16, enable_pdl=False
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
