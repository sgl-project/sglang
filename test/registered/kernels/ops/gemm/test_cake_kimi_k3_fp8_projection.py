"""Cake Kimi-K3 FP8_PB_WO projection through sglang.kernels.

Checks registry resolution of the four ops, bitwise parity between the facade
(prepared runner and one-shot form) and FlashInfer's direct calls, and
agreement with an exact quantized-operand torch reference within BF16
tolerance. Skips (with the reason) when FlashInfer lacks the Cake modules /
generated programs or the GPU is not SM100a / SM103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import gemm_kimi_k3_fp8_projection as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.gemm.cake import (
    cake_allocate_kimi_k3_fp8_projection_workspace,
    cake_kimi_k3_fp8_projection,
    cake_prepare_kimi_k3_fp8_projection,
    cake_prepare_kimi_k3_fp8_projection_weights,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "gemm.prepare_kimi_k3_fp8_projection_weights",
    "gemm.allocate_kimi_k3_fp8_projection_workspace",
    "gemm.prepare_kimi_k3_fp8_projection",
    "gemm.kimi_k3_fp8_projection",
)
BLOCK = 128
E4M3_MAX = 448.0
AMAX_FLOOR = 1e-4
ATOL = RTOL = 1e-2  # BF16 output


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.gemm_kimi_k3_fp8_projection:"
    )


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake.FI_MODULE, cake.FI_BACKEND_MODULE, cake.FI_JIT_MODULE
    ):
        pytest.skip("installed FlashInfer lacks flashinfer.gemm.kimi_k3_fp8_projection")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(
            f"Cake Kimi-K3 projection is built for sm_100a/103a, device is {cc}"
        )


def _per_block_cast_to_fp8(w):
    n, k = w.shape
    wv = w.view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    amax = wv.abs().float().amax(dim=(1, 3), keepdim=True).clamp(AMAX_FLOOR)
    sf = amax / E4M3_MAX
    q = (wv.float() * (1.0 / sf)).to(torch.float8_e4m3fn).view(n, k)
    return q, sf.view(n // BLOCK, k // BLOCK)


def _make_weight(n_valid, k, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    n128 = -(-n_valid // BLOCK) * BLOCK
    w = torch.randn((n128, k), device=device, generator=g, dtype=torch.float32) * 0.02
    w[n_valid:] = 0.0
    w_q, sf = _per_block_cast_to_fp8(w.to(torch.bfloat16))
    return w_q.contiguous(), sf.reshape(n128 // BLOCK, 1, k // BLOCK, 1).contiguous()


def _reference(x, weight, scale, n_valid):
    """Exact emulation: UE8M0 requant of the weight, per-token UE8M0 cast of x."""
    from flashinfer.experimental.kimi_k3_fp8_projection import cake_backend as cb

    n, k = weight.shape
    m = x.shape[0]
    w2, s2 = cb.requant_weight_ue8m0(weight, scale.reshape(n // BLOCK, k // BLOCK))
    xv = x.view(m, k // BLOCK, BLOCK)
    amax = xv.abs().float().amax(dim=2).clamp(AMAX_FLOOR)
    a_sf = cb.ceil_to_ue8m0(amax / E4M3_MAX)
    a_q = (xv.float() * (1.0 / a_sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, k)
    a = (a_q.float().view(m, k // BLOCK, BLOCK) * a_sf.view(m, k // BLOCK, 1)).view(
        m, k
    )
    w = (
        w2.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
        * s2.view(n // BLOCK, 1, k // BLOCK, 1)
    ).view(n, k)
    return (a @ w.T)[:, :n_valid]


# 64 rows exercise the decode table, 512 the persistent 2-CTA GEMM route.
@pytest.mark.parametrize("m", [64, 512])
def test_projection_matches_flashinfer_and_reference(m):
    _skip_unless_supported()
    device = torch.device("cuda")
    n_valid, k = 1536, 7168
    weight, scale = _make_weight(n_valid, k, device, seed=11 + m)
    if not cake.supports_kimi_k3_fp8_projection_weights(weight, scale, n_valid):
        pytest.skip("FlashInfer registers no generated program for this device")
    prepared = cake_prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
    assert isinstance(prepared, cake.get_prepared_projection_weight_class())
    x = torch.randn((m, k), device=device, dtype=torch.float32).to(torch.bfloat16)
    workspace = cake_allocate_kimi_k3_fp8_projection_workspace(prepared, m)
    assert isinstance(workspace, cake.get_projection_workspace_class())
    out = torch.empty((m, n_valid), dtype=torch.bfloat16, device=device)
    if not cake.supports_kimi_k3_fp8_projection(x, prepared, out, workspace):
        pytest.skip("FlashInfer registers no route for this (M, N, K) on this device")
    runner = cake_prepare_kimi_k3_fp8_projection(x, prepared, out, workspace)
    assert isinstance(runner, cake.get_kimi_k3_fp8_projection_runner_class())
    assert runner.launch() is out

    from flashinfer.gemm import (
        allocate_kimi_k3_fp8_projection_workspace,
        kimi_k3_fp8_projection,
        prepare_kimi_k3_fp8_projection_weights,
    )

    prepared_fi = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
    out_fi = kimi_k3_fp8_projection(
        x,
        prepared_fi,
        workspace=allocate_kimi_k3_fp8_projection_workspace(prepared_fi, m),
    )
    out_oneshot = cake_kimi_k3_fp8_projection(x, prepared)
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(out_oneshot, out_fi)
    ref = _reference(x, weight, scale, n_valid)
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


def test_supports_rejects_unpadded_weight_and_fp16_activation():
    _skip_unless_supported()
    device = torch.device("cuda")
    weight, scale = _make_weight(1536, 7168, device, seed=5)
    assert not cake.supports_kimi_k3_fp8_projection_weights(weight[:1000], scale, 1000)
    assert not cake.supports_kimi_k3_fp8_projection_weights(weight, scale, 1535)
    if not cake.supports_kimi_k3_fp8_projection_weights(weight, scale, 1536):
        pytest.skip("FlashInfer registers no generated program for this device")
    prepared = cake_prepare_kimi_k3_fp8_projection_weights(weight, scale, 1536)
    x16 = torch.randn((64, 7168), device=device, dtype=torch.float16)
    assert not cake.supports_kimi_k3_fp8_projection(x16, prepared)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
