"""Cake ragged BF16 grouped GEMM forward through sglang.kernels.

Checks registry resolution of the three forward ops, bitwise parity between
the facade (stable ``grouped_mm_bf16`` opt-in, one-shot ``grouped_gemm_fwd`` and
the prepared ``GroupedGemmLaunch``) and FlashInfer's direct calls, and agreement
with a per-group FP32 torch reference within BF16 tolerance. Skips (with the
reason) when FlashInfer lacks the Cake package / generated programs or the GPU
is not sm_100a / sm_103a / sm_107a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import gemm_grouped_bf16 as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.gemm.cake import (
    cake_grouped_gemm_fwd,
    cake_grouped_mm_bf16,
    cake_prepare_grouped_gemm_fwd,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = ("gemm.grouped_mm_bf16", "gemm.grouped_gemm_fwd", "gemm.prepare_grouped_gemm_fwd")
ATOL = RTOL = 1e-2  # BF16 operands


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.gemm_grouped_bf16:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake.FI_PACKAGE, cake.FI_MODULE, cake.FI_JIT_MODULE
    ):
        pytest.skip(
            "installed FlashInfer lacks flashinfer.experimental.cake_moe_grouped_gemm"
        )
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(
            f"Cake grouped BF16 GEMM is built for sm_100a/103a/107a, device is {cc}"
        )


def _inputs(sizes, n, k, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    sum_m = sum(sizes)
    x = torch.randn(sum_m, k, generator=g, device=device).to(torch.bfloat16)
    w = torch.randn(len(sizes), n, k, generator=g, device=device).to(torch.bfloat16)
    m_indptr = torch.tensor([0] + list(sizes), dtype=torch.int32, device=device).cumsum(
        0
    )
    return x, w, m_indptr.to(torch.int32)


def _reference(x, w, sizes):
    out = torch.empty(sum(sizes), w.shape[1], dtype=torch.float32, device=x.device)
    start = 0
    for e, size in enumerate(sizes):
        out[start : start + size] = x[start : start + size].float() @ w[e].float().T
        start += size
    return out


@pytest.mark.parametrize(
    "sizes,n,k",
    [
        pytest.param([64, 200, 0, 248], 256, 512, id="ragged_with_empty"),
        pytest.param([128, 128], 512, 4096, id="aligned_two_experts"),
        pytest.param([1, 300, 7], 256, 64, id="min_k_tiny_groups"),
    ],
)
def test_matches_flashinfer_and_reference(sizes, n, k):
    _skip_unless_supported()
    device = torch.device("cuda")
    x, w, m_indptr = _inputs(sizes, n, k, device, seed=n + k)
    if not cake.supports_grouped_mm_bf16(x, w, m_indptr):
        pytest.skip("FlashInfer registers no generated forward program for this device")
    out = cake_grouped_mm_bf16(x, w, m_indptr)
    from flashinfer.grouped_mm import grouped_mm_bf16

    out_fi = grouped_mm_bf16(x, w, m_indptr, backend="cake")
    out_fwd = cake_grouped_gemm_fwd(x, w, m_indptr[1:])  # offs[E] form
    launch = cake_prepare_grouped_gemm_fwd(x, w, m_indptr)
    assert isinstance(launch, cake.get_grouped_gemm_launch_class())
    out_prepared = launch.launch()
    torch.cuda.synchronize()
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (sum(sizes), n)
    assert torch.equal(out, out_fi)
    assert torch.equal(out_fwd, out_fi)
    assert torch.equal(out_prepared, out_fi)
    torch.testing.assert_close(
        out.float(), _reference(x, w, sizes), atol=ATOL, rtol=RTOL
    )


def test_prepared_launch_follows_new_offsets():
    _skip_unless_supported()
    device = torch.device("cuda")
    first = [100, 156, 0]
    x, w, m_indptr = _inputs(first, 256, 512, device, seed=3)
    if not cake.supports_grouped_gemm_fwd(x, w, m_indptr):
        pytest.skip("FlashInfer registers no generated forward program for this device")
    launch = cake_prepare_grouped_gemm_fwd(x, w, m_indptr)
    torch.testing.assert_close(
        launch.launch().float(), _reference(x, w, first), atol=ATOL, rtol=RTOL
    )
    second = [10, 200, 46]
    m_indptr.copy_(
        torch.tensor([0] + second, dtype=torch.int32, device=device).cumsum(0)
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        launch.launch().float(), _reference(x, w, second), atol=ATOL, rtol=RTOL
    )


def test_supports_rejects_non_bf16_output_and_bad_n():
    _skip_unless_supported()
    device = torch.device("cuda")
    x, w, m_indptr = _inputs([64, 64], 256, 512, device)
    assert not cake.supports_grouped_mm_bf16(x, w, m_indptr, out_dtype=torch.float16)
    assert not cake.supports_grouped_mm_bf16(x, w, m_indptr, tactic=0)
    x, w, m_indptr = _inputs([64, 64], 128, 512, device)  # N % 256 != 0
    assert not cake.supports_grouped_gemm_fwd(x, w, m_indptr)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
