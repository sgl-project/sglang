"""Tests for the JIT tiny GEMM kernels (SIMT and MMA templates)."""

import functools
import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import (
    get_ci_test_range,
    get_jit_cuda_arch,
    is_hip_runtime,
    parallel_compile,
)
from sglang.kernels.ops.gemm import tiny_gemm as tg
from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=400, stage="nightly", runner_config="1-gpu-large")

# Router gates (experts x hidden): DeepSeek-V3 / Kimi-K2 / GLM-5 / DeepSeek-V4-Flash /
# DeepSeek-V4.1-Flash / Kimi-K3 / GLM-5-Next, Qwen3 / Qwen3-Next / Qwen3.5 / Qwen3-Coder,
# dots3, gpt-oss (K = 2880); Kimi-K3 KDA (144 / 1536 x 128); DeepSeek-V4.1 indexer
# wk (128 x 512) and weights_proj (32 x 5120).
SHAPES = [
    (256, 7168),
    (384, 7168),
    (256, 6144),
    (256, 4096),
    (384, 5120),
    (896, 7168),
    (288, 4096),
    (144, 7168),
    (128, 2048),
    (128, 4096),
    (512, 2048),
    (512, 4096),
    (256, 3072),
    (160, 6144),
    (512, 2560),
    (128, 2560),
    (32, 2880),
    (128, 2880),
    (1536, 128),
    (128, 512),
    (32, 5120),
]
NUM_TOKENS = [1, 2, 3, 5, 8, 9, 13, 16, 17, 24, 32, 33, 64]
IMPLS = ["simt", "mma"]
# Beam, GLM-4.5, GLM-4.5-Air.
FP32_SHAPES = [(256, 5120), (160, 5120), (128, 4096)]
API_SHAPES = [(256, 7168), (1536, 128), (32, 2880), (256, 5120)]


def _skip_unsupported():
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        pytest.skip("SM90+ required")


def _supported(impl, n, k, weight_dtype=torch.bfloat16):
    if impl == "simt":
        return tg._supports_simt(n, k)
    return tg._supports_mma(n, k, weight_dtype)


def _run(impl, x, w, out_dtype):
    """Run one template with its default schedule for the shape."""
    n, k = w.shape
    out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
    tg._get_module(n, k, w.dtype, out_dtype, impl == "mma").run(x, w, out)
    return out


@pytest.fixture(scope="module", autouse=True)
def _build_all_modules():
    """Build every module the tests below reach in parallel, not one by one on first use."""
    if not torch.cuda.is_available() or is_hip_runtime():
        return
    bf16, fp32 = torch.bfloat16, torch.float32
    modules = {
        (n, k, bf16, out_dtype, impl) for impl, n, k, out_dtype in TINY_GEMM_CASES
    }
    modules |= {(n, k, fp32, fp32, impl) for impl in IMPLS for n, k in FP32_SHAPES}
    # The dispatching entry point may pick either kernel.
    for impl in IMPLS:
        modules |= {
            (n, k, fp32 if k == 5120 else bf16, fp32, impl) for n, k in API_SHAPES
        }
    builders = [
        functools.partial(tg._get_module, n, k, wd, od, impl == "mma")
        for n, k, wd, od, impl in modules
        if _supported(impl, n, k, wd)
    ]
    parallel_compile(builders)


def _check(out, x, w):
    ref = x.double() @ w.double().T
    scale = ref.abs().max().item()
    if out.dtype == torch.float32:
        # Exact products, fp32 sums: errors of a few fp32 ulps of the row norm.
        tol = 2e-6 + 2e-5 * scale
    else:
        tol = 1e-2 * scale
    err = (out.double() - ref).abs().max().item()
    assert err <= tol, f"max error {err:.3e} > {tol:.3e}"


TINY_GEMM_CASES = get_ci_test_range(
    [
        (impl, n, k, out_dtype)
        for impl, (n, k), out_dtype in itertools.product(
            IMPLS, SHAPES, (torch.float32, torch.bfloat16)
        )
    ],
    [
        ("simt", 384, 7168, torch.float32),
        ("simt", 1536, 128, torch.bfloat16),
        ("simt", 32, 5120, torch.bfloat16),
        ("mma", 256, 7168, torch.float32),
        ("mma", 896, 7168, torch.float32),
        ("mma", 32, 2880, torch.float32),
        ("mma", 128, 512, torch.bfloat16),
    ],
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("impl,n,k,out_dtype", TINY_GEMM_CASES)
def test_tiny_gemm(impl, n, k, out_dtype):
    """Both templates against FP64, and batch invariance: a row's result is
    bitwise the same whatever the batch it is computed in."""
    _skip_unsupported()
    if not _supported(impl, n, k):
        pytest.skip(f"{impl} does not support N={n}, K={k}")
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda").to(torch.bfloat16)
    x = torch.randn(max(NUM_TOKENS), k, device="cuda").to(torch.bfloat16)
    full = _run(impl, x, w, out_dtype)
    _check(full, x, w)
    for m in NUM_TOKENS:
        out = _run(impl, x[:m], w, out_dtype)
        assert torch.equal(out, full[:m]), f"m={m}: not batch invariant"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n,k", FP32_SHAPES)
def test_tiny_gemm_fp32_weights(n, k):
    """FP32 router weights are read in FP32 (SIMT only, at every m): a BF16 read
    misses FP64 by ~1e-3."""
    _skip_unsupported()
    torch.manual_seed(42)
    w = torch.randn(n, k, device="cuda") / k**0.5
    x = torch.randn(max(NUM_TOKENS), k, dtype=torch.bfloat16, device="cuda")
    full = tiny_gemm_bf16(x, w, out_dtype=torch.float32)
    _check(full, x, w)
    for m in (1, 4, 16, 33):
        assert torch.equal(tiny_gemm_bf16(x[:m], w, out_dtype=torch.float32), full[:m])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n,k", API_SHAPES)
def test_tiny_gemm_api(n, k):
    """The dispatching entry point: row-sliced x, preallocated out, FP32 weights."""
    _skip_unsupported()
    torch.manual_seed(0)
    weight_dtype = torch.float32 if k == 5120 else torch.bfloat16
    assert can_use_tiny_gemm(n, k)
    w = (torch.randn(n, k, device="cuda") / k**0.5).to(weight_dtype)
    buf = torch.randn(16, k + 64, device="cuda").to(torch.bfloat16)
    x = buf[:, :k]
    out = tiny_gemm_bf16(x, w, out_dtype=torch.float32)
    _check(out, x, w)
    out2 = torch.empty_like(out)
    tiny_gemm_bf16(x.contiguous(), w, out2)
    assert torch.equal(out, out2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
