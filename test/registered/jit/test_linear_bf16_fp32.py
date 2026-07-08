"""Tests for the shared BF16 x FP32 GEMM helper and the JIT bf16xfp32 kernel."""

import sys

import pytest
import torch

import sglang.jit_kernel.dsv4.gemm as dsv4_gemm
from sglang.jit_kernel.dsv4.gemm import (
    _linear_bf16_fp32_jit,
    linear_bf16_fp32,
)
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=600, suite="nightly-kernel-1-gpu", nightly=True)


def _sm90_available() -> bool:
    return torch.cuda.is_available() and is_sm90_supported()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m", [1, 8, 64])
def test_linear_bf16_fp32_matches_fp32_reference(weight_dtype, m):
    torch.manual_seed(42)
    x = torch.randn((m, 4096), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((256, 4096), dtype=torch.float32, device="cuda").to(weight_dtype)

    out = linear_bf16_fp32(x, w)
    ref = x.float() @ w.float().t()

    assert out.shape == (m, 256)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=8e-2)


def test_linear_bf16_fp32_explicit_jit_dispatch(monkeypatch):
    x = torch.randn((8, 16), dtype=torch.bfloat16)
    w = torch.randn((64, 16), dtype=torch.float32)
    expected = torch.randn((8, 64), dtype=torch.float32)
    seen_min_m = []

    def fake_jit(x_arg, w_arg, *, min_m=8, **_kwargs):
        assert x_arg is x
        assert w_arg is w
        seen_min_m.append(min_m)
        return expected

    monkeypatch.setattr(dsv4_gemm, "_linear_bf16_fp32_jit", fake_jit)

    out = linear_bf16_fp32(x, w, jit_kernel_min_m=128)

    assert out is expected
    assert seen_min_m == [128]


def test_linear_bf16_fp32_explicit_jit_fallback(monkeypatch):
    x = torch.randn((8, 16), dtype=torch.bfloat16)
    w = torch.randn((64, 16), dtype=torch.float32)

    monkeypatch.setattr(
        dsv4_gemm,
        "_linear_bf16_fp32_jit",
        lambda *_args, **_kwargs: None,
    )

    out = linear_bf16_fp32(x, w, jit_kernel_min_m=128)
    ref = x.float() @ w.t()

    assert out.shape == (8, 64)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref)


@pytest.mark.skipif(not _sm90_available(), reason="JIT bf16xfp32 kernel is SM90-only")
def test_linear_bf16_fp32_jit_path():
    torch.manual_seed(42)
    x = torch.randn((16, 4096), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((256, 4096), dtype=torch.float32, device="cuda")

    out = _linear_bf16_fp32_jit(x, w)
    ref = x.float() @ w.t()

    assert out is not None
    assert out.shape == (16, 256)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=8e-2)


# m values chosen to cover every launch-config branch of
# _select_launch_config: split_k 8/4/2/1, tile_m 16/64, wgn 1/2, stage 3/5.
_FULL_M_RANGE = [1, 8, 16, 64, 128, 512, 640, 832, 1024, 2048]
_CI_M_RANGE = [1, 64, 2048]


@pytest.mark.skipif(not _sm90_available(), reason="JIT bf16xfp32 kernel is SM90-only")
@pytest.mark.parametrize("m", get_ci_test_range(_FULL_M_RANGE, _CI_M_RANGE))
@pytest.mark.parametrize("n", get_ci_test_range([192, 256, 512], [256]))
@pytest.mark.parametrize("k", [4096])
def test_gemm_bf16xfp32_kernel_matches_reference(m, n, k):
    from sglang.jit_kernel.gemm_bf16xfp32 import gemm_bf16xfp32, split_fp32_weight

    torch.manual_seed(10086)
    x = torch.randn((m, k), dtype=torch.float32, device="cuda").to(torch.bfloat16)
    w = torch.randn((n, k), dtype=torch.float32, device="cuda")

    w_high, w_low = split_fp32_weight(w)
    out = gemm_bf16xfp32(x, w_high, w_low)
    ref = torch.matmul(x.float(), w.t())

    assert out.shape == (m, n)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=8e-2, atol=1e-2)


def test_select_launch_config_tuned_table():
    from sglang.jit_kernel.gemm_bf16xfp32 import _select_launch_config

    # Large-SM parts (H200-class) hit the measured table for LongCat shapes.
    assert _select_launch_config(64, 768, 6144, sm_count=132) == (16, 64, 128, 3, 2, 4)
    assert _select_launch_config(128, 768, 6144, sm_count=132) == (16, 64, 128, 3, 2, 8)
    assert _select_launch_config(512, 768, 6144, sm_count=132) == (64, 64, 64, 5, 1, 4)
    assert _select_launch_config(2048, 768, 6144, sm_count=132) == (64, 64, 64, 4, 2, 1)
    assert _select_launch_config(1024, 384, 3072, sm_count=132) == (64, 64, 64, 3, 1, 1)
    assert _select_launch_config(2048, 384, 3072, sm_count=132) == (64, 64, 64, 4, 2, 1)
    # H20-class parts and unknown shapes keep the upstream heuristic.
    assert _select_launch_config(2048, 768, 6144, sm_count=78) == (64, 64, 64, 3, 1, 1)
    assert _select_launch_config(2048, 768, 6144) == (64, 64, 64, 3, 1, 1)
    assert _select_launch_config(2048, 512, 4096, sm_count=132) == (64, 64, 64, 3, 1, 1)


# LongCat router shapes routed through the H200-tuned table on large-SM
# parts; m values chosen to cover each table row, including the
# tile64 + wgn=2 (dual math warpgroup) config the upstream dispatch never
# exposes.
_LONGCAT_SHAPES_FULL = [
    (64, 768, 6144),
    (128, 768, 6144),
    (512, 768, 6144),
    (2048, 768, 6144),
    (256, 384, 3072),
    (512, 384, 3072),
    (1024, 384, 3072),
    (2048, 384, 3072),
]
_LONGCAT_SHAPES_CI = [(128, 768, 6144), (2048, 768, 6144), (2048, 384, 3072)]


@pytest.mark.skipif(not _sm90_available(), reason="JIT bf16xfp32 kernel is SM90-only")
@pytest.mark.parametrize(
    "m,n,k", get_ci_test_range(_LONGCAT_SHAPES_FULL, _LONGCAT_SHAPES_CI)
)
def test_gemm_bf16xfp32_longcat_shapes(m, n, k):
    from sglang.jit_kernel.gemm_bf16xfp32 import gemm_bf16xfp32, split_fp32_weight

    torch.manual_seed(10086)
    x = torch.randn((m, k), dtype=torch.float32, device="cuda").to(torch.bfloat16)
    w = torch.randn((n, k), dtype=torch.float32, device="cuda")

    w_high, w_low = split_fp32_weight(w)
    out = gemm_bf16xfp32(x, w_high, w_low)
    ref = torch.matmul(x.float(), w.t())

    assert out.shape == (m, n)
    torch.testing.assert_close(out, ref, rtol=8e-2, atol=1e-2)


@pytest.mark.skipif(not _sm90_available(), reason="JIT bf16xfp32 kernel is SM90-only")
def test_gemm_bf16xfp32_kernel_bf16_output():
    from sglang.jit_kernel.gemm_bf16xfp32 import gemm_bf16xfp32, split_fp32_weight

    torch.manual_seed(10086)
    x = torch.randn((64, 4096), dtype=torch.float32, device="cuda").to(torch.bfloat16)
    w = torch.randn((256, 4096), dtype=torch.float32, device="cuda")

    w_high, w_low = split_fp32_weight(w)
    out = gemm_bf16xfp32(x, w_high, w_low, out_dtype=torch.bfloat16)
    ref = torch.matmul(x.float(), w.t())

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, rtol=8e-2, atol=5e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
