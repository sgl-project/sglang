"""Muse Spark small-M GEMM kernels (gfx950): tiled, row-major and FP8-weight GEMMs, split-K partials, FP32 router
GEMMs, unpool + boundary."""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)


def _close(a, b, rtol, atol):
    torch.testing.assert_close(a.float(), b.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("n,k", [(8192, 1024), (7168, 8192), (8192, 3584)])
@pytest.mark.parametrize("m", [17, 24, 32, 48, 64])
def test_tiled_gemm(n, k, m):
    from sglang.kernels.ops.gemm.bf16_tiled_small_m_gemm import (
        prepare_tiled_weight,
        tiled_gemm,
        tiled_gemm_applies,
    )

    torch.manual_seed(m)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    tiled = prepare_tiled_weight(w)
    if not tiled_gemm_applies(x, tiled):
        pytest.skip("no tuned launch at this width")
    ref = x.float() @ w.float().T
    _close(tiled_gemm(x, tiled, n, k), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "n,k",
    [
        (4096, 8192),
        (8192, 4096),
        (8192, 2048),
        (2048, 8192),
        (2560, 8192),
        (8192, 7168),
        (14336, 8192),
    ],
)
@pytest.mark.parametrize("m", [1, 8, 16, 17, 24, 32, 48, 64, 72, 96, 128])
def test_rowmajor_gemm(n, k, m):
    from sglang.kernels.ops.gemm.bf16_tiled_small_m_gemm import (
        rowmajor_gemm,
        rowmajor_gemm_applies,
    )

    torch.manual_seed(m)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    if not rowmajor_gemm_applies(x, w):
        pytest.skip("no tuned launch at this width")
    ref = x.float() @ w.float().T
    _close(rowmajor_gemm(x, w), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "n,k", [(4608, 8192), (8192, 2048), (4096, 8192), (8192, 4096)]
)
@pytest.mark.parametrize("m", [17, 64, 96, 128])
def test_fp8_gemm(n, k, m):
    from sglang.kernels.ops.gemm.fp8w_small_m_gemm import (
        fp8_gemm,
        fp8_gemm_applies,
        quantize_rows,
        quantize_weight,
    )

    torch.manual_seed(m)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    x[:, ::97] *= 20  # outlier channels, as residual-stream activations have
    fp8w = quantize_weight(w)
    xq, xs = quantize_rows(x)
    if not fp8_gemm_applies(xq, fp8w):
        pytest.skip("no tuned launch at this width")
    ref = x.float() @ w.float().T
    got = fp8_gemm(xq, xs, fp8w).float()
    # per-channel FP8 weights x per-row FP8 activations: a few percent of relative L2 error
    assert (got - ref).norm() / ref.norm() < 0.06
    exact = (xq.float() * xs[:, None]) @ (fp8w[0].float() * fp8w[1][:, None]).T
    assert (got - exact).norm() / exact.norm() < 4e-3


@pytest.mark.parametrize("m", [1, 16, 32])
def test_splitk_partials_sum(m):
    from sglang.kernels.ops.gemm.bf16_splitk_gemm import bf16_splitk_gemm_partials

    torch.manual_seed(m)
    w = (torch.randn(2304, 8192, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(m, 8192, device="cuda").to(torch.bfloat16)
    p = torch.empty((8, m, 2304), device="cuda", dtype=torch.float32)
    bf16_splitk_gemm_partials(x, w, p, block_n=32, block_k=128, split_k=8)
    _close(p.sum(0), x.float() @ w.float().T, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("m", [1, 8, 64, 80, 128, 384])
def test_router_gemm_fp32(m):
    from sglang.kernels.ops.gemm.router_gemm_fp32 import (
        router_gemm_tuned,
        router_gemv_fp32,
        router_gemv_fp32_supported,
    )

    torch.manual_seed(m)
    w = torch.randn(8192, 256, device="cuda") * 0.02
    x = torch.randn(m, 8192, device="cuda").to(torch.bfloat16)
    ref = x.double() @ w.double()
    if router_gemv_fp32_supported(x, w):
        _close(router_gemv_fp32(x, w), ref, rtol=1e-5, atol=1e-5)
    tuned = router_gemm_tuned(x, w)
    if m in (80, 128, 384):
        assert tuned is not None, "the MI355X config covers this verify width"
    if tuned is not None:
        _close(tuned, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("m", [1, 8, 16])
@pytest.mark.parametrize("pre_bf16", [False, True])
def test_unpool_gemm_norm_residual(m, pre_bf16):
    from sglang.kernels.ops.gemm.unpool_gemm_norm_residual import (
        unpool_gemm_norm_residual,
    )

    torch.manual_seed(m)
    h = 8192
    x = torch.randn(m, 4096, device="cuda").to(torch.bfloat16)
    w = (torch.randn(h, 4096, device="cuda") * 0.02).to(torch.bfloat16)
    stream = torch.randn(m, h, device="cuda")
    post_w = (torch.rand(h, device="cuda") + 0.5).to(torch.bfloat16)
    pre_w = (torch.rand(h, device="cuda") + 0.5).to(torch.bfloat16)
    alpha, beta = torch.rand(h, device="cuda"), torch.rand(h, device="cuda")
    out, new_stream = unpool_gemm_norm_residual(
        x, w, stream, post_w, pre_w, alpha, beta, 1e-6, 1e-5, pre_bf16=pre_bf16
    )
    branch = F.linear(x, w).float()
    post_in = (branch * post_w.float()).to(torch.bfloat16).float()
    normed = (
        (post_in * torch.rsqrt(post_in.pow(2).mean(-1, keepdim=True) + 1e-6))
        .to(torch.bfloat16)
        .float()
    )
    ref_stream = alpha * stream + beta * normed
    pre_in = ref_stream.to(torch.bfloat16).float() if pre_bf16 else ref_stream
    ref_out = (
        pre_in
        * torch.rsqrt(pre_in.pow(2).mean(-1, keepdim=True) + 1e-5)
        * pre_w.float()
    )
    _close(new_stream, ref_stream, rtol=2e-2, atol=2e-2)
    _close(out, ref_out, rtol=3e-2, atol=3e-2)
