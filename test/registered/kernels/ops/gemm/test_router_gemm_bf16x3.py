"""The prefill-width bf16x3 router GEMM (W = hi + mid + lo in three bf16 MFMA terms).

It replaces ``hidden.float() @ W`` at prefill widths. Bar: logits at least as close to an
fp64 reference as a strict fp32 GEMM, the same top-8 expert ids away from fp64 near-ties,
and one compiled variant whatever the row count (the router runs outside CUDA graphs).
"""

import pytest
import torch

from sglang.kernels.ops.gemm.router_gemm_bf16x3 import (
    ROUTER_BF16X3_MIN_M,
    router_gemm_bf16x3,
    router_gemm_bf16x3_supported,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# Muse Spark: hidden 8192, 256 experts.
K, N = 8192, 256


def _inputs(m, seed=11):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, K, generator=gen, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(K, N, generator=gen, device="cuda", dtype=torch.float32) * 0.02
    return x, w


def test_weight_split_reconstructs_bitwise():
    _, w = _inputs(1)
    hi = w.bfloat16()
    r = w - hi.float()
    mid = r.bfloat16()
    lo = (r - mid.float()).bfloat16()
    torch.testing.assert_close(hi.float() + mid.float() + lo.float(), w, rtol=0, atol=0)


def _rownorm_err(y, ref64):
    # Row-normalized: element-wise relative error is dominated by logits near zero.
    return ((y.double() - ref64).norm(dim=1) / ref64.norm(dim=1)).max().item()


@pytest.mark.parametrize("m", [ROUTER_BF16X3_MIN_M, 333, 1024, 2048, 4096])
@pytest.mark.parametrize("seed", [11, 12])
def test_matches_fp64_within_fp32_error(m, seed):
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        x, w = _inputs(m, seed=seed)
        assert router_gemm_bf16x3_supported(x, w)
        ref64 = x.double() @ w.double()
        fp32 = x.float() @ w
        out = router_gemm_bf16x3(x, w)
    finally:
        torch.set_float32_matmul_precision(prev)
    assert _rownorm_err(out, ref64) <= _rownorm_err(fp32, ref64) * 1.05 + 1e-9


@pytest.mark.parametrize("m", [1024, 2048])
def test_top8_ids_match_except_near_ties(m):
    x, w = _inputs(m, seed=23)
    ref64 = x.double() @ w.double()
    out = router_gemm_bf16x3(x, w)
    ids_ref = torch.topk(ref64, 8, dim=1).indices.sort(1).values
    ids_out = torch.topk(out.double(), 8, dim=1).indices.sort(1).values
    mismatch = (ids_out != ids_ref).any(1)
    if mismatch.any():
        top9 = torch.topk(ref64[mismatch], 9, dim=1).values
        gap = (top9[:, 7] - top9[:, 8]) / top9[:, 7].abs().clamp_min(1e-6)
        assert (gap < 1e-5).all(), "top-8 mismatch outside fp64 near-ties"


def test_dispatch_gate():
    x, w = _inputs(ROUTER_BF16X3_MIN_M)
    assert router_gemm_bf16x3_supported(x, w)
    assert not router_gemm_bf16x3_supported(x[: ROUTER_BF16X3_MIN_M - 1], w)
    assert not router_gemm_bf16x3_supported(x, w.t())
    assert not router_gemm_bf16x3_supported(x.float(), w)


def test_row_count_does_not_recompile(monkeypatch):
    import triton.knobs as knobs

    if not hasattr(getattr(knobs, "compilation", None), "listener"):
        pytest.skip("triton.knobs.compilation.listener is not available")
    compiled = []
    prev = knobs.compilation.listener

    def listener(*, src, **kwargs):
        if src.name.startswith("_router_bf16x3"):
            compiled.append(src.name)
        if prev is not None:
            prev(src=src, **kwargs)

    monkeypatch.setattr(knobs.compilation, "listener", listener)
    for m in (257, 300, 512, 1024, 1712, 2049):
        x, w = _inputs(m)
        router_gemm_bf16x3(x, w)
    torch.cuda.synchronize()
    assert len(compiled) <= 2, compiled
