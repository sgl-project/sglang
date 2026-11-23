"""Muse Spark on SM90: the stacked split FP32 router (one GEMM over bf16 hi / lo halves, or the
small-M GEMV) vs stock's two GEMMs, and the lm_head scale as GEMM alpha vs scaling after."""

import types

import pytest
import torch

from sglang.srt.models import muse_spark as ms
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="1-gpu-large")

SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
pytestmark = pytest.mark.skipif(not SM90, reason="SM90")
U, E, K = 2.0**-24, 256, 8192


def _inputs(m, e, k, integer, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if integer:
        # w = h + q * 2^-10, |h| in 1..4: hi = h, lo = q * 2^-10, all partial sums exact
        x = torch.randint(-4, 5, (m, k), generator=gen, device="cuda")
        h = torch.randint(1, 5, (e, k), generator=gen, device="cuda")
        h = torch.where(torch.rand(e, k, generator=gen, device="cuda") < 0.5, -h, h)
        q = torch.randint(-3, 4, (e, k), generator=gen, device="cuda")
        return x.bfloat16(), h.float() + q.float() * 2.0**-10
    x = torch.randn(m, k, generator=gen, device="cuda").bfloat16()
    return x, torch.randn(e, k, generator=gen, device="cuda") / k**0.5


def _router(w):
    router = ms.MuseSpark1xRouter(w.shape[1], w.shape[0], dtype=torch.float32).cuda()
    router.weight = torch.nn.Parameter(w, requires_grad=False)
    return router


def _stock(x, w):  # stock forward_split2: two fp32-output GEMMs, then hi + lo
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    acc = torch.mm(x, hi.t().contiguous(), out_dtype=torch.float32)
    acc += torch.mm(x, lo.t().contiguous(), out_dtype=torch.float32)
    xd = x.double()
    return acc, xd @ hi.double().t(), xd @ lo.double().t(), hi, lo


@pytest.mark.parametrize("hopper", [True, False])
@pytest.mark.parametrize(
    "m,sp", [(1, 0), (9, 0), (100, 0), (208, 0), (209, 0), (1024, 0), (16, 1), (256, 1)]
)
def test_split_router_matches_stock(m, sp, hopper, monkeypatch):
    from sglang.kernels.ops.gemm.hopper_bf16_small_m_gemv import _GWR, _GWS, use

    monkeypatch.setattr(ms, "_hopper_kernels_enabled", lambda: hopper)
    gemv = hopper and use(m, 2 * E, K, _GWS if sp else _GWR)
    for integer in (True, False):
        x, w = _inputs(m, E, K, integer, seed=m + sp)
        out = _router(w).forward_split2(x, sp=bool(sp))
        stock, ref_hi, ref_lo, hi, lo = _stock(x, w)
        ref = ref_hi + ref_lo
        assert out.shape == (m, E) and out.dtype == torch.float32
        if integer:  # exact dots: every path rounds once, at hi + lo
            assert torch.equal(out, ref.float()) and torch.equal(stock, out)
            continue
        # any-order fp32 dots plus the hi + lo add: 2 * gamma_K * sum|x w|
        s = x.double().abs() @ (hi.double().abs() + lo.double().abs()).t()
        assert ((out.double() - ref).abs() <= 2 * K * U * s).all()
        if gemv and m >= 16:  # its K split keeps fp32 chains shorter than cuBLAS's
            rms = [(y.double() - ref).square().mean().sqrt() for y in (out, stock)]
            assert rms[0] <= 1.1 * rms[1], rms


@pytest.mark.parametrize("e,k", [(60, 1024), (256, 4096)])
def test_split_router_other_shapes(e, k, monkeypatch):
    # E % 8 != 0 stacks all hi then all lo; non-table shapes take the stacked GEMM
    monkeypatch.setattr(ms, "_hopper_kernels_enabled", lambda: True)
    for m in (9, 64, 300):
        x, w = _inputs(m, e, k, True, seed=e + m)
        router = _router(w)
        out = router.forward_split2(x)
        stock, ref_hi, ref_lo, *_ = _stock(x, w)
        assert torch.equal(out, (ref_hi + ref_lo).float()) and torch.equal(out, stock)
        assert torch.equal(router.forward_split2(x), out)  # cached stacking reused


@pytest.mark.parametrize("scale", [1.0, 0.25, 0.17677669529663687])
@pytest.mark.parametrize("m", [1, 7, 64, 300])
@pytest.mark.parametrize("buffered", [False, True])
def test_lm_head_alpha_matches_scale_after(scale, m, buffered):
    torch.manual_seed(m)
    n = 202048 // 8  # TP8 vocab shard
    head = types.SimpleNamespace(
        weight=(torch.randn(n, K, device="cuda") * 0.02).bfloat16()
    )
    x = torch.randn(m, K, device="cuda").bfloat16()
    f = ms.MuseSpark1xLogitsProcessor._compute_lm_head

    def proc(s):
        return types.SimpleNamespace(use_fp32_output=True, lm_scale=s)

    buf = torch.full((m, n), float("nan"), device="cuda") if buffered else None
    ours = f(proc(scale), x, head, out=buf)
    # stock: the fp32 GEMM, then the caller's logits.mul_(logit_scale)
    stock = f(proc(None), x, head).mul_(scale)
    # alpha is one fp32 multiply of the fp32 accumulator, like mul_; beta = 0 never reads buf
    assert buf is None or ours.data_ptr() == buf.data_ptr()
    assert torch.equal(ours.view(torch.int32), stock.view(torch.int32))
