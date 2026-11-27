"""T40 parity: the BF16 split-K heuristic routes (SGLANG_ENABLE_BF16_SPLITK_GEMM_HEURISTIC) against
an FP32 reference and cuBLAS, at every decode graph-bucket M of the T38 decode for each listed
GLM-5.2 (N, K), for this GPU's own route and for the routes Rubin's SM count selects.

The route depends on the SM count (TGV split_k, warp-split-K output/token tiles), so a GB300 run
also forces the routes of 212 SMs (VR200 as measured on Hecate, T10 REPORT job 905821) and of 224
(the full die). Each route is checked eager (output and FP32 partials over NaN-poisoned memory,
so an unwritten or early-read tile shows), run-to-run bitwise, and under CUDA graph capture and
replay with PDL against its eager result.
"""

import contextlib
import json
import os
from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="SM 10.x only",
)

from sglang.kernels.ops.gemm import dense_bf16_gemm_warp_splitk as warp  # noqa: E402
from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (  # noqa: E402
    tgv_splitk_bf16_gemm_out,
    tgv_splitk_bf16_gemm_partials,
)
from sglang.srt.layers.quantization import unquant  # noqa: E402

DEV = "cuda"
GRAPH_BS = list(range(1, 9)) + list(range(10, 25, 2))
# Decode rows per rank: draft decode bs, target verify / draft extend 4 * bs.
ROWS = sorted(set(range(1, 25)) | {4 * b for b in GRAPH_BS})
# Forced SM counts: Rubin (VR200) 212 as measured on Hecate (T10 REPORT, job 905821); add 224 (full die)
# through T40_FORCE_SMS="212,224". A forced count equal to this GPU's own is not repeated.
RUBIN_SMS = tuple(int(v) for v in os.environ.get("T40_FORCE_SMS", "212").split(",") if v)
SHAPES = sorted(unquant._BF16_SPLITK_HEURISTIC_M_RANGES)


def _record(**fields) -> None:
    path = os.environ.get("T40_PARITY_JSONL")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"test": "bf16_splitk_heuristic", **fields}, default=str) + "\n")


def _device_sms() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


@contextlib.contextmanager
def _sm_count(num_sms: int):
    """default_tactic reads the device SM count; report ``num_sms`` instead."""
    real = torch.cuda.get_device_properties(0)

    class _Props:
        multi_processor_count = num_sms

        def __getattr__(self, name):
            return getattr(real, name)

    with mock.patch.object(warp._torch.cuda, "get_device_properties", lambda *_: _Props()):
        yield


def _route(m: int, n: int, k: int, num_sms: int):
    """The route production resolves (_resolve_bf16_splitk_heuristic_route) for ``num_sms``."""
    route = unquant.bf16_splitk_heuristic_route(m=m, n=n, k=k, num_sms=num_sms)
    if route is not None and route.kernel == "warp":
        try:
            with _sm_count(num_sms):
                route = route._replace(tactic=warp.default_tactic(m, n, k))
        except ValueError:  # production falls back to its other routes
            route = None
    return route


def _cases():
    if not torch.cuda.is_available():
        return
    own = _device_sms()
    for n, k in SHAPES:
        for m in ROWS:
            for sms in (None,) + tuple(v for v in RUBIN_SMS if v != own):
                if _route(m, n, k, sms or own) is not None:
                    yield n, k, m, sms


def test_route_table_invariants():
    """Every route at every bucket M for 148..224 SMs: one TGV wave, legal slices, M fits the tile."""
    table = {}
    for num_sms in (148, 152, 160, 208, 212, 224):
        for n, k in SHAPES:
            for m in ROWS:
                r = unquant.bf16_splitk_heuristic_route(m=m, n=n, k=k, num_sms=num_sms)
                in_range = any(
                    lo <= m <= hi for lo, hi in unquant._BF16_SPLITK_HEURISTIC_M_RANGES[(n, k)]
                )
                assert (r is not None) == in_range or (r is None and m > 32), (n, k, m, r)
                if r is None:
                    continue
                if m <= 32:
                    assert r.kernel == "warp" and r.split_k == 1
                else:
                    assert r.kernel == "tgv"
                    pair_rows = 128 if r.tactic in (23, 24) else 256
                    ctas = 2 * -(n // -pair_rows)
                    assert ctas * r.split_k <= num_sms, (num_sms, n, k, m, r)
                    assert k % r.split_k == 0 and (k // 128) % r.split_k == 0
                    assert r.split_k == 1 or k // r.split_k >= 1024
                    cta_n = {23: 64, 24: 128, 27: 64, 28: 128}[r.tactic]
                    assert cta_n >= m
                table[f"{num_sms}:{n}x{k}:M{m}"] = [r.kernel, r.tactic, r.split_k]
    _record(kind="route_table", table=table)


def _poison(numel: int) -> None:
    # The caching allocator hands this block to the next same-size request: a route that
    # reads partials or output it did not write then sees NaN.
    t = torch.full((numel,), float("nan"), dtype=torch.float32, device=DEV)
    del t


def _run(route, x, w):
    m, n = x.shape[0], w.shape[0]
    _poison(max(route.split_k, 1) * m * n)
    if route.kernel == "warp":
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=DEV)
        warp.run_warp_splitk_dense(a=x, b=w.T, out=out, pdl=True, tactic=route.tactic)
    else:
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=DEV)
        tgv_splitk_bf16_gemm_out(x, w, out, tactic=route.tactic, split_k=route.split_k)
    return out


@pytest.mark.parametrize("n,k,m,sms", list(_cases()))
def test_splitk_route_parity(n, k, m, sms):
    num_sms = sms or _device_sms()
    route = _route(m, n, k, num_sms)
    assert route is not None, (m, n, k, num_sms)
    gen = torch.Generator(device=DEV).manual_seed(n * 7 + k * 3 + m)
    x = torch.randn(m, k, device=DEV, generator=gen).to(torch.bfloat16)
    w = (torch.randn(n, k, device=DEV, generator=gen) * 0.02).to(torch.bfloat16)
    ref = x.float() @ w.float().T
    out = _run(route, x, w)
    assert torch.isfinite(out).all(), f"{route}: non-finite (unwritten or poisoned) output"
    err = (out.float() - ref).abs()
    tol = ref.abs() * 2.0**-8 + 1e-3  # one bf16 rounding of the FP32 sum, plus slack
    assert (err <= tol).all(), (route, err.max().item())
    cublas = torch.nn.functional.linear(x, w)
    vs_cublas = (out.float() - cublas.float()).abs().max().item()
    assert torch.equal(_run(route, x, w), out), f"{route}: not run-to-run deterministic"
    if route.kernel == "tgv" and route.split_k > 1:
        _poison(route.split_k * m * n)
        partials = tgv_splitk_bf16_gemm_partials(x, w, tactic=route.tactic, split_k=route.split_k)
        assert torch.isfinite(partials).all(), f"{route}: non-finite FP32 partials"
        assert torch.equal(partials.sum(0).to(torch.bfloat16), out) or (
            (partials.sum(0) - ref).abs() <= tol
        ).all()
    # CUDA graph + PDL: capture the route, replay on new inputs, compare with eager.
    x_static = x.clone()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(route, x_static, w)  # warm the compile cache outside capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if route.kernel == "warp":
            g_out = torch.empty((m, n), dtype=torch.bfloat16, device=DEV)
            warp.run_warp_splitk_dense(a=x_static, b=w.T, out=g_out, pdl=True, tactic=route.tactic)
        else:
            g_out = torch.empty((m, n), dtype=torch.bfloat16, device=DEV)
            tgv_splitk_bf16_gemm_out(x_static, w, g_out, tactic=route.tactic, split_k=route.split_k)
    for rep in range(3):
        x_new = torch.randn(m, k, device=DEV, generator=gen).to(torch.bfloat16)
        x_static.copy_(x_new)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(g_out, _run(route, x_new, w)), f"{route}: graph replay {rep} != eager"
    _record(kind="parity", n=n, k=k, m=m, num_sms=num_sms, forced=sms is not None,
            route=[route.kernel, str(route.tactic), route.split_k],
            max_err=err.max().item(), max_vs_cublas=vs_cublas)


def test_dispatch_engages_heuristic(monkeypatch):
    """The real dispatch takes the route for a listed shape and leaves an unlisted one alone."""
    monkeypatch.setattr(unquant, "_BF16_GEMM_BACKEND", unquant.Bf16GemmBackend.CUTEDSL)
    unquant._init_bf16_splitk_heuristic()
    try:
        calls = []
        real = unquant._bf16_splitk_heuristic_gemm
        monkeypatch.setattr(
            unquant, "_bf16_splitk_heuristic_gemm",
            lambda x, w, r: calls.append(r) or real(x, w, r),
        )
        x = torch.randn(48, 6144, device=DEV).to(torch.bfloat16)
        w = (torch.randn(2624, 6144, device=DEV) * 0.02).to(torch.bfloat16)
        out = unquant._bf16_gemm_dispatch_impl(x, w, None)
        assert len(calls) == 1 and calls[0].kernel == "tgv"
        assert (out.float() - x.float() @ w.float().T).abs().max().item() < 0.05
        assert unquant.prefer_bf16_splitk_over_fused_a(x=x[:8], weight=w)
        w2 = (torch.randn(2560, 6144, device=DEV) * 0.02).to(torch.bfloat16)
        unquant._bf16_gemm_dispatch_impl(x, w2, None)
        assert len(calls) == 1, "an unlisted (N, K) took the heuristic"
        assert not unquant.prefer_bf16_splitk_over_fused_a(x=x[:8], weight=w2)
    finally:
        unquant._enable_bf16_splitk_heuristic = False
