"""SM90 small-M BF16 GEMV (TileLang): fp64 reference, cuBLAS, rows past M, determinism and
CUDA graph replay. CI takes the smallest M and both sides of each table row boundary; outside
CI (or with SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1) every M the tables admit."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")

SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
if SM90:
    from sglang.kernels.ops.gemm import hopper_bf16_small_m_gemv as g

pytestmark = pytest.mark.skipif(not SM90, reason="SM90")
U, PAD, NAN = 2.0**-24, 64, float("nan")
# GWC: bf16 out (unquant decode GEMMs); GWR / GWS: fp32 router out (decode / SP)
CASES = (
    [(n, k, "GWC") for n, k in g._GWC] + [(512, 8192, "GWR"), (512, 8192, "GWS")]
    if SM90
    else []
)


def _bits(t):
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def _fold(y):  # router columns: per 16, 8 experts' hi then their lo; logit = hi + lo
    return y.view(y.shape[0], -1, 2, 8).sum(2).flatten(1)


def _setup(n, k, mode, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tab = getattr(g, "_" + mode)
    x = torch.randn(tab[(n, k)][-1][0], k, generator=gen, device="cuda").bfloat16()
    w = (torch.randn(n, k, generator=gen, device="cuda") * 0.05).bfloat16()
    f32 = mode != "GWC"  # fp32 out takes the weight as [K, N]
    return tab, f32, x, w, w.t().contiguous() if f32 else w


def _test_ms(rows):
    """Row boundaries pick the kernel config: both sides of each, and M = 1."""
    edges = {1} | {r[0] for r in rows} | {r[0] + 1 for r in rows[:-1]}
    return get_ci_test_range(
        full_range=list(range(1, rows[-1][0] + 1)), ci_range=sorted(edges)
    )


def _err(y, ref):  # bf16: outputs not correctly rounded; fp32: squared error
    if y.dtype == torch.bfloat16:
        return (y != ref.bfloat16()).sum().item()
    return (y.double() - ref).square().sum().item()


def test_use_admits_only_table_shapes():
    for tab in (g._GWC, g._GWR, g._GWS):
        for (n, k), rows in tab.items():
            hi = rows[-1][0]
            assert g.use(1, n, k, tab) and g.use(hi, n, k, tab)
            assert not g.use(0, n, k, tab) and not g.use(hi + 1, n, k, tab)
            # non-multiple N / K are not in the tables: the caller keeps cuBLAS
            assert not g.use(1, n + 8, k, tab) and not g.use(1, n, k + 8, tab)


def test_wide_wgmma_stays_inside_the_build():
    """Importing the module and building its kernels leave TileLang's wgmma emitter and
    log level as they were; the kernels still get one m64nTBk16 per k16 step."""
    import logging
    import subprocess

    from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import (
        TensorCoreIntrinEmitter as E,
    )

    probe = (
        "import logging, tilelang\n"
        "from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import "
        "TensorCoreIntrinEmitter as E\n"
        "before = (logging.getLogger('tilelang').level, E._initialize_wgmma_prefix)\n"
        "import sglang.kernels.ops.gemm.hopper_bf16_small_m_gemv\n"
        "assert (logging.getLogger('tilelang').level, E._initialize_wgmma_prefix) == before\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
    stock, level = E._initialize_wgmma_prefix, logging.getLogger("tilelang").level
    tab, f32, x, w, arg = _setup(2304, 8192, "GWC", seed=0)
    m = 112  # TB 112: TileLang alone would issue gcd(112, 256) = 16-wide wgmmas
    g.gemv(x[:m], arg, torch.empty(m, 2304, dtype=torch.bfloat16, device="cuda"), tab)
    assert E._initialize_wgmma_prefix is stock
    assert logging.getLogger("tilelang").level == level
    assert "64, 112, 16," in g._K[g._config(m, 2304, 8192)].get_kernel_source()


@pytest.mark.parametrize("n,k,mode", CASES)
def test_table_ms(n, k, mode):
    tab, f32, x, w, arg = _setup(n, k, mode, seed=n + k)
    ms, dt = _test_ms(tab[(n, k)]), torch.float32 if f32 else torch.bfloat16
    xd, wd = x.double(), w.double()
    ref, s = xd @ wd.t(), xd.abs() @ wd.abs().t()
    if f32:
        ref, s = _fold(ref), _fold(s)
    ours = stock = 0.0
    for m in ms:
        xb = torch.full((m + PAD, k), NAN, dtype=torch.bfloat16, device="cuda")
        yb = torch.full((m + PAD, ref.shape[1]), NAN, dtype=dt, device="cuda")
        xb[:m] = x[:m]
        y = g.gemv(xb[:m], arg, yb[:m], tab)
        assert y.data_ptr() == yb.data_ptr() and yb[m:].isnan().all(), m
        # bitwise repeatable, and rows past M of x are never read
        again = g.gemv(x[:m].clone(), arg, torch.empty_like(y), tab)
        assert torch.equal(_bits(y), _bits(again)), m
        # any-order fp32 dots: gamma_K * sum|x w| (2x covers the hi + lo add and
        # non-RN accumulators); bf16 out adds one rounding (2^-9 |y|)
        tol = 2 * k * U * s[:m] + (0 if f32 else 2**-8 * ref[:m].abs())
        assert ((y.double() - ref[:m]).abs() <= tol).all(), m
        c = _fold(torch.mm(x[:m], arg, out_dtype=dt)) if f32 else F.linear(x[:m], w)
        ours, stock = ours + _err(y, ref[:m]), stock + _err(c, ref[:m])
    # as accurate as cuBLAS (the stock path) against the fp64 reference, up to the
    # summation order (it decides some roundings on long K: 1.27x at K = 40960)
    slack = 0 if f32 else 2e-5 * n * sum(ms)
    assert ours <= 1.5 * stock + slack, (ours, stock)


_GRAPH_CASES = [
    (2304, 8192, 5, "GWC"),
    (512, 8192, 100, "GWC"),
    (8192, 1024, 256, "GWC"),
    (1024, 40960, 64, "GWC"),
    (512, 8192, 64, "GWR"),
    (512, 8192, 256, "GWS"),
]


# CI: a K-split cluster kernel with the fp32 router epilogue
@pytest.mark.parametrize(
    "n,k,m,mode",
    get_ci_test_range(full_range=_GRAPH_CASES, ci_range=[(512, 8192, 64, "GWR")]),
)
def test_cuda_graph_replay(n, k, m, mode):
    tab, f32, x, _, arg = _setup(n, k, mode, seed=m)
    xs = x[:m].clone()
    dt = torch.float32 if f32 else torch.bfloat16
    y = torch.empty(m, n // 2 if f32 else n, dtype=dt, device="cuda")
    g.gemv(xs, arg, y, tab)  # build outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g.gemv(xs, arg, y, tab)
    for _ in range(3):
        xs.copy_(torch.randn(m, k, device="cuda").bfloat16())
        graph.replay()
        eager = g.gemv(xs.clone(), arg, torch.empty_like(y), tab)
        assert torch.equal(_bits(y), _bits(eager))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
