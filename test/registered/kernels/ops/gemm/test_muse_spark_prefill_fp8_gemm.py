"""Muse Spark prefill-width FP8 dense GEMMs (gfx950): the Triton per-row quantization and aiter's CK a8w8 GEMM on
the TP4 attention QKV / O, MoE pooler / unpooler shapes and the gathered Ulysses shapes. ``-s`` prints the row
quantization bandwidth and the GEMM times against ``F.linear``."""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=150, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)

TP4_SHAPES = [(4608, 8192), (8192, 2048), (4096, 8192), (8192, 4096)]
ULYSSES_TP4_SHAPES = [(18432, 8192), (8192, 8192)]


def _timed_us(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


def _activation(m, k, seed):
    torch.manual_seed(seed)
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    x[:, ::97] *= 20  # outlier channels, as residual-stream activations have
    return x


def _ref_quantize(x):
    """The kernel's formula in torch: per-row ``amax / 448`` scale, elements scaled by the rounded ``448 / amax``.
    Tensor / tensor division: torch's division by a Python scalar multiplies by a reciprocal and is not correctly
    rounded."""
    amax = x.float().abs().amax(1).clamp_min(1e-20)
    f8_max = torch.full_like(amax, 448.0)
    return (x.float() * (f8_max / amax)[:, None]).to(torch.float8_e4m3fn), amax / f8_max


@pytest.mark.parametrize("k", [2048, 4096, 8192])
@pytest.mark.parametrize("m", [129, 4096, 32768])
def test_quantize_rows(k, m):
    """The kernel's rows and scales equal the torch per-row quantization bit for bit (correctly rounded per-row
    divisions, one multiply per element); an all-zero row stays finite."""
    from sglang.kernels.ops.gemm.fp8_prefill_gemm import quantize_rows_fp8

    x = _activation(m, k, seed=m + k)
    x[7] = 0
    xq, xs = quantize_rows_fp8(x)
    ref_q, ref_s = _ref_quantize(x)
    assert torch.equal(xs, ref_s)
    assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8))
    t = _timed_us(lambda: quantize_rows_fp8(x))
    # 2 B read + 1 B written per element
    print(
        f"\nquantize_rows_fp8 [{m}, {k}]: {t:.1f} us, {3 * x.numel() / t / 1e6:.2f} TB/s"
    )


def test_quantize_head_rows():
    """The in-place head-major read of the Ulysses all-to-all output [W, M, HW] quantizes the same rows as the
    transposed copy."""
    from sglang.kernels.ops.gemm.fp8_prefill_gemm import (
        quantize_head_rows_fp8,
        quantize_rows_fp8,
    )

    torch.manual_seed(0)
    heads = torch.randn(4, 8193, 2048, device="cuda").to(torch.bfloat16)
    xq, xs = quantize_head_rows_fp8(heads)
    ref_q, ref_s = quantize_rows_fp8(heads.transpose(0, 1).reshape(8193, -1))
    assert torch.equal(xs, ref_s)
    assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8))


def _check_gemm(n, k, m):
    from sglang.kernels.ops.gemm.fp8_prefill_gemm import (
        fp8_prefill_gemm,
        fp8_prefill_gemm_applies,
        quantize_rows_fp8,
    )
    from sglang.kernels.ops.gemm.fp8w_small_m_gemm import quantize_weight

    torch.manual_seed(n + k)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    x = _activation(m, k, seed=m)
    fp8w = quantize_weight(w)
    assert fp8_prefill_gemm_applies(x, fp8w)
    xq, xs = quantize_rows_fp8(x)
    got = fp8_prefill_gemm(xq, xs, fp8w).float()
    ref = x.float() @ w.float().T
    # per-channel FP8 weights x per-row FP8 activations: a few percent of relative L2 error
    assert (got - ref).norm() / ref.norm() < 0.06
    # the CK kernel applies the [M, 1] / [N, 1] scales the way the dequantized product does
    exact = (xq.float() * xs[:, None]) @ (fp8w[0].float() * fp8w[1][:, None]).T
    assert (got - exact).norm() / exact.norm() < 4e-3
    t_bf16 = _timed_us(lambda: F.linear(x, w))
    t_quant = _timed_us(lambda: quantize_rows_fp8(x))
    t_gemm = _timed_us(lambda: fp8_prefill_gemm(xq, xs, fp8w))
    print(
        f"\n[{n}, {k}] x {m} rows: bf16 {t_bf16:.0f} us | quant {t_quant:.0f} + fp8 {t_gemm:.0f} = "
        f"{t_quant + t_gemm:.0f} us ({(t_quant + t_gemm) / t_bf16 - 1:+.0%})"
    )


@pytest.mark.parametrize("n,k", TP4_SHAPES)
@pytest.mark.parametrize("m", [4096, 32768])
def test_prefill_fp8_gemm(n, k, m):
    _check_gemm(n, k, m)


@pytest.mark.parametrize("n,k", ULYSSES_TP4_SHAPES)
def test_prefill_fp8_gemm_ulysses(n, k):
    """The gathered Ulysses projections at the per-rank shard width of a 32K-token extend."""
    _check_gemm(n, k, 8192)


@pytest.mark.parametrize(
    "n,k,m", [(4608, 8192, 2049), (8192, 4096, 8191), (18432, 8192, 513)]
)
def test_prefill_fp8_gemm_odd_rows(n, k, m):
    """Row counts that are no multiple of any tile (SP shards of odd prefill lengths)."""
    _check_gemm(n, k, m)


def test_admission_widths():
    """Per-rank shapes stay BF16 below 2048 rows (the CK default config loses there), the gathered Ulysses shapes
    run FP8 from any SP shard width; a mismatched K or a missing copy never admits."""
    from sglang.kernels.ops.gemm.fp8_prefill_gemm import fp8_prefill_gemm_applies
    from sglang.kernels.ops.gemm.fp8w_small_m_gemm import quantize_weight

    per_rank = quantize_weight(
        torch.zeros(4608, 8192, device="cuda", dtype=torch.bfloat16)
    )
    ulysses = quantize_weight(
        torch.zeros(18432, 8192, device="cuda", dtype=torch.bfloat16)
    )

    def rows(m, k=8192):
        return torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)

    assert not fp8_prefill_gemm_applies(rows(2047), per_rank)
    assert fp8_prefill_gemm_applies(rows(2048), per_rank)
    assert fp8_prefill_gemm_applies(rows(129), ulysses)
    assert not fp8_prefill_gemm_applies(rows(128), ulysses)
    assert not fp8_prefill_gemm_applies(rows(4096, k=4096), per_rank)
    assert not fp8_prefill_gemm_applies(rows(4096), None)
