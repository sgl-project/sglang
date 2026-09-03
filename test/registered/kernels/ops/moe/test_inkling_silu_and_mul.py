"""Numerics tests for the plain-Triton silu_and_mul (former helion kernels).

The kernels compute silu(gate) * up (* weight) in fp32 with one rounding cast
at the store, in the exact operation order of the helion kernels they
replaced, so bf16 outputs sit within 1 bf16 ulp of a same-order torch fp32
reference (tl.sigmoid and torch.sigmoid can differ in the last fp32 ulp).

The interleaved and non-interleaved kernels share the operation order, so the
two layouts must produce bitwise-identical outputs for the same gate/up data.

The small-batch tests pin the shared-expert shape family where the deleted
helion kernels were reported to produce NaN in EP+DP configs (see the
InklingBatchDenseMLP._swiglu comment); the port must stay finite and correct
there.
"""

import pytest
import torch

from sglang.kernels.ops.moe.inkling_moe import silu_and_mul
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _reference(
    gateup: torch.Tensor,
    weights: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    interleaved: bool,
) -> torch.Tensor:
    xf = gateup.float()
    if interleaved:
        gate, up = xf[:, 0::2], xf[:, 1::2]
    else:
        n = gateup.shape[1] // 2
        gate, up = xf[:, :n], xf[:, n:]
    out = gate * torch.sigmoid(gate) * up
    if weights is not None:
        out = out * weights.float()[:, None]
    return out.to(out_dtype or gateup.dtype)


def _ulp_diff_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dtype == b.dtype == torch.bfloat16
    mask = (1 << 15) - 1
    ai = a.contiguous().view(torch.int16).to(torch.int64)
    bi = b.contiguous().view(torch.int16).to(torch.int64)
    ai = torch.where(ai < 0, -(ai & mask), ai)
    bi = torch.where(bi < 0, -(bi & mask), bi)
    return (ai - bi).abs()


def _check(m: int, n: int, interleaved: bool, use_weights: bool, out_dtype=None):
    torch.manual_seed(m * 7919 + n)
    x = torch.randn(m, 2 * n, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(m, dtype=torch.bfloat16, device="cuda") if use_weights else None
    out = silu_and_mul(x, w, out_dtype, use_interleaved=interleaved)
    assert out.shape == (m, n)
    assert out.isfinite().all(), "kernel produced non-finite values"
    ref = _reference(x, w, out_dtype, interleaved)
    if out.dtype == torch.bfloat16:
        assert int(_ulp_diff_bf16(out, ref).max()) <= 1
    else:
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
    # Deterministic: an exact repeat must be bitwise identical.
    again = silu_and_mul(x, w, out_dtype, use_interleaved=interleaved)
    assert torch.equal(out, again)
    return out


# d42 serves N=2048 (tp1) / 1024 (tp2); d66 serves N=3072/1536/768.
# Rows: decode <= 256 reqs x topk 6; prefill up to 8192 tokens x topk 6.
@requires_cuda
@pytest.mark.parametrize(
    ("m", "n"),
    [(1, 1024), (6, 768), (384, 1024), (1536, 2048), (4096, 3072), (49152, 1024)],
)
@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("use_weights", [False, True])
def test_matches_reference(m: int, n: int, interleaved: bool, use_weights: bool):
    _check(m, n, interleaved, use_weights)


@requires_cuda
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_out_dtype(out_dtype: torch.dtype):
    out = _check(512, 1024, True, True, out_dtype=out_dtype)
    assert out.dtype == out_dtype


@requires_cuda
@pytest.mark.parametrize(("m", "n"), [(7, 384), (33, 960), (5, 512)])
def test_odd_widths(m: int, n: int):
    _check(m, n, True, False)
    _check(m, n, False, True)


@requires_cuda
def test_layouts_bitwise_consistent():
    """Same gate/up data through both layouts must match bitwise."""
    torch.manual_seed(0)
    m, n = 1234, 1536
    gate = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(m, dtype=torch.bfloat16, device="cuda")
    x_il = torch.stack([gate, up], dim=2).reshape(m, 2 * n).contiguous()
    x_nil = torch.cat([gate, up], dim=1).contiguous()
    out_il = silu_and_mul(x_il, w, None, use_interleaved=True)
    out_nil = silu_and_mul(x_nil, w, None, use_interleaved=False)
    assert torch.equal(out_il, out_nil)


@requires_cuda
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("n", [768, 1024, 1536])
def test_small_shared_expert_batches(m: int, n: int):
    """Regression for the helion NaN report: tiny gamma-weighted batches.

    m = n_shared_experts * tokens_on_dp_rank (2 shared experts, 1-8 tokens);
    n = the per-rank shared-expert width (d66 tp4 = 768, d42 tp2 = 1024).
    """
    out = _check(m, n, True, True)
    assert out.isfinite().all()


@requires_cuda
def test_zero_rows():
    x = torch.empty(0, 2048, dtype=torch.bfloat16, device="cuda")
    out = silu_and_mul(x, None, None, use_interleaved=True)
    assert out.shape == (0, 1024)


@requires_cuda
def test_int64_offsets():
    """Element offsets beyond 2**31 must address correctly (INT64_INDEX path)."""
    if torch.cuda.get_device_properties(0).total_memory < 16 * 2**30:
        pytest.skip("needs >= 16 GB GPU memory")
    torch.manual_seed(0)
    n = 1024
    m = 2**20 + 8  # numel = m * 2n = 2**31 + 16384 > 2**31
    x = torch.randn(m, 2 * n, dtype=torch.bfloat16, device="cuda")
    assert x.numel() > 2**31
    for interleaved in (True, False):
        out = silu_and_mul(x, None, None, use_interleaved=interleaved)
        # The tail rows sit at element offsets >= 2**31.
        tail_ref = _reference(x[-4:], None, None, interleaved)
        assert int(_ulp_diff_bf16(out[-4:], tail_ref).max()) <= 1
        assert out[-4:].isfinite().all()
        del out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-x"]))
