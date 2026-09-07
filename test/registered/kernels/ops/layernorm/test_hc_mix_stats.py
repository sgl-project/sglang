import sys

import pytest
import torch

from sglang.kernels.ops.layernorm.mhc import hc_mix_stats
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="hc_mix_stats requires CUDA",
)

EPS = 1e-6


def _reference(x_flat: torch.Tensor, hc_fn: torch.Tensor) -> torch.Tensor:
    x = x_flat.double()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return (x @ hc_fn.double().T) * rsqrt


def _torch_path(x_flat: torch.Tensor, hc_fn: torch.Tensor) -> torch.Tensor:
    # The implementation this kernel replaces (cuBLAS fp32 GEMM + torch reduction).
    x = x_flat.float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return torch.nn.functional.linear(x, hc_fn) * rsqrt


@pytest.mark.parametrize("m", [1, 7, 64, 300, 1024, 4096])
@pytest.mark.parametrize("k", [4096, 20480])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_hc_mix_stats_matches_reference(m: int, k: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x_flat = torch.randn(m, k, device="cuda", dtype=dtype)
    hc_fn = torch.randn(24, k, device="cuda", dtype=torch.float32) * 0.02

    got = hc_mix_stats(x_flat, hc_fn, EPS)
    ref = _reference(x_flat, hc_fn)
    torch_path = _torch_path(x_flat, hc_fn)

    assert got.shape == (m, 24) and got.dtype == torch.float32
    scale = ref.abs().max().clamp(min=1e-6)
    err_kernel = (got.double() - ref).abs().max() / scale
    err_torch = (torch_path.double() - ref).abs().max() / scale
    # fp32 accumulation over k terms; the kernel must be as close to the fp64
    # reference as the fp32 path it replaces.
    assert err_kernel < 1e-4
    assert err_kernel <= err_torch * 4 + 1e-6


@pytest.mark.parametrize("k", [4096, 20480])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_hc_mix_stats_is_batch_invariant(k: int, dtype: torch.dtype):
    """A row's result must not depend on how many other rows share the call."""
    torch.manual_seed(1)
    m = 300
    x_flat = torch.randn(m, k, device="cuda", dtype=dtype)
    hc_fn = torch.randn(24, k, device="cuda", dtype=torch.float32) * 0.02

    full = hc_mix_stats(x_flat, hc_fn, EPS)
    for rows in ([0], [5], [299], list(range(3, 10)), list(range(0, 300, 7))):
        idx = torch.tensor(rows, device="cuda")
        sub = hc_mix_stats(x_flat.index_select(0, idx).contiguous(), hc_fn, EPS)
        assert torch.equal(sub, full.index_select(0, idx)), rows


def test_hc_mix_stats_rows_are_independent():
    """Padding rows of a partial tile must not leak into real rows."""
    torch.manual_seed(2)
    k = 4096
    hc_fn = torch.randn(24, k, device="cuda", dtype=torch.float32)
    x33 = torch.randn(33, k, device="cuda", dtype=torch.bfloat16)
    x_big = torch.cat(
        [x33, torch.full((31, k), 1e4, device="cuda", dtype=torch.bfloat16)]
    )
    assert torch.equal(
        hc_mix_stats(x33, hc_fn, EPS), hc_mix_stats(x_big, hc_fn, EPS)[:33]
    )


def test_hc_mix_stats_empty():
    hc_fn = torch.randn(24, 4096, device="cuda", dtype=torch.float32)
    x_flat = torch.empty(0, 4096, device="cuda", dtype=torch.bfloat16)
    assert hc_mix_stats(x_flat, hc_fn, EPS).shape == (0, 24)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
