"""Fused inverse-RoPE + WO-A (the SM90 DSV4.1 decode path) against a torch reference."""

import sys

import pytest
import torch

from sglang.kernels.ops.gemm import dsv4_wo_a
from sglang.kernels.ops.gemm.dsv4_wo_a import wo_a_bf16_fused_inv_rope
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="requires a CUDA GPU",
)

MAX_POS = 4096


def _inputs(tokens: int, groups: int, position_dtype: torch.dtype):
    torch.manual_seed(1234)
    # The attention output carries every head; the layer reads its local slice.
    backing = torch.randn(tokens, 64, 512, dtype=torch.bfloat16, device="cuda")
    o = backing[:, : 8 * groups]
    weight = torch.randn(groups, 1024, 4096, dtype=torch.bfloat16, device="cuda")
    weight.mul_(0.05)
    angles = torch.outer(
        torch.arange(MAX_POS, device="cuda", dtype=torch.float32),
        1.0 / 10000.0 ** (torch.arange(32, device="cuda", dtype=torch.float32) / 32),
    )
    freqs = torch.stack([angles.cos(), angles.sin()], dim=-1).reshape(MAX_POS, 64)
    positions = torch.randint(
        0, MAX_POS, (tokens,), device="cuda", dtype=position_dtype
    )
    return o, weight, freqs, positions


def _reference(o, weight, freqs, positions):
    """Inverse RoPE (fp32, rounded to bf16) then the grouped GEMM in fp64."""
    rotated = o.float().clone()
    f = freqs[positions.long()]
    cos, sin = f[:, None, 0::2], f[:, None, 1::2]
    tail = rotated[..., -64:]
    re = tail[..., 0::2].clone()
    im = tail[..., 1::2].clone()
    tail[..., 0::2] = (re * cos + im * sin).bfloat16().float()
    tail[..., 1::2] = (im * cos - re * sin).bfloat16().float()
    groups = weight.shape[0]
    x = rotated.reshape(o.shape[0], groups, 4096).double()
    return torch.einsum("tgd,grd->tgr", x, weight.double())


def _relative_error(o, weight, freqs, positions) -> float:
    tokens, groups = o.shape[0], weight.shape[0]
    y = wo_a_bf16_fused_inv_rope(o.view(tokens, groups, 4096), weight, freqs, positions)
    ref = _reference(o, weight, freqs, positions)
    return ((y.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("tokens", [1, 2, 7, 16])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_matches_reference(tokens: int, groups: int, position_dtype: torch.dtype):
    o, weight, freqs, positions = _inputs(tokens, groups, position_dtype)
    # bf16 output rounding only; cuBLAS bf16 einsum lands at the same ~1.7e-3.
    err = _relative_error(o, weight, freqs, positions)
    assert err < 3e-3, f"relative error {err:.3g}"


def test_split_k_arrivals_survive_int32_wrap():
    """The last of eight split-K programs reduces each tile, picked by counters that
    are never reset; launches after int32 overflow must still reduce every tile."""
    o, weight, freqs, positions = _inputs(4, 2, torch.int32)
    wo_a_bf16_fused_inv_rope(o.view(4, 2, 4096), weight, freqs, positions)
    arrivals = dsv4_wo_a._wo_a_arrivals[o.device]
    # Each launch adds 8 per tile; start two launches short of the wrap.
    arrivals.fill_(2**31 - 16)
    for step in range(4):
        # Fresh positions each launch, so a skipped reduce cannot pass by reading
        # the previous launch's result back out of the caching allocator.
        positions = torch.randint(0, MAX_POS, (4,), device="cuda", dtype=torch.int32)
        err = _relative_error(o, weight, freqs, positions)
        assert err < 3e-3, f"relative error {err:.3g} at launch {step}"
    # 16 N tiles x 2 groups; slots past these belong to wider launches.
    assert bool((arrivals[:32] < 0).all()), "the counters did not wrap"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
