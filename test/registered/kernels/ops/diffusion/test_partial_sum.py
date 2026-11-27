"""Each epilogue kernel that takes a two-rank row-parallel ``PartialSum`` writes,
byte for byte, what it writes for the reduced tensor: the two partials' sum
rounded to BF16, which is what a two-rank all-reduce leaves. Both partials are
local here; test_ipc_a2a_2_gpu covers one mapped from the peer device."""

import sys

import pytest
import torch

from sglang.kernels.kda_kernels.residual_gate_add_jit import _route
from sglang.kernels.ops.diffusion import (
    can_use_indexed_gate_rmsnorm_scale_shift,
    indexed_gate_bf16,
    indexed_gate_bf16_,
    indexed_gate_rmsnorm_scale_shift_,
    residual_gate_add,
    rmsnorm_scale,
    rmsnorm_tanh_residual,
)
from sglang.kernels.ops.diffusion.common.partial_sum import PartialSum
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _randn(*shape, g, scale=1.0):
    return (scale * torch.randn(*shape, device="cuda", generator=g)).to(torch.bfloat16)


def _partial(shape, g) -> PartialSum:
    # partial sums of one output: roughly opposite halves, so their sum rounds
    return PartialSum(_randn(*shape, g=g), _randn(*shape, g=g, scale=0.5))


# H3's 5376-wide rows (2 full column tiles and a partial one), one partial tile
@pytest.mark.parametrize("rows,hidden", [(37, 1000), (1025, 5376), (4, 2048)])
def test_indexed_gate(rows: int, hidden: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(rows * hidden)
    x = _randn(rows, hidden, g=g)
    gate = _randn(3, hidden, g=g)
    indices = torch.randint(0, 3, (rows,), device="cuda", generator=g)
    partial = _partial((rows, hidden), g)

    ref = indexed_gate_bf16(x, gate, partial.reduce(), indices)
    assert torch.equal(indexed_gate_bf16(x, gate, partial, indices), ref)
    assert torch.equal(indexed_gate_bf16_(x.clone(), gate, partial, indices), ref)


@pytest.mark.parametrize("rows,hidden", [(37, 1000), (1025, 5376)])
def test_indexed_gate_rmsnorm_scale_shift(rows: int, hidden: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(rows + hidden)
    x = _randn(rows, hidden, g=g)
    gate = _randn(3, hidden, g=g)
    shift, scale = _randn(3, 2 * hidden, g=g, scale=0.1).chunk(2, -1)
    weight = 1 + _randn(hidden, g=g, scale=0.1)
    indices = torch.randint(0, 3, (rows,), device="cuda", generator=g)
    partial = _partial((rows, hidden), g)
    assert can_use_indexed_gate_rmsnorm_scale_shift(
        x, gate, partial, weight, shift, scale, indices
    )

    x_ref, x_got = x.clone(), x.clone()
    args = (weight, shift, scale, indices, 1e-6)
    y_ref = indexed_gate_rmsnorm_scale_shift_(x_ref, gate, partial.reduce(), *args)
    y_got = indexed_gate_rmsnorm_scale_shift_(x_got, gate, partial, *args)
    assert torch.equal(x_got, x_ref)
    assert torch.equal(y_got, y_ref)


# Z-Image's 3840-wide rows at 1024x1024, a batch, a width not a power of two
@pytest.mark.parametrize("shape", [(1, 4128, 3840), (2, 96, 3840), (1, 64, 1000)])
def test_rmsnorm_tanh_residual(shape) -> None:
    batch, _, dim = shape
    g = torch.Generator(device="cuda").manual_seed(sum(shape))
    gate = _randn(batch, 1, dim, g=g)
    residual = _randn(*shape, g=g)
    weight = 1 + _randn(dim, g=g, scale=0.1)
    scale_weight = 1 + _randn(dim, g=g, scale=0.1)
    scale = 1 + _randn(batch, 1, dim, g=g, scale=0.1)
    partial = _partial(shape, g)

    ref = rmsnorm_tanh_residual(partial.reduce(), gate, residual, weight, 1e-5)
    assert ref is not None
    got = rmsnorm_tanh_residual(partial, gate, residual, weight, 1e-5)
    assert torch.equal(got, ref)
    out, scaled = rmsnorm_tanh_residual(
        partial, gate, residual, weight, 1e-5, scale_weight, scale
    )
    assert torch.equal(out, ref)
    assert torch.equal(scaled, rmsnorm_scale(ref, scale_weight, scale, 1e-5))


def _gate(mode: str, shape, g):
    batch, tokens, dim = shape
    return {
        "batched_row": lambda: _randn(batch, 1, dim, g=g),
        "row": lambda: _randn(1, 1, dim, g=g),
        "full": lambda: _randn(*shape, g=g),
        "per_token": lambda: _randn(batch, tokens, 1, g=g),
    }[mode]()


# a vectorized width and one that leaves the scalar kernels
@pytest.mark.parametrize("dim", [1024, 1001])
@pytest.mark.parametrize("mode", ["batched_row", "row", "full", "per_token"])
def test_residual_gate_add(mode: str, dim: int) -> None:
    shape = (2, 77, dim)
    g = torch.Generator(device="cuda").manual_seed(dim)
    residual = _randn(*shape, g=g)
    gate = _gate(mode, shape, g)
    partial = _partial(shape, g)
    # the CUDA kernel adds the partials; the eager fallback would pass too
    assert _route(residual, partial.mine, gate, partial.theirs) is not None

    ref = residual_gate_add(residual, partial.reduce(), gate)
    assert torch.equal(residual_gate_add(residual, partial, gate), ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
