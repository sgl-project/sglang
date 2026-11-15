"""MiniMax-H3's indexed scale/shift and gate kernels against the same arithmetic in
torch, byte for byte, across row widths the column tiles split differently. The fp32
variants serve serialized INT8 checkpoints, whose AdaLN tables stay in fp32."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    indexed_gate_bf16,
    indexed_gate_bf16_,
    indexed_gate_fp32,
    indexed_gate_fp32_,
    indexed_scale_shift_bf16_,
    indexed_scale_shift_fp32_,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# one tile, a partial tile, whole tiles, and MiniMax-H3's 5376 (2 full + 1 partial)
HIDDEN = [1000, 2048, 4096, 5376]
ROWS = [1, 37, 1025]


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).float()


def _inputs(rows: int, hidden: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, hidden, device="cuda", generator=g).to(torch.bfloat16)
    other = torch.randn(rows, hidden, device="cuda", generator=g).to(torch.bfloat16)
    shift = torch.randn(3, hidden, device="cuda", generator=g).to(torch.bfloat16)
    scale = (0.1 * torch.randn(3, hidden, device="cuda", generator=g)).to(
        torch.bfloat16
    )
    indices = torch.randint(0, 3, (rows,), device="cuda", generator=g)
    return x, other, shift, scale, indices


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("rows", ROWS)
def test_indexed_scale_shift_matches_torch(rows: int, hidden: int) -> None:
    x, _, shift, scale, indices = _inputs(rows, hidden, rows * hidden)
    one_plus_scale = _bf16(1.0 + scale[indices].float())
    ref = (_bf16(x.float() * one_plus_scale) + shift[indices].float()).to(
        torch.bfloat16
    )
    got = indexed_scale_shift_bf16_(x.clone(), shift, scale, indices)
    assert torch.equal(got, ref)


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("rows", ROWS)
def test_indexed_gate_matches_torch(rows: int, hidden: int) -> None:
    x, other, gate, _, indices = _inputs(rows, hidden, rows * hidden + 1)
    ref = (x.float() + _bf16(gate[indices].float() * other.float())).to(torch.bfloat16)
    assert torch.equal(indexed_gate_bf16(x, gate, other, indices), ref)
    assert torch.equal(indexed_gate_bf16_(x.clone(), gate, other, indices), ref)


def _fp32_table(rows: int, hidden: int, seed: int) -> torch.Tensor:
    """A [rows, hidden] slice of a [rows, 6 * hidden] AdaLN table: row stride > width."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    table = torch.randn(rows, 6 * hidden, device="cuda", generator=g)
    return table[:, 2 * hidden : 3 * hidden]


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("rows", ROWS)
def test_indexed_scale_shift_fp32_matches_torch(rows: int, hidden: int) -> None:
    x, _, _, _, indices = _inputs(rows, hidden, rows * hidden + 2)
    shift = _fp32_table(3, hidden, rows + 3)
    scale = 0.1 * _fp32_table(3, hidden, rows + 4)
    # A product whose rounding differs once it is contracted into an FMA.
    scale[indices[0], 0] = -0.9999999403953552
    shift[indices[0], 0] = 1.234567e-7
    ref = (x.float() * (1.0 + scale[indices]) + shift[indices]).to(torch.bfloat16)
    assert torch.equal(indexed_scale_shift_fp32_(x.clone(), shift, scale, indices), ref)


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("rows", ROWS)
def test_indexed_gate_fp32_matches_torch(rows: int, hidden: int) -> None:
    x, other, _, _, indices = _inputs(rows, hidden, rows * hidden + 5)
    gate = _fp32_table(3, hidden, rows + 6)
    x[0, 0], other[0, 0] = -0.1484375, 14.25
    gate[indices[0], 0] = 0.01041664183139801
    ref = (x.float() + gate[indices] * other.float()).to(torch.bfloat16)
    original = x.clone()
    assert torch.equal(indexed_gate_fp32(x, gate, other, indices), ref)
    assert torch.equal(x, original)
    assert torch.equal(indexed_gate_fp32_(x.clone(), gate, other, indices), ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
