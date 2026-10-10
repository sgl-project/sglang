"""MiniMax-H3's fused RMSNorm + indexed AdaLN (quality lossless/high) against the
same math in fp32 with one rounding, and against the reference rounding chain."""

import sys

import pytest
import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion import (
    can_use_rmsnorm_indexed_scale_shift,
    indexed_scale_shift_bf16_,
    rmsnorm_indexed_scale_shift,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

EPS = 1e-5


def _inputs(rows: int, hidden: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (2 * torch.randn(rows, hidden, device="cuda", generator=g)).to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(hidden, device="cuda", generator=g)).to(
        torch.bfloat16
    )
    # H3's shift/scale are chunks of one [M, 6 * hidden] AdaLN output
    adaln = (0.1 * torch.randn(3, 6 * hidden, device="cuda", generator=g)).to(
        torch.bfloat16
    )
    shift, scale = adaln[:, :hidden], adaln[:, hidden : 2 * hidden]
    indices = torch.randint(0, 3, (rows,), device="cuda", generator=g)
    return x, weight, shift, scale, indices


@pytest.mark.parametrize("hidden", [1024, 5376])
@pytest.mark.parametrize("rows", [1, 37, 1025])
def test_matches_fp32_single_rounding(rows: int, hidden: int) -> None:
    x, weight, shift, scale, indices = _inputs(rows, hidden, rows + hidden)
    assert can_use_rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices)
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()
    ref = normed * (1 + scale[indices].float()) + shift[indices].float()
    got = rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, EPS)
    # only the reduction order differs: at most one bf16 rounding step apart
    torch.testing.assert_close(got.float(), ref, rtol=2**-7, atol=2**-7)


def test_stays_within_rounding_of_the_reference_chain() -> None:
    x, weight, shift, scale, indices = _inputs(1025, 5376, 7)
    norm = nn.RMSNorm(5376, eps=EPS, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)
    chain = indexed_scale_shift_bf16_(norm(x), shift, scale, indices)
    got = rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, EPS)
    torch.testing.assert_close(got.float(), chain.float(), rtol=2**-6, atol=2**-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
