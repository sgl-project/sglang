# SPDX-License-Identifier: Apache-2.0
"""Correctness coverage for MiniMax H3 indexed RMSNorm+AdaLN fusion."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.triton.indexed_modulation import (
    indexed_scale_shift_bf16_,
)
from sglang.kernels.ops.diffusion.triton.indexed_rmsnorm_adaln import (
    can_use_fused_indexed_rmsnorm_adaln,
    fused_indexed_rmsnorm_adaln,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=8, suite="nightly-amd-kernel-1-gpu", nightly=True)

HIDDEN_SIZE = 5376
EPS = 1e-5


@pytest.fixture(autouse=True)
def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm GPU required")
    torch.cuda.manual_seed(0)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows,adaln_rows", [(7, 3), (128, 6)])
def test_fused_indexed_rmsnorm_adaln_matches_h3_chain(
    rows: int, adaln_rows: int, index_dtype: torch.dtype
) -> None:
    x = torch.randn(rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    # H3 obtains these tensors by chunking the 6H AdaLN projection. They are
    # unit-stride over H but not fully contiguous because the row stride is 6H.
    adaln = (
        torch.randn(adaln_rows, 6 * HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        * 0.1
    )
    shift, scale, *_ = adaln.chunk(6, dim=-1)
    assert not shift.is_contiguous() and shift.stride(1) == 1
    assert not scale.is_contiguous() and scale.stride(1) == 1
    indices = (
        torch.arange(rows, device="cuda", dtype=index_dtype) % adaln_rows
    ).contiguous()

    expected = F.rms_norm(x, (HIDDEN_SIZE,), weight, EPS)
    expected = indexed_scale_shift_bf16_(expected, shift, scale, indices)
    actual = fused_indexed_rmsnorm_adaln(x, weight, shift, scale, indices, EPS)

    # The Triton and eager RMSNorm reductions are not bit-exact, but their
    # outputs must remain within BF16-level rounding error.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_fused_indexed_rmsnorm_adaln_guards_layout_and_shape() -> None:
    x = torch.randn(4, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(2, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn_like(shift)
    indices = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.long)

    assert can_use_fused_indexed_rmsnorm_adaln(x, weight, shift, scale, indices)
    packed = torch.randn(2, 6 * HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    strided_shift, strided_scale, *_ = packed.chunk(6, dim=-1)
    assert can_use_fused_indexed_rmsnorm_adaln(
        x, weight, strided_shift, strided_scale, indices
    )
    noncontiguous_x = x.t().contiguous().t()
    assert noncontiguous_x.shape == x.shape and not noncontiguous_x.is_contiguous()
    assert not can_use_fused_indexed_rmsnorm_adaln(
        noncontiguous_x, weight, shift, scale, indices
    )
    assert not can_use_fused_indexed_rmsnorm_adaln(
        x, weight.float(), shift, scale, indices
    )
    assert not can_use_fused_indexed_rmsnorm_adaln(
        x, weight, shift, scale, indices[:-1]
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
