"""Unit test for ``fp8_per_token_to_per_tensor_quant_triton``.

The hidden size is only guaranteed to be a multiple of the scale group (128),
rows past ``masked_m`` must stay untouched, and every launch geometry must
produce the same bytes.
"""

import pytest
import torch
import triton

from sglang.kernels.ops.moe.ep_moe_kernels import (
    _fp8_per_token_quant_to_per_tensor_quant_kernel,
    fp8_per_token_to_per_tensor_quant_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

dev = "cuda"
FP8 = torch.float8_e4m3fn
K_SCALE_BLOCK_SIZE = 128
# Every value the kernel can produce below is a multiple of 0.25, so this
# sentinel cannot be matched by a kernel that wrongly writes a padding row.
SENTINEL = 0.375
OUTPUT_SCALE = 2.0


def _build(num_experts, m, k, seed, column_major_scales=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Integers in [-8, 8] with power-of-two group scales keep every intermediate
    # exactly representable in e4m3, so the reference matches bit-for-bit.
    x = torch.randint(-8, 9, (num_experts, m, k), generator=g).float()
    exps = torch.randint(-1, 2, (num_experts, m, k // K_SCALE_BLOCK_SIZE), generator=g)
    x_scale = torch.pow(2.0, exps.float()).to(dev)
    if column_major_scales:
        # DeepEP returns the last two scale dims column-major (for TMA).
        x_scale = x_scale.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    return x.to(dev).to(FP8), x_scale


def _ref(x, x_scale):
    dequant = x.float() * x_scale.repeat_interleave(K_SCALE_BLOCK_SIZE, dim=2)
    return (dequant * (1.0 / OUTPUT_SCALE)).to(FP8)


def _assert_output(output, x, x_scale, masked):
    ref = _ref(x, x_scale)
    for e, valid in enumerate(masked):
        torch.testing.assert_close(
            output[e, :valid].float(), ref[e, :valid].float(), rtol=0, atol=0
        )
        # Rows past masked_m must stay as the caller left them.
        padding = output[e, valid:].float()
        torch.testing.assert_close(
            padding, torch.full_like(padding, SENTINEL), rtol=0, atol=0
        )


def _run_and_check(num_experts, m, k, masked, expected_rows, column_major_scales):
    x, x_scale = _build(
        num_experts, m, k, seed=k + num_experts, column_major_scales=column_major_scales
    )
    masked_m = torch.tensor(masked, dtype=torch.int32, device=dev)
    output_scale = torch.tensor([OUTPUT_SCALE], dtype=torch.float32, device=dev)
    output = torch.full((num_experts, m, k), SENTINEL, device=dev).to(FP8)

    fp8_per_token_to_per_tensor_quant_triton(
        x=x,
        x_scale=x_scale,
        masked_m=masked_m,
        output_scale=output_scale,
        output=output,
        expected_rows=expected_rows,
    )

    _assert_output(output, x, x_scale, masked)


# 7168: fills every tile of scale groups (the DeepSeek-V3 hidden size).
# 3584 / 1152: only 128-aligned, so the last tile is partially masked.
@pytest.mark.parametrize("k", [7168, 3584, 1152])
# None: shape-independent grid; 4 and 64: both ends of the m-grid heuristic.
@pytest.mark.parametrize("expected_rows", [None, 4, 64])
@pytest.mark.parametrize("column_major_scales", [False, True])
def test_masked_rows_and_group_tail(k, expected_rows, column_major_scales):
    _run_and_check(
        num_experts=4,
        m=48,
        k=k,
        masked=[0, 1, 17, 48],
        expected_rows=expected_rows,
        column_major_scales=column_major_scales,
    )


# Row estimates chosen so the expert-count cap binds: 40 experts cap at 16
# programs, 128 at 8; uncapped these would be 32 and 16.
@pytest.mark.parametrize("num_experts,expected_rows", [(40, 32), (128, 16)])
def test_many_experts(num_experts, expected_rows):
    m = 48
    # Every expert gets a different row count, as a real dispatch would.
    masked = [(e * 7) % (m + 1) for e in range(num_experts)]
    _run_and_check(
        num_experts=num_experts,
        m=m,
        k=3584,
        masked=masked,
        expected_rows=expected_rows,
        column_major_scales=True,
    )


# The wrapper only launches the running vendor's tile, so drive the kernel
# directly across every width either vendor can pick, plus the degenerate
# single-group tile.
@pytest.mark.parametrize("g_block", [1, 8, 16, 32])
@pytest.mark.parametrize("k", [7168, 3584])
@pytest.mark.parametrize("m_grid", [1, 4, 32])
# 0 sends every row to the shared overflow path, 48 keeps every row on its own
# expert, and 4 splits the batch across both.
@pytest.mark.parametrize("row_cap", [0, 4, 48])
def test_every_launch_geometry_agrees(g_block, k, m_grid, row_cap):
    num_experts, m = 4, 48
    masked = [0, 1, 17, 48]
    x, x_scale = _build(num_experts, m, k, seed=k + g_block, column_major_scales=True)
    masked_m = torch.tensor(masked, dtype=torch.int32, device=dev)
    output_scale = torch.tensor([OUTPUT_SCALE], dtype=torch.float32, device=dev)
    output = torch.full((num_experts, m, k), SENTINEL, device=dev).to(FP8)

    num_groups = k // K_SCALE_BLOCK_SIZE
    grid = (triton.cdiv(num_groups, g_block), m_grid, num_experts)
    _fp8_per_token_quant_to_per_tensor_quant_kernel[grid](
        x,
        x_scale,
        *x_scale.stride(),
        masked_m,
        output_scale,
        output,
        m,
        k,
        num_experts,
        row_cap,
        K_SCALE_BLOCK_SIZE=K_SCALE_BLOCK_SIZE,
        G_BLOCK_SIZE=g_block,
        HAS_G_TAIL=(num_groups % g_block != 0),
        EXPERT_BLOCK=triton.next_power_of_2(num_experts),
        num_warps=4,
    )

    _assert_output(output, x, x_scale, masked)


# Experts with no live rows must be stepped over by the prefix-sum mapping,
# the case most likely to be off by one.
@pytest.mark.parametrize(
    "masked", [[0, 0, 0, 0], [0, 5, 0, 7], [9, 0, 0, 0], [0, 0, 0, 9]]
)
def test_experts_with_no_rows_are_skipped(masked):
    _run_and_check(
        num_experts=4,
        m=48,
        k=3584,
        masked=masked,
        expected_rows=2,
        column_major_scales=True,
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
