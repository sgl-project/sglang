import sys

import pytest
import torch

from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=4, stage="jit-kernel-unit", runner_config="amd")


def test_ep_requires_topk_ids():
    with pytest.raises(AssertionError, match="topk_ids is required"):
        moe_fused_mul_sum(
            inputs=torch.empty((1, 1, 1), device="cuda"),
            topk_weights=torch.empty((1, 1), device="cuda"),
            is_ep=True,
        )


def test_ep_masking_skips_invalid_nan_slots_and_accumulates_in_fp32():
    expert_outputs = torch.tensor(
        [
            [[2.0, 4.0], [torch.nan, torch.nan]],
            [[torch.nan, torch.nan], [6.0, 8.0]],
            [[torch.nan, torch.nan], [torch.nan, torch.nan]],
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_weights = torch.tensor(
        [[0.25, 0.0], [0.0, 0.5], [0.0, 0.0]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_ids = torch.tensor(
        [[0, -1], [-1, 1], [-1, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    output = torch.empty((3, 2), dtype=torch.float32, device="cuda")

    result = moe_fused_mul_sum(
        inputs=expert_outputs,
        topk_weights=topk_weights,
        outputs=output,
        topk_ids=topk_ids,
        is_ep=True,
    )

    assert result is output
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result,
        torch.tensor(
            [[0.5, 1.0], [3.0, 4.0], [0.0, 0.0]],
            dtype=torch.float32,
            device="cuda",
        ),
        rtol=0,
        atol=0,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
