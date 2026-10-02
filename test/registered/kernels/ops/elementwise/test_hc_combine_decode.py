"""Decode HC fusion preserves the two BF16 rounding boundaries under replay."""

import pytest
import torch

from sglang.kernels.ops.elementwise.hc_combine import hc_combine_split
from sglang.kernels.ops.elementwise.hc_combine_decode import (
    hc_combine_apply,
    hc_combine_apply_norm,
    hc_combine_gate,
)
from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Decode specialization is enabled on SM100/SM103",
)


@pytest.mark.parametrize("rows", [1, 2, 4])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@torch.inference_mode()
def test_replay_matches_unfused(rows, eps):
    torch.manual_seed(42)
    block = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
    normed = torch.randn_like(residual)
    inject = torch.randn(4, 10240, device="cuda", dtype=torch.bfloat16) * 0.02
    weight = torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.1

    def fused():
        partials = hc_combine_gate(normed, inject)
        applied = hc_combine_apply(block, residual, partials)
        combined, normalized = hc_combine_apply_norm(
            block, residual, partials, weight, eps
        )
        return applied, combined, normalized

    for _ in range(3):
        fused()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        applied, combined, normalized = fused()
    for _ in range(5):
        block.normal_()
        residual.normal_()
        normed.normal_()
        inject.normal_(std=0.02)
        weight.normal_(std=0.1)
        graph.replay()
        expected = hc_combine_split(block, residual, normed, inject, 4, 2560)
        expected_norm = grouped_gemma_rmsnorm(expected, weight, 2560, eps)
        for result, reference in (
            (applied, expected),
            (combined, expected),
            (normalized, expected_norm),
        ):
            assert torch.equal(result.view(torch.int16), reference.view(torch.int16))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
