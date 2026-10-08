"""CUDA correctness coverage for MoVA's production routed projection."""

import pytest
import torch

from sglang.srt.layers.mova import routed_linear, routed_linear_reference
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_mova_routed_linear_matches_reference_implementation():
    """Exercise the fused-MoE CUDA path with multiple routes and experts."""

    torch.manual_seed(0)
    num_tokens, num_experts = 7, 5
    input_size, output_size, top_k = 128, 96, 2

    hidden_states = torch.randn(
        num_tokens, input_size, device="cuda", dtype=torch.bfloat16
    )
    expert_weights = (
        torch.randn(
            num_experts,
            output_size,
            input_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * input_size**-0.5
    ).contiguous()
    selected_experts = torch.tensor(
        [[0, 3], [1, 4], [2, 0], [3, 1], [4, 2], [0, 4], [2, 3]],
        device="cuda",
        dtype=torch.int32,
    )
    routing_weights = torch.rand(num_tokens, top_k, device="cuda")
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

    with get_context().override_server_args():
        actual = routed_linear(
            hidden_states, expert_weights, routing_weights, selected_experts
        )
    expected = routed_linear_reference(
        hidden_states, expert_weights, routing_weights, selected_experts
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
