import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.sampling.watermark import WatermarkState
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def test_repeated_context_and_greedy_bypass():
    """A repeated context must sample normally instead of forcing forever."""
    device = "cuda"
    state = WatermarkState(
        max_num_reqs=2,
        context_window=2,
        max_contexts_per_req=8,
        key="0123456789abcdef",
        device=device,
    )
    req_pool_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    state.init_from_prompt(req_pool_indices, [[10, 11], [20, 21]])
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((2, 1), device=device),
        top_ks=torch.tensor([64, 1], device=device, dtype=torch.int32),
        top_ps=torch.ones(2, device=device),
        min_ps=torch.zeros(2, device=device),
    )

    first_logits = torch.zeros((2, 64), device=device)
    state.force(first_logits, req_pool_indices, sampling_info)

    assert torch.isfinite(first_logits[0]).sum().item() == 1
    assert torch.equal(first_logits[1], torch.zeros(64, device=device))
    assert state.num_watermarked_contexts.tolist() == [1, 0]

    state.append(
        req_pool_indices[:1], torch.tensor([10], device=device, dtype=torch.int32)
    )
    state.append(
        req_pool_indices[:1], torch.tensor([11], device=device, dtype=torch.int32)
    )
    repeated_logits = torch.zeros((2, 64), device=device)
    state.force(repeated_logits, req_pool_indices, sampling_info)

    assert torch.equal(repeated_logits, torch.zeros_like(repeated_logits))
    assert state.num_watermarked_contexts.tolist() == [1, 0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
