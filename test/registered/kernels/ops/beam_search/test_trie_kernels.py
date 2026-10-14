"""Grouped SID projection must preserve dense logits across graph replays."""

import pytest
import torch

from sglang.srt.beam_search.trie_kernels import project_active_codebook_logits
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
        and hasattr(torch, "_grouped_mm")
    ),
    reason="SM90 grouped GEMM required",
)


@pytest.mark.parametrize("num_rows", [1024, 8193])
def test_active_codebook_projection_is_exact_and_graph_safe(num_rows):
    device = torch.device("cuda")
    hidden_size, codebook_size = 128, 256
    hidden_states = torch.randn(
        (num_rows, hidden_size), device=device, dtype=torch.bfloat16
    )
    weights = torch.randn(
        (3 * codebook_size, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    depths = torch.arange(num_rows, device=device) % 3

    project_active_codebook_logits(hidden_states, weights, depths, codebook_size)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = project_active_codebook_logits(
            hidden_states, weights, depths, codebook_size
        )

    for next_depths in (depths.clone(), torch.ones_like(depths)):
        depths.copy_(next_depths)
        graph.replay()
        all_logits = torch.matmul(hidden_states, weights.T).view(
            num_rows, 3, codebook_size
        )
        expected = all_logits[torch.arange(num_rows, device=device), next_depths]
        assert actual is not None
        assert torch.equal(actual, expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
