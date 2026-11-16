import sys

import pytest
import torch

from sglang.kernels.ops.quantization.act_quant_triton import act_quant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("tokens", [0, 1, 17, 1024])
@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"])
def test_quant_head_gate_matches_bf16_reference(tokens, scale_fmt):
    from sglang.kernels.ops.attention.dsa.head_gate import indexer_q_quant_head_gate

    torch.manual_seed(42)
    heads, width = 32, 128
    query = torch.randn(tokens, heads, width, device="cuda", dtype=torch.bfloat16)
    # Merged projection slices need not have a contiguous row stride.
    gate = torch.randn(tokens, 160, device="cuda", dtype=torch.bfloat16)[:, 128:]
    if tokens:
        query[0, 0] = 0
        query[0, 1] *= 1e-6
    actual_q, actual_w = indexer_q_quant_head_gate(query, gate, width**-0.5, scale_fmt)
    if not tokens:
        assert actual_q.shape == query.shape
        assert actual_w.shape == (0, heads, 1)
        return
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_q, actual_w = indexer_q_quant_head_gate(
            query, gate, width**-0.5, scale_fmt
        )
    for factor in [1.0, -0.25]:
        query.mul_(factor)
        gate.mul_(factor)
        expected_q, scale = act_quant(query, 128, scale_fmt)
        expected_w = (gate * heads**-0.5).float().unsqueeze(-1) * scale * width**-0.5
        graph.replay()
        torch.testing.assert_close(
            actual_q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(actual_w, expected_w, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
