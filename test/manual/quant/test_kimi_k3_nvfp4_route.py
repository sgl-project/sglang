"""SM103 numerical and graph-replay checks for fused NVFP4 routing."""

import pytest
import torch
from flashinfer import fp4_quantize

from sglang.kernels.ops.moe.moe_route_nvfp4_fused import route_nvfp4_fused
from sglang.kernels.ops.moe.moe_route_radix import route_radix

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="Requires SM103",
)


@pytest.mark.parametrize("m", [1, 8, 64])
@pytest.mark.parametrize("variant", ["random", "strided_fp32", "ties_zero"])
def test_route_quant_bytes_and_replay(m, variant):
    torch.manual_seed(4568)
    scores = torch.randn(m, 896, device="cuda", dtype=torch.float32)
    bias = torch.randn(896, device="cuda", dtype=torch.float32) * 0.01
    strided = variant == "strided_fp32"
    x = torch.randn(
        m,
        7168 if strided else 3584,
        device="cuda",
        dtype=torch.float32 if strided else torch.bfloat16,
    )[:, :3584]
    scale = torch.tensor([268.8], device="cuda", dtype=torch.float32)
    if variant == "ties_zero":
        scores.zero_()
        bias.zero_()
        x.zero_()

    def baseline():
        weights, ids = route_radix(scores, bias, 16, True, 1.0, False)
        q, scales = fp4_quantize(
            x.to(torch.bfloat16).contiguous(), scale, 16, False, False
        )
        return (
            weights,
            ids,
            q.reshape(m, 1792),
            scales.view(torch.uint8).reshape(m, 224),
        )

    def candidate():
        weights, ids, _, q, scales = route_nvfp4_fused(
            scores, bias, x, scale, 16, True, 1.0, False
        )
        return weights, ids, q, scales

    for _ in range(3):
        candidate()
    for actual, expected in zip(candidate(), baseline()):
        assert torch.equal(actual, expected)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate()
    # Exercise reuse of the captured input/output buffers with different data.
    x.mul_(0.875)
    scores.add_(0.03125)
    graph.replay()
    torch.cuda.synchronize()
    for actual, expected in zip(captured, baseline()):
        assert torch.equal(actual, expected)
