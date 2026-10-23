import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("bs", [0, 1, 129])
@pytest.mark.parametrize("width", [1, 17, 2560])
@pytest.mark.parametrize("strided", [False, True])
def test_relay_scatter_graph(bs, width, strided):
    from sglang.kernels.ops.speculative.gather_spec_extras import scatter_spec_extras

    slots = bs + 3
    indices = torch.randperm(slots, device="cuda")[:bs].repeat_interleave(2)[::2]
    if bs:
        indices[0] -= slots
    pairs = []
    for n, src_type, dst_type in [
        (1, torch.int32, torch.int64),
        (3, torch.float32, torch.float32),
        (width, torch.float32, torch.bfloat16),
        (7, torch.int64, torch.int64),
    ]:
        stride = 2 if strided else 1
        src = (
            torch.arange(bs * n * stride, device="cuda")
            .reshape(bs, n * stride)[:, ::stride]
            .to(src_type)
        )
        dst = torch.full((slots, n), -7, dtype=dst_type, device="cuda")
        pairs.append((dst, src))
    scatter_spec_extras(indices, pairs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scatter_spec_extras(indices, pairs)
    for replay in range(3):
        expected = []
        for dst, src in pairs:
            dst.fill_(-7)
            src.add_(replay)
            ref = dst.clone()
            ref[indices] = src.to(dst.dtype)
            expected.append(ref)
        graph.replay()
        for (actual, _), ref in zip(pairs, expected):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
