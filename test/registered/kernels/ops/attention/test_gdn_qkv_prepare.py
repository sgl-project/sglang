"""GDN preparation preserves norm rounding across layouts and graph replays."""

import pytest
import torch

from sglang.kernels.ops.attention.fla.l2norm import (
    gdn_prefill_qkv_prepare_fwd,
    l2norm_fwd,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("tokens", [0, 1, 33, 2048])
@pytest.mark.parametrize("heads,dim", [((2, 4), 63), ((16, 64), 128)])
def test_normalization_and_layout(dtype, tokens, heads, dim):
    hq, hv = heads
    torch.manual_seed(0)
    packed = torch.randn(
        tokens, (2 * hq + hv + 1) * dim * 2, device="cuda", dtype=dtype
    )[:, ::2]
    q = packed[:, : hq * dim].reshape(tokens, hq, dim)
    k = packed[:, hq * dim : 2 * hq * dim].reshape(tokens, hq, dim)
    v = packed[:, 2 * hq * dim : (2 * hq + hv) * dim].reshape(tokens, hv, dim)
    for contiguous in (False, True):
        inputs = [x.contiguous() if contiguous else x for x in (q, k, v)]
        actual = gdn_prefill_qkv_prepare_fwd(*inputs)
        expected = [
            l2norm_fwd(x.contiguous()) if i < 2 and tokens else x.contiguous()
            for i, x in enumerate(inputs)
        ]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert a.is_contiguous()
        if contiguous:
            assert actual[2].data_ptr() == inputs[2].data_ptr()


def test_graph_replay_reads_updated_inputs():
    packed = torch.randn(33, 1024, device="cuda", dtype=torch.bfloat16)
    q, k, v = (
        packed[:, :256].view(33, 2, 128),
        packed[:, 256:512].view(33, 2, 128),
        packed[:, 512:].view(33, 4, 128),
    )
    gdn_prefill_qkv_prepare_fwd(q, k, v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gdn_prefill_qkv_prepare_fwd(q, k, v)
    for scale in (0.0, 1e-4, 2.0):
        packed.normal_().mul_(scale)
        graph.replay()
        expected = (
            l2norm_fwd(q.contiguous()),
            l2norm_fwd(k.contiguous()),
            v.contiguous(),
        )
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
