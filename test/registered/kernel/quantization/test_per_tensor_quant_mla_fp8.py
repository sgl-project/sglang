import sys

import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import per_tensor_quant_mla_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def quantize(x, count=None):
    return per_tensor_quant_mla_fp8(
        x,
        torch.zeros(1, dtype=torch.float32, device=x.device),
        num_valid_tokens=count,
    )


@pytest.mark.parametrize("head_size", [192, 512])
@pytest.mark.parametrize("padding", [1000.0, float("nan")])
def test_padding_does_not_change_real_tokens(head_size, padding):
    # MLA passes a transposed, non-contiguous [heads, tokens, dim] view.
    x = torch.randn(64, 4, head_size, device="cuda", dtype=torch.bfloat16).transpose(
        0, 1
    )
    count = torch.tensor([50], device="cuda", dtype=torch.int32)
    x[:, 50:] = padding
    expected, expected_scale = quantize(x[:, :50])
    actual, actual_scale = quantize(x, count)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    assert torch.equal(actual[:, :50].view(torch.uint8), expected.view(torch.uint8))
    assert torch.count_nonzero(actual[:, 50:].float()) == 0


@pytest.mark.parametrize("head_size", [192, 512])
def test_graph_replay_uses_live_count(head_size):
    count = torch.zeros(1, device="cuda", dtype=torch.int32)
    graphs = {}
    # Exercise multiple buckets and return to earlier ones, sharing the count.
    for bucket in [64, 128]:
        x = torch.zeros(
            bucket, 4, head_size, device="cuda", dtype=torch.bfloat16
        ).transpose(0, 1)
        count.fill_(bucket)
        quantize(x, count)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, scale = quantize(x, count)
        graphs[bucket] = (graph, x, output, scale)

    for bucket, valid in [(64, 50), (128, 114), (64, 33), (64, 64), (128, 0), (64, 50)]:
        graph, x, output, scale = graphs[bucket]
        x.normal_()
        x[:, valid:] = float("nan")
        count.fill_(valid)
        graph.replay()
        if valid:
            expected, expected_scale = quantize(x[:, :valid])
            torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
            assert torch.equal(
                output[:, :valid].view(torch.uint8), expected.view(torch.uint8)
            )
        else:
            assert torch.isfinite(scale).all() and (scale > 0).all()
        assert torch.count_nonzero(output[:, valid:].float()) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
