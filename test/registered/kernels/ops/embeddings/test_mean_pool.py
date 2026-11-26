import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.pooler import PoolingType, pool_hidden_states
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lengths", [(9, 3), (2048, 8, 0)])
def test_packed_mean_cuda_graph(dtype, lengths):
    # Include padding, a strided view, and a partial tile in the hidden dimension.
    torch.manual_seed(42)
    hidden = (torch.randn(sum(lengths) + 16, 130, device="cuda") + 2).to(dtype)[:, ::2]
    device_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    batch = SimpleNamespace(extend_seq_lens=device_lengths)
    pool_hidden_states(PoolingType.MEAN, hidden, batch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = pool_hidden_states(PoolingType.MEAN, hidden, batch)

    for order in (lengths, lengths[::-1]):
        device_lengths.copy_(torch.tensor(order, device="cuda", dtype=torch.int32))
        graph.replay()
        reference = torch.stack(
            [row.double().mean(0) for row in hidden[: sum(order)].split(order)]
        ).to(dtype)
        torch.testing.assert_close(actual, reference, equal_nan=True)


def test_mean_pool_preserves_native_autograd():
    hidden = torch.randn(7, 11, device="cuda", requires_grad=True)
    batch = SimpleNamespace(extend_seq_lens=torch.tensor([2, 5], device="cuda"))
    pooled = pool_hidden_states(PoolingType.MEAN, hidden, batch)
    grad = torch.autograd.grad(pooled.sum(), hidden)[0]
    torch.testing.assert_close(grad[:2], torch.full_like(grad[:2], 0.5))
    torch.testing.assert_close(grad[2:], torch.full_like(grad[2:], 0.2))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
