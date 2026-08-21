import sys

import pytest
import torch

from sglang.kernels.ops.moe.pack_topk_ids import PackTopkIds
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(1, 4), (7, 8), (1025,)])
def test_pack_topk_ids_matches_reference(dtype: torch.dtype, shape: tuple) -> None:
    torch.manual_seed(42)
    topk_ids = torch.randint(0, 384, shape, dtype=dtype, device="cuda")
    topk_weights = torch.rand(shape, dtype=torch.float32, device="cuda")

    expected = PackTopkIds.vanilla(topk_ids, topk_weights)
    actual = PackTopkIds.execute(topk_ids, topk_weights)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.int32


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_pack_topk_ids_cuda_graph(dtype: torch.dtype) -> None:
    topk_ids = torch.randint(0, 384, (32, 8), dtype=dtype, device="cuda")
    topk_weights = torch.rand((32, 8), dtype=torch.float32, device="cuda")

    # Warm the dtype-specific Triton specialization before CUDA-graph capture.
    PackTopkIds.execute(topk_ids, topk_weights)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = PackTopkIds.execute(topk_ids, topk_weights)
    graph.replay()
    torch.cuda.synchronize()

    expected = PackTopkIds.vanilla(topk_ids, topk_weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pack_topk_ids_rejects_non_integer_ids() -> None:
    topk_ids = torch.zeros((1, 4), dtype=torch.float32, device="cuda")
    topk_weights = torch.zeros_like(topk_ids)

    with pytest.raises(AssertionError, match="must be int32 or int64"):
        PackTopkIds.execute(topk_ids, topk_weights)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
