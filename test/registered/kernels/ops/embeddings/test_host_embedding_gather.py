import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.embeddings.host_embedding_gather import host_embedding_gather
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test."
)


def _run_case(input_ids, table):
    # The gather must be bit-exact with a plain device-side lookup.
    expected = F.embedding(input_ids.long(), table.cuda())
    actual = host_embedding_gather(input_ids, table)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("hidden_dim", [7, 128, 5120])
def test_host_embedding_gather(dtype, input_dtype, hidden_dim):
    table = torch.randn((64, hidden_dim), dtype=dtype).pin_memory()
    input_ids = torch.tensor([0, 63, 5, 5, 17], dtype=input_dtype, device="cuda")
    _run_case(input_ids, table)
    # Any input shape is allowed; the hidden dim is appended.
    _run_case(input_ids.view(5, 1).expand(5, 3).contiguous(), table)
    # Non-contiguous ids are handled by the wrapper.
    _run_case(torch.arange(0, 64, device="cuda", dtype=input_dtype)[::4], table)


def test_host_embedding_gather_empty_input():
    table = torch.randn((8, 16), dtype=torch.bfloat16).pin_memory()
    ids = torch.empty((0,), dtype=torch.int64, device="cuda")
    out = host_embedding_gather(ids, table)
    assert out.shape == (0, 16) and out.dtype == torch.bfloat16 and out.is_cuda


def test_host_embedding_gather_cuda_graph():
    # The whole point of reading the host table from the GPU: no host-side
    # work per lookup, so the launch can live inside a captured graph.
    table = torch.randn((256, 128), dtype=torch.bfloat16).pin_memory()
    static_ids = torch.zeros((9,), dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            host_embedding_gather(static_ids, table)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = host_embedding_gather(static_ids, table)
    for _ in range(3):
        ids = torch.randint(0, 256, (9,), device="cuda")
        static_ids.copy_(ids)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            static_out, F.embedding(ids, table.cuda()), rtol=0, atol=0
        )


def test_host_embedding_gather_rejects_pageable_table():
    # A pageable table would be an illegal address for the GPU; refuse it
    # before launching anything.
    table = torch.randn((8, 16), dtype=torch.bfloat16)
    ids = torch.zeros((2,), dtype=torch.int64, device="cuda")
    with pytest.raises(AssertionError, match="pinned"):
        host_embedding_gather(ids, table)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
