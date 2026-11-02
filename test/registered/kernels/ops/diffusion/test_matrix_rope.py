# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import apply_matrix_rope
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def reference(x, rope):
    pairs = x.reshape(*x.shape[:-1], -1, 1, 2).float()
    return (rope * pairs).sum(-1).reshape_as(x).to(x.dtype)


def inputs(shape, dtype, broadcast_batch=True):
    torch.manual_seed(42)
    batch, seq, _, dim = shape
    x = torch.randn(shape, device="cuda", dtype=dtype)
    rope = torch.randn(
        1 if broadcast_batch else batch,
        seq + 3,
        1,
        dim // 2,
        2,
        2,
        device="cuda",
        dtype=torch.float32,
    )[:, 3:]
    return x, rope


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("broadcast_batch", [False, True])
@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 32), (2, 17, 3, 64), (2, 257, 4, 192), (1, 4096, 32, 128)]
)
def test_matrix_rope_bitexact(shape, dtype, broadcast_batch):
    x, rope = inputs(shape, dtype, broadcast_batch)
    original = x.clone()
    actual = apply_matrix_rope(x, rope)
    expected = reference(x, rope)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(x, original)
    assert actual.is_contiguous()


@pytest.mark.parametrize("dtype", [None, torch.float16, torch.bfloat16])
def test_matrix_rope_compile_and_graph_replay(dtype):
    x, rope = inputs((2, 257, 8, 128), torch.float32)
    compiled = torch.compile(apply_matrix_rope, fullgraph=True)
    assert torch.equal(compiled(x, rope, dtype), reference(x.to(dtype=dtype), rope))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = apply_matrix_rope(x, rope, dtype)
    x.normal_()
    rope.normal_()
    graph.replay()
    assert torch.equal(result, reference(x.to(dtype=dtype), rope))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("broadcast_batch", [False, True])
@pytest.mark.parametrize("scale", [1e-4, 1.0, 100.0])
def test_matrix_rope_cast_bitexact(dtype, input_dtype, broadcast_batch, scale):
    x, rope = inputs((2, 257, 8, 128), input_dtype, broadcast_batch)
    x *= scale
    original = x.clone()
    expected = reference(x.to(dtype), rope)
    actual = apply_matrix_rope(x, rope, dtype)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(x, original)
    assert actual.dtype == dtype
    assert actual.is_contiguous()


def test_matrix_rope_empty_input():
    x, rope = inputs((1, 0, 8, 128), torch.bfloat16)
    assert apply_matrix_rope(x, rope).shape == x.shape


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
