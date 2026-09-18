# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_rmsnorm_preserve_reduction,
    rmsnorm_preserve_reduction,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def reference(x, weight, eps):
    value = x.float()
    variance = value.pow(2).mean(dim=-1, keepdim=True)
    return weight * (value * torch.rsqrt(variance + eps)).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 128), (2, 17, 3, 128), (1, 4096, 32, 128)])
@pytest.mark.parametrize("scale,eps", [(1e-4, 1e-6), (1.0, 1e-6), (100.0, 1e-5)])
def test_preserves_native_reduction_and_rounding(dtype, shape, scale, eps):
    torch.manual_seed(42)
    x = (torch.randn(shape, device="cuda") * scale).to(dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    assert can_use_rmsnorm_preserve_reduction(x, weight)
    actual = rmsnorm_preserve_reduction(x, weight, eps)
    torch.testing.assert_close(actual, reference(x, weight, eps), atol=0, rtol=0)


def test_layout_guards_and_offset():
    x = torch.randn(259, 128, device="cuda", dtype=torch.bfloat16)[2:]
    weight = torch.randn(128, device="cuda", dtype=x.dtype)
    assert can_use_rmsnorm_preserve_reduction(x, weight)
    torch.testing.assert_close(
        rmsnorm_preserve_reduction(x, weight, 1e-6),
        reference(x, weight, 1e-6),
        atol=0,
        rtol=0,
    )
    assert not can_use_rmsnorm_preserve_reduction(x.cpu(), weight.cpu())
    assert not can_use_rmsnorm_preserve_reduction(x.float(), weight.float())
    assert not can_use_rmsnorm_preserve_reduction(x[:, ::2], weight[::2])
    assert not can_use_rmsnorm_preserve_reduction(x, weight.float())
    assert not can_use_rmsnorm_preserve_reduction(x, weight[:-1])
    assert not can_use_rmsnorm_preserve_reduction(x[:0], weight)


def test_compile_and_graph_replay():
    x = torch.randn(257, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=x.dtype)
    compiled = torch.compile(rmsnorm_preserve_reduction, fullgraph=True)
    torch.testing.assert_close(
        compiled(x, weight, 1e-6), reference(x, weight, 1e-6), atol=0, rtol=0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rmsnorm_preserve_reduction(x, weight, 1e-6)
    x.normal_()
    weight.normal_()
    graph.replay()
    torch.testing.assert_close(out, reference(x, weight, 1e-6), atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
