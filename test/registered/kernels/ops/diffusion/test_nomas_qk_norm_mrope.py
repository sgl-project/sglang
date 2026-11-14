# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)


def reference(x, mean_t, mean_hw, eps, weight_t, weight_hw, cos, sin):
    t, hw = x.float().chunk(2, dim=-1)
    normalized = torch.cat(
        (
            weight_t * (t * torch.rsqrt(mean_t + eps)),
            weight_hw * (hw * torch.rsqrt(mean_hw + eps)),
        ),
        dim=-1,
    )
    rotated = []
    for section in normalized.split((64, 32, 32), dim=-1):
        left, right = section.chunk(2, dim=-1)
        rotated.append(torch.cat((-right, left), dim=-1))
    return torch.addcmul(torch.cat(rotated, dim=-1) * sin, normalized, cos).to(x.dtype)


def inputs(batch, tokens, heads, offset, amplitude=1.0):
    torch.manual_seed(831)
    storage = torch.randn(
        batch, offset + tokens + 3, heads, 128, device="cuda", dtype=torch.bfloat16
    )
    # Retain a real chunk view, including the gap between batch members.
    x = storage[:, offset : offset + tokens]
    x.mul_(amplitude)
    means = [part.float().square().mean(-1, keepdim=True) for part in x.chunk(2, -1)]
    weights = [torch.randn(64, device="cuda", dtype=x.dtype) for _ in range(2)]
    angles = torch.randn(1, tokens, 1, 128, device="cuda")
    return (x, *means, 1e-6, *weights, angles.cos(), angles.sin())


@pytest.mark.parametrize(
    "shape", [(1, 17, 8, 3), (2, 1025, 32, 7), (1, 16384, 32, 0), (1, 2432, 32, 32768)]
)
@pytest.mark.parametrize("amplitude", [1e-30, 1.0, 1e15])
@torch.inference_mode()
def test_qk_norm_mrope_exact_chunk_views(shape, amplitude):
    from sglang.kernels.ops.diffusion import fused_nomas_qk_norm_mrope

    args = inputs(*shape, amplitude)
    expected = reference(*args)
    actual = fused_nomas_qk_norm_mrope(*args)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@torch.inference_mode()
def test_qk_norm_mrope_graph_reads_updated_inputs():
    from sglang.kernels.ops.diffusion import fused_nomas_qk_norm_mrope

    args = inputs(2, 37, 8, 5)
    for _ in range(3):
        fused_nomas_qk_norm_mrope(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_nomas_qk_norm_mrope(*args)
    args[0].add_(0.25)
    for source, mean in zip(args[0].chunk(2, -1), args[1:3]):
        mean.copy_(source.float().square().mean(-1, keepdim=True))
    args[4].mul_(0.75)
    args[6].neg_()
    graph.replay()
    assert torch.equal(actual.view(torch.int16), reference(*args).view(torch.int16))


@torch.inference_mode()
def test_qk_norm_mrope_rejects_invalid_mean_shape():
    from sglang.kernels.ops.diffusion import fused_nomas_qk_norm_mrope

    args = list(inputs(1, 17, 8, 0))
    args[1] = args[1][..., 0]
    with pytest.raises(ValueError, match="mean squares"):
        fused_nomas_qk_norm_mrope(*args)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
