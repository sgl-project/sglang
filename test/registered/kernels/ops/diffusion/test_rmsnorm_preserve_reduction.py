# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    rmsnorm_modulate_preserve_reduction,
    rmsnorm_preserve_reduction,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

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

    actual = rmsnorm_preserve_reduction(x, weight, eps)
    torch.testing.assert_close(actual, reference(x, weight, eps), atol=0, rtol=0)


def test_packed_projection_view():
    packed = torch.randn(2, 17, 3, 4, 128, device="cuda", dtype=torch.bfloat16)
    x = packed.unbind(2)[0]
    weight = torch.randn(128, device=x.device, dtype=x.dtype)
    actual = rmsnorm_preserve_reduction(x, weight, 1e-6)
    assert torch.equal(actual, reference(x, weight, 1e-6))


def test_compiled_transposed_input_has_contiguous_output():
    x = torch.randn(17, 2, 128, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    weight = torch.randn(128, device=x.device, dtype=x.dtype)
    compiled = torch.compile(rmsnorm_preserve_reduction, fullgraph=True)
    actual = compiled(x, weight, 1e-6)
    assert torch.equal(actual, reference(x, weight, 1e-6))
    assert actual.is_contiguous()


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


def modulate_reference(x, affine, adaptive):
    # Separate FP32 square, mean and affine operations, with a final dtype cast.
    variance = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + 1e-6)
    if adaptive:
        scale, shift, gate = affine.unsqueeze(1).chunk(3, dim=-1)
        return (normalized * (1.0 + scale.float()) + shift.float()).to(
            x.dtype
        ), gate.to(x.dtype)
    return (normalized * (1.0 + affine.float())).to(x.dtype), None


@pytest.mark.parametrize(
    "shape,dtype,affine_dtype,adaptive",
    [
        ((1, 50, 1024), torch.bfloat16, torch.bfloat16, True),
        ((1, 819, 2048), torch.bfloat16, torch.bfloat16, False),
        ((2, 7, 1024), torch.float16, torch.float32, True),
        ((2, 7, 1024), torch.float16, torch.float32, False),
    ],
)
def test_pointwise_fusion_preserves_torch_rounding(
    shape, dtype, affine_dtype, adaptive
):
    """Changing reduction order or contracting the affine FMA breaks exact parity."""
    torch.manual_seed(42)
    # Exercise a non-current input device when multiple GPUs are available.
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    x = torch.randn(shape, device=device, dtype=dtype)
    x[:, 0] *= 1e-4
    affine_shape = (shape[0], shape[2] * 3) if adaptive else (shape[2],)
    affine = torch.randn(affine_shape, device=device, dtype=affine_dtype)
    with torch.cuda.device(0), torch.inference_mode():
        expected, expected_gate = modulate_reference(x, affine, adaptive)
        actual, gate = rmsnorm_modulate_preserve_reduction(
            x, None if adaptive else affine, affine if adaptive else None, 1e-6
        )
        assert torch.cuda.current_device() == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if adaptive:
        torch.testing.assert_close(gate, expected_gate, rtol=0, atol=0)


def test_norm_replay_reads_current_modulation_and_preserves_owned_outputs():
    x = torch.randn(2, 50, 1024, device="cuda", dtype=torch.bfloat16)
    modulation = torch.randn(2, 3072, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        retained, _ = rmsnorm_modulate_preserve_reduction(x, None, modulation, 1e-6)
        snapshot = retained.clone()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph, stream=stream):
        actual, gate = rmsnorm_modulate_preserve_reduction(x, None, modulation, 1e-6)
    with torch.inference_mode():
        x.add_(0.25)
        modulation.mul_(0.5)
        graph.replay()
        expected, expected_gate = modulate_reference(x, modulation, True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(gate, expected_gate, rtol=0, atol=0)
    torch.testing.assert_close(retained, snapshot, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
