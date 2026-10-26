# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import residual_gate_fp32
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def reference(x, update, gate):
    return (x.float() + gate.float() * update.float()).to(x.dtype)


def inputs(shape, dtype, gate_dtype, broadcast_batch, device="cuda"):
    torch.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=dtype)
    update = torch.randn_like(x)
    # modulation chunks have a nontrivial batch stride and storage offset
    gate = torch.randn(
        1 if broadcast_batch else shape[0],
        1,
        shape[-1] * 9,
        device=device,
        dtype=gate_dtype,
    )[..., shape[-1] * 2 : shape[-1] * 3]
    return x, update, gate


@requires_cuda
@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp32_gate", [False, True])
@pytest.mark.parametrize("broadcast_batch", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 17, 67), (2, 1024, 4096)])
def test_bitexact(shape, dtype, fp32_gate, broadcast_batch):
    x, update, gate = inputs(
        shape, dtype, torch.float32 if fp32_gate else dtype, broadcast_batch
    )
    originals = [tensor.clone() for tensor in (x, update, gate)]
    actual = residual_gate_fp32(x, update, gate)
    expected = reference(x, update, gate)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    for tensor, original in zip((x, update, gate), originals, strict=True):
        assert torch.equal(tensor, original)


@requires_cuda
@torch.no_grad()
def test_fp32_rounding_not_fma_or_bf16_product():
    x = torch.tensor([[[-1.0, -0.435546875]]], device="cuda")
    update = torch.tensor([[[1.0000001192092896, 0.48046875]]], device="cuda")
    gate = torch.tensor([[[0.9999998807907104, 0.90625]]], device="cuda")
    actual = residual_gate_fp32(x, update, gate)
    assert torch.equal(actual, reference(x, update, gate))
    # the first product rounds to 1 before adding -1; FMA would retain -2**-46
    assert actual[0, 0, 0].item() == 0
    x, update, gate = (tensor.bfloat16() for tensor in (x, update, gate))
    expected = reference(x, update, gate)
    assert not torch.equal(expected, x + gate * update)
    assert torch.equal(residual_gate_fp32(x, update, gate), expected)


@requires_cuda
@torch.no_grad()
def test_compile_and_graph():
    x, update, gate = inputs((2, 257, 128), torch.bfloat16, torch.float32, False)
    compiled = torch.compile(residual_gate_fp32, fullgraph=True)
    assert torch.equal(compiled(x, update, gate), reference(x, update, gate))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = residual_gate_fp32(x, update, gate)
    x.normal_()
    update.normal_()
    gate.normal_()
    graph.replay()
    assert torch.equal(actual, reference(x, update, gate))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_fallback_and_empty(device):
    x, update, gate = inputs((2, 17, 67), torch.float32, torch.float32, False, device)
    x = x[:, ::2].requires_grad_()
    update = update[:, ::2].requires_grad_()
    gate.requires_grad_()
    actual = residual_gate_fp32(x, update, gate)
    expected = reference(x, update, gate)
    assert torch.equal(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.sum(), (x, update, gate)),
        torch.autograd.grad(expected.sum(), (x, update, gate)),
        strict=True,
    ):
        assert torch.equal(actual_grad, expected_grad)
    with torch.no_grad():
        assert torch.equal(residual_gate_fp32(x, update, gate), expected)
        assert residual_gate_fp32(x[:, :0], update[:, :0], gate).shape == (2, 0, 67)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
