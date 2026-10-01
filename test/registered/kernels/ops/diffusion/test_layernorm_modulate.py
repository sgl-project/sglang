# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch
from torch.nn import functional as F

from sglang.kernels.ops.diffusion import (
    can_use_fused_layernorm_modulate,
    fused_layernorm_modulate,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def reference(x, scale, shift, eps):
    out = F.layer_norm(x, (x.shape[-1],), eps=eps) * (1 + scale[:, None])
    return out if shift is None else out + shift[:, None]


@pytest.mark.parametrize("shape", [(1, 1, 128), (1, 4359, 4096), (2, 1024, 4096)])
@pytest.mark.parametrize("amplitude,eps", [(1e-4, 1e-6), (1.0, 1e-6), (100.0, 1e-5)])
@pytest.mark.parametrize("has_shift", [False, True])
def test_modulation_preserves_bits(shape, amplitude, eps, has_shift):
    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * amplitude
    modulation = torch.randn(shape[0], 4 * shape[-1], device="cuda", dtype=x.dtype)
    scale, shift = modulation.chunk(4, dim=-1)[:2]
    scale[:, :3] = torch.tensor([-1, 0, 1], device=x.device, dtype=x.dtype)
    if not has_shift:
        shift = None
    assert can_use_fused_layernorm_modulate(x, scale, shift)
    actual = fused_layernorm_modulate(x, scale, shift, eps)
    expected = reference(x, scale, shift, eps)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_scale_only_preserves_signed_zero():
    x = torch.ones(1, 17, 128, device="cuda", dtype=torch.bfloat16)
    scale = torch.full((1, 128), -2, device=x.device, dtype=x.dtype)
    actual = fused_layernorm_modulate(x, scale, None, 1e-6)
    expected = reference(x, scale, None, 1e-6)
    assert torch.signbit(expected).all()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_scale_only_layout_guards():
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(2, 128, device=x.device, dtype=x.dtype)
    assert can_use_fused_layernorm_modulate(x, scale, None)
    assert not can_use_fused_layernorm_modulate(x.cpu(), scale.cpu(), None)
    assert not can_use_fused_layernorm_modulate(x.float(), scale.float(), None)
    assert not can_use_fused_layernorm_modulate(x[:, ::2], scale, None)
    assert not can_use_fused_layernorm_modulate(x, scale.float(), None)
    assert not can_use_fused_layernorm_modulate(x, scale[:, :-1], None)
    assert not can_use_fused_layernorm_modulate(x[:, :0], scale, None)
    assert not can_use_fused_layernorm_modulate(x, scale, scale.float())
    strided = torch.empty(2, 256, device=x.device, dtype=x.dtype)[:, :128]
    assert not can_use_fused_layernorm_modulate(x, scale, strided)


@pytest.mark.parametrize("has_shift", [False, True])
def test_compile_and_graph_replay(has_shift):
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.bfloat16)
    modulation = torch.randn(2, 512, device=x.device, dtype=x.dtype)
    scale, shift = modulation.chunk(4, dim=-1)[:2]
    if not has_shift:
        shift = None
    compiled = torch.compile(fused_layernorm_modulate, fullgraph=True)
    expected = reference(x, scale, shift, 1e-6)
    actual = compiled(x, scale, shift, 1e-6)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused_layernorm_modulate(x, scale, shift, 1e-6)
    x.normal_()
    modulation.normal_()
    graph.replay()
    expected = reference(x, scale, shift, 1e-6)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
