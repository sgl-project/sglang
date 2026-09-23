# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    can_use_fused_complex_rope,
    fused_complex_rope,
)
from sglang.multimodal_gen.runtime.models.dits import qwen_image21
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def reference(x, rope):
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)


def inputs(shape, dtype):
    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    # a contiguous slice retains the nonzero cache offset used by SP ranks
    angles = torch.randn(shape[1] + 5, shape[-1] // 2, device="cuda") * 20
    rope = torch.polar(torch.ones_like(angles), angles)[5:]
    return x, rope


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 32), (2, 17, 3, 64), (1, 257, 16, 128), (1, 4096, 32, 128)]
)
def test_complex_rope_matches_complex_multiplication(dtype, shape):
    x, rope = inputs(shape, dtype)
    assert can_use_fused_complex_rope(x, rope)
    actual = fused_complex_rope(x, rope)
    torch.testing.assert_close(actual, reference(x, rope), atol=0, rtol=0)


def test_complex_rope_layout_guards():
    x, rope = inputs((2, 17, 3, 64), torch.bfloat16)
    assert not can_use_fused_complex_rope(x.cpu(), rope.cpu())
    assert not can_use_fused_complex_rope(x.double(), rope)
    assert not can_use_fused_complex_rope(x, rope.to(torch.complex128))
    assert not can_use_fused_complex_rope(x[:, ::2], rope[::2])
    assert not can_use_fused_complex_rope(x, rope[:-1])
    assert not can_use_fused_complex_rope(x[:, :0], rope[:0])


def test_complex_rope_compile_and_graph_replay():
    x, rope = inputs((1, 257, 8, 128), torch.bfloat16)
    compiled = torch.compile(fused_complex_rope, fullgraph=True)
    torch.testing.assert_close(compiled(x, rope), reference(x, rope), atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused_complex_rope(x, rope)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(out, reference(x, rope), atol=0, rtol=0)


def test_qwen21_rope_first_sight_verification(monkeypatch):
    x, rope = inputs((1, 257, 8, 128), torch.bfloat16)
    gate = BitExactFusionGate("test complex RoPE")
    monkeypatch.setattr(qwen_image21, "_ROPE_FUSION", gate)
    torch.testing.assert_close(
        qwen_image21.apply_rope(x, rope), reference(x, rope), atol=0, rtol=0
    )
    assert gate.verified and not gate.disabled

    gate = BitExactFusionGate("test mismatched RoPE")
    monkeypatch.setattr(qwen_image21, "_ROPE_FUSION", gate)
    monkeypatch.setattr(
        qwen_image21, "fused_complex_rope", lambda x, rope: torch.zeros_like(x)
    )
    torch.testing.assert_close(
        qwen_image21.apply_rope(x, rope), reference(x, rope), atol=0, rtol=0
    )
    assert gate.disabled and not gate.verified


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
