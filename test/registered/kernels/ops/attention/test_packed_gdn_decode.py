"""Packed decode recurrence, state indexing and graph replay against torch."""

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_recurrent_graph_with_strided_state(batch, normalize, bias_dtype):
    torch.manual_seed(42)
    packed = torch.randn(batch, 2560, device="cuda", dtype=torch.bfloat16) * 0.05
    a = torch.randn(batch, 12, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    log_a = torch.randn(12, device="cuda") * 0.1
    bias = torch.randn(12, device="cuda", dtype=bias_dtype)
    # State slots and request IDs have larger strides than their logical shape.
    backing = torch.randn(16, 12, 128, 128, device="cuda") * 0.1
    state = backing[::2]
    indices = torch.zeros(batch * 2, device="cuda", dtype=torch.int64)[::2]
    output = torch.empty(batch, 1, 12, 128, device="cuda", dtype=torch.bfloat16)
    scale = 128**-0.5

    def run():
        fused_recurrent_gated_delta_rule_packed_decode(
            packed, a, b, log_a, bias, scale, state, output, indices, normalize
        )

    indices.copy_(torch.arange(batch, device="cuda"))
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    reference_state = state.double().clone()
    for step in range(8):
        packed.normal_(std=0.05)
        a.normal_()
        b.normal_()
        slots = [(i + step) % 8 for i in range(batch)]
        if step % 3 == 0:
            slots[-1] = -1
        indices.copy_(torch.tensor(slots, device="cuda"))
        q, k, v = torch.split(packed.double(), [512, 512, 1536], dim=-1)
        q = q.view(batch, 4, 128).repeat_interleave(3, dim=1)
        k = k.view(batch, 4, 128).repeat_interleave(3, dim=1)
        v = v.view(batch, 12, 128)
        if normalize:
            q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        q = q * scale
        decay = (-log_a.exp() * F.softplus(a.float() + bias.float())).exp().double()
        beta = b.float().sigmoid().to(b.dtype).double()
        expected = torch.zeros_like(output, dtype=torch.float64)
        for row, slot in enumerate(slots):
            if slot < 0:
                continue
            h = reference_state[slot] * decay[row, :, None, None]
            correction = (v[row] - (h * k[row, :, None, :]).sum(-1)) * beta[
                row, :, None
            ]
            h = h + correction[..., None] * k[row, :, None, :]
            reference_state[slot] = h
            expected[row, 0] = (h * q[row, :, None, :]).sum(-1)
        graph.replay()
        torch.testing.assert_close(
            state.double(), reference_state, rtol=1e-4, atol=2e-6
        )
        torch.testing.assert_close(output.double(), expected, rtol=1e-2, atol=1e-4)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
