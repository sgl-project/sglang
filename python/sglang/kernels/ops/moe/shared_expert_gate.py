"""FP32 sigmoid gate, preserving the shared-expert fused-add reduction."""

import torch

from sglang.kernels.ops.elementwise.elementwise import _launch_fused_gate_sigmoid_mul


def shared_expert_gate(
    hidden: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    assert hidden.is_cuda and weight.device == hidden.device
    gate = out
    if gate is None:
        gate = torch.empty(hidden.shape[0], device=hidden.device, dtype=torch.float32)
    assert gate.device == hidden.device
    _launch_fused_gate_sigmoid_mul(
        hidden, weight.reshape(-1), None, gate, do_add=False, gate_only=True
    )
    return gate
