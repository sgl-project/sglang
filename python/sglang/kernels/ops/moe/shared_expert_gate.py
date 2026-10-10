"""FP32 sigmoid gate, preserving the shared-expert fused-add reduction."""

import torch

from sglang.kernels.ops.elementwise.elementwise import (
    _fused_gate_sigmoid_mul_add_kernel,
)


def shared_expert_gate(
    hidden: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    assert hidden.is_cuda and weight.device == hidden.device
    assert hidden.shape == (1, 2560) and hidden.is_contiguous()
    assert hidden.dtype == weight.dtype == torch.bfloat16
    assert weight.numel() == 2560 and weight.is_contiguous()
    gate = out
    if gate is None:
        gate = torch.empty(hidden.shape[0], device=hidden.device, dtype=torch.float32)
    assert gate.shape == (hidden.shape[0],) and gate.dtype == torch.float32
    assert gate.device == hidden.device and gate.is_contiguous()
    _fused_gate_sigmoid_mul_add_kernel[(hidden.shape[0],)](
        hidden,
        weight,
        None,
        gate,
        hidden_dim=2560,
        BLOCK_SIZE=4096,
        DO_ADD=False,
        USE_PDL=True,
        GATE_ONLY=True,
        num_warps=16,
        launch_pdl=True,
    )
    return gate
