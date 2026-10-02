"""FP32 sigmoid gate, preserving the shared-expert fused-add reduction."""

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_expert_gate_kernel(X, W, G, H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    h = tl.arange(0, B)
    w = tl.load(W + h, h < H, 0).to(tl.float32)
    tl.extra.cuda.gdc_wait()
    x = tl.load(X + row * H + h, h < H, 0).to(tl.float32)
    gate = tl.sigmoid(tl.sum(x * w, 0))
    tl.extra.cuda.gdc_launch_dependents()
    tl.store(G + row, gate)


def shared_expert_gate(
    hidden: torch.Tensor, weight: torch.Tensor, out=None
) -> torch.Tensor:
    assert hidden.shape == (1, 2560) and hidden.is_contiguous()
    assert hidden.dtype == weight.dtype == torch.bfloat16
    assert weight.numel() == 2560 and weight.is_contiguous()
    gate = out
    if gate is None:
        gate = torch.empty(hidden.shape[0], device=hidden.device, dtype=torch.float32)
    assert gate.shape == (hidden.shape[0],) and gate.dtype == torch.float32
    assert gate.device == hidden.device and gate.is_contiguous()
    _shared_expert_gate_kernel[(hidden.shape[0],)](
        hidden, weight, gate, 2560, 4096, num_warps=16, launch_pdl=True
    )
    return gate
