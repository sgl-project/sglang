from typing import Optional, Tuple

import torch

# Import triggers TORCH_LIBRARY_FRAGMENT registration for sgl_kernel ops
from sgl_kernel import cula_kda_ops  # noqa: F401


def kda_fwd_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    scale: float,
    safe_gate: bool = True,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    input_state: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cuLA SM90 fully-fused KDA forward prefill kernel.

    Args:
        q: [packed_seq, H, K], bf16
        k: [packed_seq, H, K], bf16
        v: [packed_seq, H, V], bf16
        cu_seqlens: [N+1], int32
        workspace_buffer: [sm_count * 128], uint8
        scale: attention scale factor
        safe_gate: whether gate values are in safe range [-5, 0)
        output: optional pre-allocated output [packed_seq, H, V], bf16
        output_state: optional pre-allocated output state [N, H, K, V], fp32
        input_state: optional input state [N, H, K, V], fp32
        alpha: optional gate cumsum [packed_seq, H, K], fp32
        beta: optional beta [packed_seq, H], fp32

    Returns:
        (output, output_state) tuple
    """
    return torch.ops.sgl_kernel.kda_fwd_prefill(
        output,
        output_state,
        q,
        k,
        v,
        input_state,
        alpha,
        beta,
        cu_seqlens,
        workspace_buffer,
        scale,
        safe_gate,
    )
