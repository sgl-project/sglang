from typing import Optional
import torch


def fused_rs_ln_ag_cta(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    mcptr: int,
    signal_pads: int,
    rank: int,
    world_size: int,
    MAX_CTAS: int,
    epsilon: float,
) -> None:
    """Fused ReduceScatter + RMSNorm + AllGather CTA-based kernel"""
    torch.ops.sgl_kernel.fused_rs_ln_ag_cta.default(
        input, residual, weight, mcptr, signal_pads, rank, world_size, MAX_CTAS, epsilon
    )