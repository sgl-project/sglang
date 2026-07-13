"""Wrapper utilities for the hpc-ops library.

hpc-ops is Tencent Hunyuan AI Infra team's production-grade operator library
for LLM inference, optimized for NVIDIA H20/Hopper GPUs (SM90+).

This module provides thin wrappers around hpc.fuse_moe / hpc.fuse_moe_blockwise
so that the MoE runner and attention backend can import them without a hard
dependency on the hpc package at module load time.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

import torch


def has_hpc() -> bool:
    """Check whether the *hpc* extension package is installed."""
    return importlib.util.find_spec("hpc") is not None


def hpc_fuse_moe(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_scale: torch.Tensor,
    act_and_mul_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wrapper for hpc.fuse_moe (per-tensor FP8 FusedMoE).

    Args:
        x: Input features [num_seq, hidden_size], fp8.
        gate_up_weight: Gate+up expert weights [num_expert, 2*inter_dim, hidden_size].
        down_weight: Down expert weights [num_expert, hidden_size, inter_dim].
        gate_up_scale: Per-tensor scale for gate+up GEMM.
        down_scale: Per-tensor scale for down GEMM.
        act_and_mul_scale: Scale applied at the activation boundary.
        topk_ids: Expert assignment [num_seq, top_k], int32.
        topk_scale: Expert weights (routing scores) [num_seq, top_k], float32.
        rank_ep: Current EP rank.
        num_expert_total: Total number of experts (across all EP ranks).
        use_bf16_mul: Whether to use bf16 for the final multiplication.
        shared_output: Optional shared-expert output to add to the result.
        output: Optional pre-allocated output tensor.

    Returns:
        Output tensor [num_seq, hidden_size], bfloat16.
    """
    import hpc

    return hpc.fuse_moe(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        use_bf16_mul,
        shared_output,
        output,
    )


def hpc_fuse_moe_blockwise(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wrapper for hpc.fuse_moe_blockwise (blockwise FP8 FusedMoE).

    Args:
        x: Input features [num_seq, hidden_size], fp8.
        x_scale: Per-block input scale.
        gate_up_weight: Gate+up expert weights, fp8.
        gate_up_weight_scale: Per-block weight scale for gate+up.
        down_weight: Down expert weights, fp8.
        down_weight_scale: Per-block weight scale for down.
        topk_ids: Expert assignment [num_seq, top_k], int32.
        topk_scale: Expert weights (routing scores) [num_seq, top_k], float32.
        rank_ep: Current EP rank.
        num_expert_total: Total number of experts (across all EP ranks).
        shared_output: Optional shared-expert output to add to the result.
        output: Optional pre-allocated output tensor.

    Returns:
        Output tensor [num_seq, hidden_size], bfloat16.
    """
    import hpc

    return hpc.fuse_moe_blockwise(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        shared_output,
        output,
    )
