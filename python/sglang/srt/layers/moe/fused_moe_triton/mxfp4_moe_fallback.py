"""PyTorch fallback for MXFP4 MoE GEMM on SM120.

The Marlin MXFP4 kernel produces NaN on SM120 (Blackwell Desktop).
This module provides a pure-PyTorch implementation that dequantizes
MXFP4 weights (packed int8 + float8_e8m0fnu scales) to BF16 and uses
torch.matmul for the GEMM, per active expert.

Slow but functionally correct — matches the FlashMLA fallback pattern.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── FP4 E2M1 lookup table ──────────────────────────────────────────
# Nibble encoding: bit3=sign, bit2-1=exponent (bias=1), bit0=mantissa
# 16 possible values for 4-bit float
_FP4_E2M1_LUT = torch.tensor(
    [
        0.0,   0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,   # positive (0x0-0x7)
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,   # negative (0x8-0xF)
    ],
    dtype=torch.float32,
)


def _dequant_mxfp4_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    unpacked_k: int,
) -> torch.Tensor:
    """Dequantize one expert's MXFP4 weight from packed int8 to bfloat16.

    Args:
        packed: [N, K//2] int8 — 2 FP4 values per byte (low nibble=even, high=odd)
        scales: [N, K//32] float32 — dequantization scale per group of 32 elements
        unpacked_k: K, the full unpacked dimension

    Returns:
        [N, K] bfloat16 weight matrix
    """
    device = packed.device
    lut = _FP4_E2M1_LUT.to(device=device)

    # View as unsigned bytes for bit manipulation
    u8 = packed.view(torch.uint8).to(torch.int32)
    low = u8 & 0x0F           # even-index elements
    high = (u8 >> 4) & 0x0F   # odd-index elements

    # Lookup FP4 → float32
    vals_low = lut[low.long()]    # [N, K//2]
    vals_high = lut[high.long()]  # [N, K//2]

    # Interleave: [low0, high0, low1, high1, ...]
    unpacked = torch.stack([vals_low, vals_high], dim=-1)  # [N, K//2, 2]
    unpacked = unpacked.reshape(packed.shape[0], -1)       # [N, K]
    unpacked = unpacked[:, :unpacked_k]                    # trim if needed

    # Apply group scales (group_size=32)
    # scales: [N, K//32] — each scale covers 32 consecutive elements along K
    if scales.dtype == torch.float8_e8m0fnu:
        scales_f32 = scales.to(torch.float32)
    else:
        scales_f32 = scales.float()
    scales_expanded = scales_f32.repeat_interleave(32, dim=-1)[:, :unpacked_k]

    result = unpacked * scales_expanded
    return result.to(torch.bfloat16)


def mxfp4_moe_forward_fallback(
    hidden_states: torch.Tensor,
    w13_packed: torch.Tensor,
    w2_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
    clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    """Pure-PyTorch MXFP4 MoE forward pass.

    Args:
        hidden_states: [M, K] bfloat16 input activations
        w13_packed: [E, 2*I, K//2] int8 packed gate_up_proj weights
        w2_packed: [E, K, I//2] int8 packed down_proj weights
        w13_scale: [E, 2*I, K//32] scales for gate_up_proj
        w2_scale: [E, K, I//32] scales for down_proj
        topk_ids: [M, topk] int32 expert assignments
        topk_weights: [M, topk] float32 routing weights
        hidden_size: K
        intermediate_size: I (per partition)
        inplace: whether to write output in-place
        routed_scaling_factor: optional global scaling factor
        clamp_limit: optional SwiGLU clamp limit (2604B submode)

    Returns:
        [M, K] bfloat16 output tensor
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype
    I = intermediate_size

    output = hidden_states if inplace else torch.zeros(M, K, dtype=dtype, device=device)
    if not inplace:
        output.zero_()

    # Find all active experts
    active_experts = topk_ids.unique()

    for eid in active_experts:
        eid_val = eid.item()
        if eid_val < 0:
            continue

        # Find (token_idx, slot_idx) pairs assigned to this expert
        mask = topk_ids == eid_val  # [M, topk]
        token_mask = mask.any(dim=1)  # [M]
        token_indices = token_mask.nonzero(as_tuple=True)[0]

        if len(token_indices) == 0:
            continue

        # ── GEMM1: gate_up_proj ──
        # w13: [2*I, K//2] int8 → dequant → [2*I, K] bf16
        w13_dq = _dequant_mxfp4_weight(
            w13_packed[eid_val], w13_scale[eid_val], K
        )  # [2*I, K]

        h = hidden_states[token_indices]  # [n, K]
        # y = h @ W13^T  → [n, K] @ [K, 2*I] = [n, 2*I]
        intermediate = torch.matmul(h.float(), w13_dq.float().T).to(dtype)

        # ── SiLU + Mul (with optional clamp) ──
        gate = intermediate[:, :I]
        up = intermediate[:, I:]
        if clamp_limit is not None and clamp_limit > 0:
            gate = torch.clamp(gate, max=clamp_limit)
            up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
        intermediate2 = F.silu(gate) * up  # [n, I]

        # ── GEMM2: down_proj ──
        # w2: [K, I//2] int8 → dequant → [K, I] bf16
        w2_dq = _dequant_mxfp4_weight(
            w2_packed[eid_val], w2_scale[eid_val], I
        )  # [K, I]

        # y = intermediate2 @ W2^T  → [n, I] @ [I, K] = [n, K]
        down = torch.matmul(intermediate2.float(), w2_dq.float().T).to(dtype)

        # ── Accumulate with topk weights (vectorized over topk slots) ──
        expert_mask = (topk_ids[token_indices] == eid_val).to(dtype)  # [n, topk]
        combined_weights = (expert_mask * topk_weights[token_indices].to(dtype)).sum(dim=1, keepdim=True)  # [n, 1]
        output[token_indices] += down * combined_weights

    if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
        output.mul_(routed_scaling_factor)

    return output
