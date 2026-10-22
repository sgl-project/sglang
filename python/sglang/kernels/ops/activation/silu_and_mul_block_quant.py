"""Fused SwiGLU and blockwise FP8 quantization for MoE down projections with 128 < K < 256.

input:   [M, 2 * K] BF16/FP16, gate followed by up
output:  [M, K] FP8, keeping the logical width
scales:  [M, 2] FP32, the second computed from the live K-tail values only
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_FP8_MAX = 448.0
_MIN_ABSMAX = 1e-10
_SUPPORTED_GROUP_SIZE = 128
_ROW_ALIGNMENT = 16


@triton.jit
def _silu_and_mul_block_quant_two_group_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    expert_ids_ptr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EXPERT_STEP: tl.constexpr,
    HAS_EXPERT_MASK: tl.constexpr,
    ROUTED_LIMIT,
    SHARED_LIMIT,
    NUM_ROUTED_EXPERTS: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    SPLIT_SHARED_LIMIT: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_ABSMAX: tl.constexpr,
):
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)

    offsets = tl.arange(0, 2 * GROUP_SIZE)
    live = offsets < K
    if HAS_EXPERT_MASK:
        expert_id = tl.load(expert_ids_ptr + row // EXPERT_STEP)
        live = live & (expert_id != -1)

    input_base = row_i64 * (2 * K)
    gate = tl.load(input_ptr + input_base + offsets, mask=live, other=0.0).to(
        tl.float32
    )
    up = tl.load(input_ptr + input_base + K + offsets, mask=live, other=0.0).to(
        tl.float32
    )

    activated_gate = gate * tl.sigmoid(gate)
    activated = activated_gate * up
    if HAS_LIMIT:
        if SPLIT_SHARED_LIMIT:
            # Fused shared experts follow the routed ones.
            limit = tl.where(
                expert_id >= NUM_ROUTED_EXPERTS,
                SHARED_LIMIT,
                ROUTED_LIMIT,
            )
        else:
            limit = ROUTED_LIMIT
        activated = tl.minimum(activated_gate, limit) * tl.maximum(
            tl.minimum(up, limit), -limit
        )
    # Round to the input dtype like the unfused silu_and_mul output before quantizing.
    activated = activated.to(input_ptr.dtype.element_ty).to(tl.float32)

    grouped = tl.reshape(activated, (2, GROUP_SIZE))
    absmax = tl.maximum(tl.max(tl.abs(grouped), axis=1), MIN_ABSMAX)
    scales = absmax / FP8_MAX
    quantized = tl.maximum(tl.minimum(grouped / scales[:, None], FP8_MAX), -FP8_MAX)

    scale_offsets = tl.arange(0, 2)
    tl.store(
        scale_ptr + row_i64 * 2 + scale_offsets,
        scales,
    )
    tl.store(
        output_ptr + row_i64 * K + offsets,
        tl.reshape(quantized, (2 * GROUP_SIZE,)),
        mask=offsets < K,
    )


def silu_and_mul_block_quant_two_group(
    x: torch.Tensor,
    group_size: int = _SUPPORTED_GROUP_SIZE,
    *,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
    routed_limit: Optional[float] = None,
    shared_limit: Optional[float] = None,
    num_routed_experts: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse native SwiGLU with dynamic block-FP8 quantization.

    Only covers two quantization groups (128 < K < 256 with group size 128).
    Set ``num_routed_experts`` with ``expert_ids`` when routed and fused
    shared experts use distinct clamp limits.
    """
    assert x.is_cuda, "the fused K-tail activation requires CUDA"
    assert x.dim() == 2, f"expected a 2D gate/up tensor, got {x.dim()}D"
    assert x.is_contiguous(), "the gate/up tensor must be contiguous"
    assert x.dtype in (
        torch.bfloat16,
        torch.float16,
    ), f"expected BF16 or FP16 input, got {x.dtype}"
    assert x.shape[-1] % 2 == 0, "the gate/up width must be even"
    assert group_size == _SUPPORTED_GROUP_SIZE, (
        f"only group_size={_SUPPORTED_GROUP_SIZE} is supported, got {group_size}"
    )

    output_k = x.shape[-1] // 2
    assert group_size < output_k < 2 * group_size, (
        "the optimized kernel requires exactly two quantization groups with a "
        f"partial final group, got K={output_k}, group_size={group_size}"
    )
    assert output_k % _ROW_ALIGNMENT == 0, (
        f"K must be {_ROW_ALIGNMENT}-element aligned, got {output_k}"
    )
    assert expert_step > 0, f"expert_step must be positive, got {expert_step}"
    if num_routed_experts is not None:
        assert expert_ids is not None, "num_routed_experts requires expert_ids"
        assert num_routed_experts is not None and num_routed_experts > 0, (
            "num_routed_experts must be positive"
        )
    if expert_ids is not None:
        assert expert_ids.is_cuda, "expert_ids must be on CUDA"
        assert expert_ids.dtype == torch.int32, (
            f"expert_ids must be int32, got {expert_ids.dtype}"
        )
        assert expert_ids.is_contiguous(), "expert_ids must be contiguous"
        assert expert_ids.numel() * expert_step >= x.shape[0], (
            "expert_ids does not cover every activation row: "
            f"{expert_ids.numel()} * {expert_step} < {x.shape[0]}"
        )

    if shared_limit is not None:
        assert num_routed_experts is not None, (
            "shared_limit requires num_routed_experts"
        )
    # None means no clamp; an infinite limit leaves silu(gate) * up unchanged.
    routed_limit_value = math.inf if routed_limit is None else float(routed_limit)
    shared_limit_value = math.inf if shared_limit is None else float(shared_limit)
    assert routed_limit_value >= 0.0, "routed_limit must be non-negative"
    assert shared_limit_value >= 0.0, "shared_limit must be non-negative"

    output = torch.empty(
        (x.shape[0], output_k),
        device=x.device,
        dtype=torch.float8_e4m3fn,
    )
    scales = torch.empty((x.shape[0], 2), device=x.device, dtype=torch.float32)

    if x.shape[0] > 0:
        _silu_and_mul_block_quant_two_group_kernel[(x.shape[0],)](
            x,
            output,
            scales,
            expert_ids,
            K=output_k,
            GROUP_SIZE=group_size,
            EXPERT_STEP=expert_step,
            HAS_EXPERT_MASK=expert_ids is not None,
            ROUTED_LIMIT=routed_limit_value,
            SHARED_LIMIT=shared_limit_value,
            NUM_ROUTED_EXPERTS=(
                0 if num_routed_experts is None else num_routed_experts
            ),
            HAS_LIMIT=routed_limit is not None or shared_limit is not None,
            SPLIT_SHARED_LIMIT=num_routed_experts is not None,
            FP8_MAX=_FP8_MAX,
            MIN_ABSMAX=_MIN_ABSMAX,
            num_warps=4,
            num_stages=3,
        )

    return output, scales
