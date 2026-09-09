"""Gather a permuted (expert-sorted) MoE activation back to expanded order.

The trtllm-gen fused MoE hands its FC1 activation back in permuted order --
``[max_num_padded_tokens_gemm1, intermediate]`` sorted by expert -- together
with ``expanded_idx_to_permuted_idx``, an int32 ``[num_tokens * top_k]`` map
from the expanded ``(token, slot)`` order into it. ``-1`` marks a slot routed
outside the local EP shard or into cuda-graph padding.

A down-projection LoRA consumes the activation in expanded order and sums over
every ``top_k`` slot unconditionally, so inactive slots must read back as exact
zeros. This reproduces the reference in FlashInfer's
``flashinfer/fused_moe/moe_lora_delta.py:271-274``::

    perm  = expanded_idx_to_permuted_idx.to(torch.int64)
    valid = perm >= 0
    a_exp = torch.zeros(num_tokens * top_k, intermediate)
    a_exp[valid] = activation_permuted[perm[valid]]

with one difference that matters to the caller: every destination row is
written (inactive rows with zeros) rather than only the valid ones, so the
destination may be an uninitialized ``torch.empty`` and needs no separate fill.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_permuted_to_expanded_kernel(
    src_ptr,  # [num_padded_rows, I] permuted activation
    perm_ptr,  # [num_tokens * top_k] int32, -1 = inactive slot
    dst_ptr,  # [num_tokens * top_k, I] expanded activation (may be uninitialized)
    I,
    stride_sm,
    stride_si,
    stride_dm,
    stride_di,
    BLOCK_I: tl.constexpr,
):
    # int64 row indices: num_padded_rows * I passes 2**31 at prefill shapes.
    row = tl.program_id(0).to(tl.int64)
    col_blk = tl.program_id(1)

    p = tl.load(perm_ptr + row).to(tl.int64)
    valid = p >= 0
    # Clamp before the multiply so an inactive slot never forms a negative
    # address; the mask below is what actually keeps the load from happening.
    src_row = tl.maximum(p, 0)

    offs = col_blk * BLOCK_I + tl.arange(0, BLOCK_I)
    col_mask = offs < I
    vals = tl.load(
        src_ptr + src_row * stride_sm + offs * stride_si,
        mask=col_mask & valid,
        other=0.0,  # the zero-fill the down-LoRA expand relies on
    )
    # Unconditional store: the destination is allowed to arrive uninitialized.
    tl.store(dst_ptr + row * stride_dm + offs * stride_di, vals, mask=col_mask)


def gather_permuted_activation(
    activation_permuted: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    num_tokens: int,
    top_k: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather ``[num_padded_rows, I]`` permuted rows into ``[num_tokens, top_k, I]``.

    ``expanded_idx_to_permuted_idx`` is the map returned alongside the trtllm-gen
    MoE output. ``out``, if given, must be contiguous ``[num_tokens, top_k, I]``;
    it does not need to be zeroed.
    """
    assert activation_permuted.dim() == 2, (
        f"expected a 2-D permuted activation, got {tuple(activation_permuted.shape)}"
    )
    assert expanded_idx_to_permuted_idx.is_contiguous(), (
        "expanded_idx_to_permuted_idx must be contiguous"
    )
    inter = activation_permuted.shape[1]
    rows = num_tokens * top_k
    # The map is [num_tokens * (top_k + num_fused_shared_experts)]; a fused shared
    # expert would add slots this gather has no destination for.
    assert expanded_idx_to_permuted_idx.numel() == rows, (
        "gather requires num_fused_shared_experts == 0: expanded_idx_to_permuted_idx "
        f"has {expanded_idx_to_permuted_idx.numel()} entries, expected "
        f"num_tokens * top_k = {rows}"
    )

    if out is None:
        out = activation_permuted.new_empty((num_tokens, top_k, inter))
    assert out.shape == (num_tokens, top_k, inter) and out.is_contiguous(), (
        f"out must be a contiguous {(num_tokens, top_k, inter)} tensor, got "
        f"{tuple(out.shape)} contiguous={out.is_contiguous()}"
    )
    if rows == 0 or inter == 0:
        return out

    out_2d = out.view(rows, inter)
    BLOCK_I = min(1024, triton.next_power_of_2(inter))
    grid = (rows, triton.cdiv(inter, BLOCK_I))
    _gather_permuted_to_expanded_kernel[grid](
        activation_permuted,
        expanded_idx_to_permuted_idx,
        out_2d,
        inter,
        activation_permuted.stride(0),
        activation_permuted.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        BLOCK_I=BLOCK_I,
        num_warps=4,
    )
    return out
