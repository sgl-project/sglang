"""Fuse TRTLLM MoE finalize with the sigmoid-gated shared-expert add."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled


@triton.jit
def _moe_finalize_gate_add_kernel(
    gemm2_ptr,  # [num_permuted_rows, gemm2_stride]
    perm_ptr,  # [num_tokens * TOP_K] int32, -1 = not local
    expert_weights_ptr,  # [num_tokens, TOP_K]
    hidden_states_ptr,  # [num_tokens, hidden_dim]
    gate_weight_ptr,  # [hidden_dim]
    shared_output_ptr,  # [num_tokens, hidden_dim]
    output_ptr,  # [num_tokens, hidden_dim]
    gemm2_stride: tl.constexpr,
    hidden_dim: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    row_offset = pid * hidden_dim
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # Shared output and gate inputs are ready before the routed GEMM.
    # Only the routed output gather waits for its PDL dependency.
    w = tl.load(gate_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(hidden_states_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    s = tl.load(shared_output_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    gate_val = tl.sigmoid(tl.sum(h * w, axis=0))

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        p = tl.load(perm_ptr + pid * TOP_K + k)
        ew = tl.load(expert_weights_ptr + pid * TOP_K + k).to(tl.float32)
        x = tl.load(
            gemm2_ptr + p.to(tl.int64) * gemm2_stride + offsets,
            mask=mask & (p >= 0),
            other=0.0,
        ).to(tl.float32)
        acc = tl.where(p >= 0, acc + ew * x, acc)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    f = acc.to(output_ptr.dtype.element_ty).to(tl.float32)
    result = gate_val * s
    result += f
    tl.store(output_ptr + row_offset + offsets, result, mask=mask)


def moe_finalize_gate_add(
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    if expert_weights.dtype != torch.bfloat16:
        from sglang.kernels.ops.elementwise.elementwise import (
            fused_gate_sigmoid_mul_add,
        )
        from sglang.kernels.ops.moe.moe_finalize_fuse_shared import (
            moe_finalize_fuse_shared,
        )

        out = moe_finalize_fuse_shared(
            gemm2_out.contiguous(),
            expanded_idx_to_permuted_idx,
            expert_weights,
            None,
            top_k,
        )
        fused_gate_sigmoid_mul_add(hidden_states, gate_weight, shared_output, out)
        return out

    num_tokens, hidden_dim = shared_output.shape
    assert gemm2_out.dtype == torch.bfloat16 and gemm2_out.stride(1) == 1
    assert expanded_idx_to_permuted_idx.dtype == torch.int32
    assert expanded_idx_to_permuted_idx.numel() >= num_tokens * top_k
    assert expert_weights.numel() >= num_tokens * top_k
    assert hidden_states.is_contiguous() and shared_output.is_contiguous()
    assert hidden_states.shape == shared_output.shape
    assert gate_weight.is_contiguous() and gate_weight.shape == (hidden_dim,)
    assert gemm2_out.shape[1] >= hidden_dim

    # Match _launch_fused_gate_sigmoid_mul's reduction order, including invariant mode.
    num_warps = max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), 32), 4)
    if num_tokens >= 1024 or is_batch_invariant_mode_enabled():
        num_warps = min(num_warps, 8)
    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}

    out = torch.empty_like(shared_output)
    _moe_finalize_gate_add_kernel[(num_tokens,)](
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        hidden_states,
        gate_weight,
        shared_output,
        out,
        gemm2_stride=gemm2_out.stride(0),
        hidden_dim=hidden_dim,
        TOP_K=top_k,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        USE_PDL=use_pdl,
        num_warps=num_warps,
        **pdl_kwargs,
    )
    return out
