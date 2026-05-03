"""Triton grouped LoRA delta for the NVFP4 MoE LoRA path.

Defines ``triton_moe_lora_delta``: a standalone Triton kernel computing
``output += scaling * (sorted_input @ A^T) @ B^T`` per expert group on
flat sorted ``[total_tokens, *]`` data.  Used by ``FusedMoEWithLoRA``
in ``layers.py`` for LoRA injection between the CUTLASS base GEMMs in
the NVFP4 expert-LoRA path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_lora_a_proj_kernel(
    input_ptr,
    lora_A_ptr,
    inter_ptr,
    expert_offsets_ptr,
    stride_input_m,
    stride_input_k,
    stride_A_e,
    stride_A_r,
    stride_A_k,
    stride_inter_m,
    stride_inter_r,
    K: tl.constexpr,
    rank_a: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """A-projection: inter = input @ lora_A^T, grouped by expert.

    Grid: (ceil(max_tokens_per_expert / BLOCK_M), num_experts)
    """
    pid_m = tl.program_id(0)
    expert_id = tl.program_id(1)

    start = tl.load(expert_offsets_ptr + expert_id)
    end = tl.load(expert_offsets_ptr + expert_id + 1)

    m_start = pid_m * BLOCK_M
    if m_start >= (end - start):
        return

    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < (end - start)
    global_m = start + m_range
    # Clamp masked-out lanes to a valid in-bounds row so the pointer
    # arithmetic itself stays safe even when `BLOCK_M` straddles the end of
    # this expert's segment (mask would otherwise drop the load, but some
    # Triton/HW combinations materialise the OOB pointer first and trip
    # CUDA "illegal memory access").
    safe_global_m = tl.minimum(global_m, end - 1)

    acc = tl.zeros((BLOCK_M, rank_a), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load input block [BLOCK_M, BLOCK_K]
        inp = tl.load(
            input_ptr
            + safe_global_m[:, None] * stride_input_m
            + k_range[None, :] * stride_input_k,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Load lora_A block [rank_a, BLOCK_K] for this expert
        r_range = tl.arange(0, rank_a)
        la = tl.load(
            lora_A_ptr
            + expert_id * stride_A_e
            + r_range[:, None] * stride_A_r
            + k_range[None, :] * stride_A_k,
            mask=k_mask[None, :],
            other=0.0,
        )

        acc += tl.dot(inp, tl.trans(la))

    # Store inter [BLOCK_M, rank_a]
    r_range = tl.arange(0, rank_a)
    tl.store(
        inter_ptr
        + safe_global_m[:, None] * stride_inter_m
        + r_range[None, :] * stride_inter_r,
        acc.to(inter_ptr.dtype.element_ty),
        mask=m_mask[:, None],
    )


@triton.jit
def _grouped_lora_b_proj_add_kernel(
    inter_ptr,
    lora_B_ptr,
    output_ptr,
    expert_offsets_ptr,
    scaling_ptr,
    stride_inter_m,
    stride_inter_r,
    stride_B_e,
    stride_B_n,
    stride_B_r,
    stride_out_m,
    stride_out_n,
    rank: tl.constexpr,
    inter_rank_offset: tl.constexpr,
    out_n_offset: tl.constexpr,
    N_block: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """B-projection + add: output += scaling * inter @ lora_B^T, grouped by expert.

    For stacked projections, inter_rank_offset and out_n_offset select the
    gate or up portion. For non-stacked, both are 0.

    scaling_ptr points to a 1-element float32 tensor on GPU (no .item() needed,
    CUDA graph safe).

    Grid: (ceil(max_tokens_per_expert / BLOCK_M), ceil(N_block / BLOCK_N), num_experts)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = tl.program_id(2)

    start = tl.load(expert_offsets_ptr + expert_id)
    end = tl.load(expert_offsets_ptr + expert_id + 1)

    m_start = pid_m * BLOCK_M
    if m_start >= (end - start):
        return

    n_start = pid_n * BLOCK_N
    if n_start >= N_block:
        return

    scaling = tl.load(scaling_ptr).to(tl.float32)

    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < (end - start)
    global_m = start + m_range
    # Clamp masked-out lanes (see _grouped_lora_a_proj_kernel for rationale).
    safe_global_m = tl.minimum(global_m, end - 1)

    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N_block
    safe_n_block = tl.minimum(n_range, N_block - 1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    r_range = tl.arange(0, rank)

    # Load inter block [BLOCK_M, rank] (with rank offset for stacked)
    inter_block = tl.load(
        inter_ptr
        + safe_global_m[:, None] * stride_inter_m
        + (inter_rank_offset + r_range[None, :]) * stride_inter_r,
        mask=m_mask[:, None],
        other=0.0,
    )

    # Load lora_B block [BLOCK_N, rank] for this expert (with n offset for stacked)
    lb = tl.load(
        lora_B_ptr
        + expert_id * stride_B_e
        + (out_n_offset + safe_n_block[:, None]) * stride_B_n
        + r_range[None, :] * stride_B_r,
        mask=n_mask[:, None],
        other=0.0,
    )

    acc = tl.dot(inter_block.to(lb.dtype), tl.trans(lb))

    # Load existing output, add delta, store back
    out_n = out_n_offset + safe_n_block
    existing = tl.load(
        output_ptr
        + safe_global_m[:, None] * stride_out_m
        + out_n[None, :] * stride_out_n,
        mask=m_mask[:, None] & n_mask[None, :],
        other=0.0,
    )

    result = existing + scaling * acc.to(existing.dtype)

    tl.store(
        output_ptr
        + safe_global_m[:, None] * stride_out_m
        + out_n[None, :] * stride_out_n,
        result,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def triton_moe_lora_delta(
    output: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: torch.Tensor,
    is_stacked: bool,
    inter: torch.Tensor,
):
    """Drop-in LoRA delta on flat sorted-by-expert data.

    Computes: output += scaling * (sorted_input @ lora_A^T) @ lora_B^T
    per expert group (grouped by expert_offsets).

    All CUDA-graph safe: no .item() calls, no dynamic allocations, fixed grid.

    Args:
        output: [total_tokens, N] bf16, modified in-place
        sorted_input: [total_tokens, K] bf16, tokens sorted by expert
        expert_offsets: [E+1] int32, cumulative token counts per expert
        lora_A: [E, rank_a, K] bf16 (rank_a = 2*rank for stacked)
        lora_B: [E, N, rank] bf16
        scaling: [1] float32 tensor (LoRA scaling factor)
        is_stacked: True for w13 (gate+up), False for w2
        inter: [max_tokens, rank_a] bf16, pre-allocated intermediate buffer
    """
    total_tokens = sorted_input.shape[0]
    K = sorted_input.shape[1]
    E = lora_A.shape[0]
    rank_a = lora_A.shape[1]
    N = lora_B.shape[1]

    if is_stacked:
        rank = rank_a // 2
    else:
        rank = rank_a

    # Ensure scaling is a 1-element float32 tensor on GPU (for CUDA graph safety:
    # the kernel reads it via tl.load, no .item() / GPU-CPU sync needed).
    if scaling.numel() != 1:
        scaling = scaling[0:1]
    scaling = scaling.to(dtype=torch.float32, device=output.device).contiguous()

    # Triton tl.dot requires the contraction and result dims to be >= 16.
    # For LoRA rank >= 8 with stacked (rank_a = 2*8 = 16), this is satisfied.
    # For smaller ranks, pad to 16.
    assert rank_a >= 16, (
        f"LoRA rank_a must be >= 16 for Triton tl.dot, got {rank_a}. "
        f"Use LoRA rank >= 8 for stacked (w13) or rank >= 16 for non-stacked (w2)."
    )
    # Use next power of 2 for rank_a constexpr (Triton tl.dot perf)
    rank_a_constexpr = max(triton.next_power_of_2(rank_a), 16)

    BLOCK_M = 64
    BLOCK_K = 128

    # Fixed grid: use total_tokens as upper bound for per-expert tokens.
    # Blocks for experts with fewer tokens exit early (first check in kernel).
    max_m_blocks = triton.cdiv(total_tokens, BLOCK_M)

    # === Step 1: A-projection ===
    # If rank_a != rank_a_constexpr, we need to mask stores. For simplicity,
    # require rank_a to be a power of 2 (which LoRA ranks almost always are).
    assert (
        rank_a == rank_a_constexpr or rank_a_constexpr == 16
    ), f"rank_a={rank_a} must be a power of 2 >= 16"
    grid_a = (max_m_blocks, E)
    _grouped_lora_a_proj_kernel[grid_a](
        sorted_input,
        lora_A,
        inter,
        expert_offsets,
        sorted_input.stride(0),
        sorted_input.stride(1),
        lora_A.stride(0),
        lora_A.stride(1),
        lora_A.stride(2),
        inter.stride(0),
        inter.stride(1),
        K=K,
        rank_a=rank_a,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )

    # === Step 2: B-projection + add ===
    BLOCK_N = 128

    if is_stacked:
        I = N // 2
        max_n_blocks = triton.cdiv(I, BLOCK_N)
        grid_b = (max_m_blocks, max_n_blocks, E)

        # Gate projection: inter[:, :rank] @ lora_B[:, :I, :]^T
        _grouped_lora_b_proj_add_kernel[grid_b](
            inter,
            lora_B,
            output,
            expert_offsets,
            scaling,
            inter.stride(0),
            inter.stride(1),
            lora_B.stride(0),
            lora_B.stride(1),
            lora_B.stride(2),
            output.stride(0),
            output.stride(1),
            rank=max(triton.next_power_of_2(rank), 16),
            inter_rank_offset=0,
            out_n_offset=0,
            N_block=I,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        # Up projection: inter[:, rank:] @ lora_B[:, I:, :]^T
        _grouped_lora_b_proj_add_kernel[grid_b](
            inter,
            lora_B,
            output,
            expert_offsets,
            scaling,
            inter.stride(0),
            inter.stride(1),
            lora_B.stride(0),
            lora_B.stride(1),
            lora_B.stride(2),
            output.stride(0),
            output.stride(1),
            rank=max(triton.next_power_of_2(rank), 16),
            inter_rank_offset=rank,
            out_n_offset=I,
            N_block=I,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    else:
        max_n_blocks = triton.cdiv(N, BLOCK_N)
        grid_b = (max_m_blocks, max_n_blocks, E)

        _grouped_lora_b_proj_add_kernel[grid_b](
            inter,
            lora_B,
            output,
            expert_offsets,
            scaling,
            inter.stride(0),
            inter.stride(1),
            lora_B.stride(0),
            lora_B.stride(1),
            lora_B.stride(2),
            output.stride(0),
            output.stride(1),
            rank=max(triton.next_power_of_2(rank), 16),
            inter_rank_offset=0,
            out_n_offset=0,
            N_block=N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
