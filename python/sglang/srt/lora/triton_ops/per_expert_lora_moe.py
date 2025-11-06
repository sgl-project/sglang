# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Per-expert LoRA computation kernel for MoE layers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _per_expert_lora_kernel(
    # Input/Output pointers
    hidden_states_ptr,
    lora_a_weights_ptr,
    lora_b_weights_ptr,
    output_ptr,
    # Dispatch info
    token_ids_ptr,
    expert_ids_ptr,
    lora_ids_ptr,
    # Dimensions
    hidden_dim: tl.constexpr,
    intermediate_dim: tl.constexpr,
    max_rank: tl.constexpr,
    num_experts: tl.constexpr,
    num_tokens: tl.constexpr,
    # Strides for 4D LoRA weights [num_loras, num_experts, *, *]
    lora_a_stride_lora: tl.constexpr,
    lora_a_stride_expert: tl.constexpr,
    lora_a_stride_rank: tl.constexpr,
    lora_a_stride_hidden: tl.constexpr,
    lora_b_stride_lora: tl.constexpr,
    lora_b_stride_expert: tl.constexpr,
    lora_b_stride_intermediate: tl.constexpr,
    lora_b_stride_rank: tl.constexpr,
    # LoRA ranks per adapter
    lora_ranks_ptr,
    # Scaling factors per adapter
    lora_scalings_ptr,
    # Block sizes
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_INTERMEDIATE: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
):
    """
    Compute per-expert LoRA delta: delta = B @ A @ hidden_states.

    Grid: (spatial_tiles, intermediate_slices, num_loras)
    - spatial_tiles: Number of token tiles
    - intermediate_slices: Number of output dimension tiles
    - num_loras: Process each LoRA adapter in parallel
    """
    # Grid IDs
    token_tile_id = tl.program_id(0)
    output_tile_id = tl.program_id(1)
    lora_id = tl.program_id(2)

    # Get rank and scaling for this LoRA adapter
    rank = tl.load(lora_ranks_ptr + lora_id)
    scaling = tl.load(lora_scalings_ptr + lora_id)

    # Early exit if rank is 0
    if rank == 0:
        return

    # Token range for this tile
    token_start = token_tile_id * BLOCK_HIDDEN
    token_end = tl.minimum(token_start + BLOCK_HIDDEN, num_tokens)

    # Output dimension range
    out_start = output_tile_id * BLOCK_INTERMEDIATE
    out_end = tl.minimum(out_start + BLOCK_INTERMEDIATE, intermediate_dim)

    # Process each token in this tile
    for token_idx in range(token_start, token_end):
        if token_idx >= num_tokens:
            break

        # Load dispatch info for this token
        actual_token_id = tl.load(token_ids_ptr + token_idx)
        expert_id = tl.load(expert_ids_ptr + token_idx)
        token_lora_id = tl.load(lora_ids_ptr + token_idx)

        # Skip if this token doesn't belong to current LoRA
        if token_lora_id != lora_id:
            continue

        # Load hidden states for this token: [hidden_dim]
        hidden_ptr = hidden_states_ptr + actual_token_id * hidden_dim
        hidden_offs = tl.arange(0, BLOCK_HIDDEN)
        hidden_mask = hidden_offs < hidden_dim
        hidden = tl.load(hidden_ptr + hidden_offs, mask=hidden_mask, other=0.0)

        # Compute A @ hidden: [rank] = [rank, hidden_dim] @ [hidden_dim]
        intermediate_a = tl.zeros([BLOCK_RANK], dtype=tl.float32)

        for k_tile in range(0, tl.cdiv(hidden_dim, BLOCK_HIDDEN)):
            k_start = k_tile * BLOCK_HIDDEN
            k_offs = tl.arange(0, BLOCK_HIDDEN) + k_start
            k_mask = k_offs < hidden_dim

            # Load from hidden states
            h_vals = tl.load(hidden_ptr + k_offs, mask=k_mask, other=0.0)

            # Load LoRA A weights: [rank, hidden_dim]
            for r in range(BLOCK_RANK):
                if r >= rank:
                    break
                lora_a_offset = (
                    lora_id * lora_a_stride_lora
                    + expert_id * lora_a_stride_expert
                    + r * lora_a_stride_rank
                    + k_start * lora_a_stride_hidden
                )
                a_vals = tl.load(
                    lora_a_weights_ptr + lora_a_offset + k_offs,
                    mask=k_mask,
                    other=0.0,
                )
                intermediate_a[r] += tl.sum(a_vals * h_vals)

        # Compute B @ intermediate_a: [intermediate_dim] = [intermediate_dim, rank] @ [rank]
        out_offs = tl.arange(0, BLOCK_INTERMEDIATE) + out_start
        out_mask = out_offs < intermediate_dim

        output_vals = tl.zeros([BLOCK_INTERMEDIATE], dtype=tl.float32)

        for r in range(BLOCK_RANK):
            if r >= rank:
                break

            # Load LoRA B weights: [intermediate_dim, rank]
            lora_b_offset = (
                lora_id * lora_b_stride_lora
                + expert_id * lora_b_stride_expert
                + out_start * lora_b_stride_intermediate
                + r * lora_b_stride_rank
            )
            b_vals = tl.load(
                lora_b_weights_ptr
                + lora_b_offset
                + out_offs * lora_b_stride_intermediate,
                mask=out_mask,
                other=0.0,
            )
            output_vals += b_vals * intermediate_a[r]

        # Scale and accumulate to output
        output_vals *= scaling
        output_offset = actual_token_id * intermediate_dim + out_start
        tl.atomic_add(output_ptr + output_offset + out_offs, output_vals, mask=out_mask)


def per_expert_lora_forward(
    hidden_states: torch.Tensor,
    lora_a_weights: torch.Tensor,
    lora_b_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    lora_ids: torch.Tensor,
    lora_ranks: torch.Tensor,
    lora_scalings: torch.Tensor,
    num_experts: int,
    base_output: torch.Tensor = None,
) -> torch.Tensor:
    """
    Forward pass for per-expert LoRA computation.

    Args:
        hidden_states: [num_tokens, hidden_dim]
        lora_a_weights: [num_loras, num_experts, max_rank, hidden_dim]
        lora_b_weights: [num_loras, num_experts, intermediate_dim, max_rank]
        token_ids: [num_dispatched] - Original token indices
        expert_ids: [num_dispatched] - Expert ID for each dispatched token
        lora_ids: [num_dispatched] - LoRA ID for each dispatched token
        lora_ranks: [num_loras] - Rank for each LoRA
        lora_scalings: [num_loras] - Scaling factor for each LoRA
        num_experts: Total number of experts
        base_output: [num_tokens, intermediate_dim] - Base MoE output (modified in-place)

    Returns:
        output: [num_tokens, intermediate_dim] - Base output + LoRA delta (in-place)
    """
    num_tokens, hidden_dim = hidden_states.shape
    num_loras, _, intermediate_dim, max_rank = lora_b_weights.shape
    num_dispatched = token_ids.shape[0]

    # Initialize or reuse output tensor for in-place addition
    if base_output is None:
        output = torch.zeros(
            num_tokens,
            intermediate_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    else:
        output = base_output

    # Block sizes (tuned for typical dimensions)
    BLOCK_HIDDEN = 128
    BLOCK_INTERMEDIATE = 128
    BLOCK_RANK = 64

    # Grid dimensions: (spatial_tiles, intermediate_slices, num_loras)
    grid = (
        triton.cdiv(num_dispatched, BLOCK_HIDDEN),
        triton.cdiv(intermediate_dim, BLOCK_INTERMEDIATE),
        num_loras,
    )

    _per_expert_lora_kernel[grid](
        hidden_states,
        lora_a_weights,
        lora_b_weights,
        output,
        token_ids,
        expert_ids,
        lora_ids,
        hidden_dim,
        intermediate_dim,
        max_rank,
        num_experts,
        num_dispatched,
        # LoRA A strides: [num_loras, num_experts, max_rank, hidden_dim]
        lora_a_weights.stride(0),
        lora_a_weights.stride(1),
        lora_a_weights.stride(2),
        lora_a_weights.stride(3),
        # LoRA B strides: [num_loras, num_experts, intermediate_dim, max_rank]
        lora_b_weights.stride(0),
        lora_b_weights.stride(1),
        lora_b_weights.stride(2),
        lora_b_weights.stride(3),
        lora_ranks,
        lora_scalings,
        BLOCK_HIDDEN,
        BLOCK_INTERMEDIATE,
        BLOCK_RANK,
    )

    return output
