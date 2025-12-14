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
    hidden_states_ptr,        # [num_total_tokens, hidden_dim]
    lora_a_weights_ptr,       # [num_loras, num_experts, max_rank, hidden_dim]
    lora_b_weights_ptr,       # [num_loras, num_experts, intermediate_dim, max_rank]
    output_ptr,               # [num_total_tokens, intermediate_dim]

    # Dispatch info (length = num_dispatched)
    token_ids_ptr,            # [num_dispatched] -> index into hidden/output
    expert_ids_ptr,           # [num_dispatched]
    lora_ids_ptr,             # [num_dispatched]

    # Dimensions
    hidden_dim: tl.constexpr,
    intermediate_dim: tl.constexpr,
    max_rank: tl.constexpr,
    num_experts: tl.constexpr,
    num_dispatched,

    # Strides for 4D LoRA A weights [num_loras, num_experts, max_rank, hidden_dim]
    lora_a_stride_lora: tl.constexpr,
    lora_a_stride_expert: tl.constexpr,
    lora_a_stride_rank: tl.constexpr,
    lora_a_stride_hidden: tl.constexpr,

    # Strides for 4D LoRA B weights [num_loras, num_experts, intermediate_dim, max_rank]
    lora_b_stride_lora: tl.constexpr,
    lora_b_stride_expert: tl.constexpr,
    lora_b_stride_intermediate: tl.constexpr,
    lora_b_stride_rank: tl.constexpr,

    # LoRA ranks per adapter [num_loras]
    lora_ranks_ptr,
    # Scaling factors per adapter [num_loras]
    lora_scalings_ptr,

    # Block size (used for hidden and output tiling; rank is not tiled)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute per-expert LoRA delta:

        delta[token, out_slice, lora] = B[out_slice, :] @ (A @ hidden_states[token])

    3D Grid: (spatial, slices, loras)
      - spatial = program_id(0): dispatched token index
      - slices  = program_id(1): tile index along intermediate_dim
      - loras   = program_id(2): LoRA adapter index
    """

    # 3D grid indices
    spatial_id = tl.program_id(0)   # dispatched token index
    slice_id = tl.program_id(1)     # output slice index
    lora_id_grid = tl.program_id(2) # LoRA adapter index

    # Bounds check on dispatched tokens
    if spatial_id >= num_dispatched:
        return

    # Load dispatch info for this dispatched index
    actual_token_id = tl.load(token_ids_ptr + spatial_id)
    expert_id = tl.load(expert_ids_ptr + spatial_id)
    token_lora_id = tl.load(lora_ids_ptr + spatial_id)

    # Skip if this token does not use this LoRA adapter
    if token_lora_id != lora_id_grid:
        return

    # Load LoRA rank and scaling (scalar tensors) for this LoRA adapter
    rank = tl.load(lora_ranks_ptr + lora_id_grid)
    scaling = tl.load(lora_scalings_ptr + lora_id_grid)
    has_rank = rank > 0
    if not has_rank:
        return

    # ----------------------------
    # Base pointers
    # ----------------------------
    # hidden_states[actual_token_id, :]
    hidden_ptr = hidden_states_ptr + actual_token_id * hidden_dim

    # A[lora_id_grid, expert_id, :, :]
    lora_a_base = (
        lora_a_weights_ptr
        + lora_id_grid * lora_a_stride_lora
        + expert_id * lora_a_stride_expert
    )

    # B[lora_id_grid, expert_id, :, :]
    lora_b_base = (
        lora_b_weights_ptr
        + lora_id_grid * lora_b_stride_lora
        + expert_id * lora_b_stride_expert
    )

    # ----------------------------
    # Stage 1: intermediate = A @ hidden
    # ----------------------------

    # We assume max_rank is small enough to keep as a single 1D vector
    r_offs = tl.arange(0, max_rank)                    # [max_rank]
    rank_mask = r_offs < rank                          # [max_rank]

    # Accumulator for intermediate: [max_rank]
    intermediate = tl.zeros((max_rank,), dtype=tl.float32)

    # Tile over hidden_dim in chunks of BLOCK_SIZE
    NUM_HIDDEN_TILES = (hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    for hidden_tile_idx in range(NUM_HIDDEN_TILES):
        hidden_start = hidden_tile_idx * BLOCK_SIZE
        hidden_offs = hidden_start + tl.arange(0, BLOCK_SIZE)    # [BLOCK_SIZE]
        hidden_mask = hidden_offs < hidden_dim                   # [BLOCK_SIZE]

        # Load hidden values for this tile: [BLOCK_SIZE]
        h_vals = tl.load(
            hidden_ptr + hidden_offs,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)

        # Build [max_rank, BLOCK_SIZE] tile of A:
        #   rows: r_offs
        #   cols: hidden_offs
        # offset = base + r * stride_rank + h * stride_hidden
        a_ptrs = (
            lora_a_base
            + r_offs[:, None] * lora_a_stride_rank
            + hidden_offs[None, :] * lora_a_stride_hidden
        )
        a_vals = tl.load(
            a_ptrs,
            mask=rank_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Dot over hidden axis: [max_rank]
        # intermediate[r] += sum_h A[r, h] * h_vals[h]
        intermediate += tl.sum(a_vals * h_vals[None, :], axis=1)

    # ----------------------------
    # Stage 2: y_slice = B[out_slice, :] @ intermediate
    # One output slice per program along intermediate_dim.
    # ----------------------------
    out_start = slice_id * BLOCK_SIZE
    out_offs = out_start + tl.arange(0, BLOCK_SIZE)     # [BLOCK_SIZE]
    out_mask = out_offs < intermediate_dim              # [BLOCK_SIZE]

    # If this slice is entirely out of bounds, we can early-exit
    # (not strictly necessary but cheap)
    # NOTE: Triton doesn't have a direct "if not any(mask)" primitive,
    # but the mask will naturally guard loads/stores below, so this is safe to omit.
    # We'll just rely on masks.

    # Build [max_rank, BLOCK_SIZE] tile of B:
    #   rows: r_offs (rank dimension)
    #   cols: out_offs (output dimension)
    # offset = base + out * stride_intermediate + r * stride_rank
    b_ptrs = (
        lora_b_base
        + out_offs[None, :] * lora_b_stride_intermediate
        + r_offs[:, None] * lora_b_stride_rank
    )
    b_vals = tl.load(
        b_ptrs,
        mask=rank_mask[:, None] & out_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Contribution:
    #   out_vals[j] = sum_r B[j, r] * intermediate[r]
    out_vals = tl.sum(b_vals * intermediate[:, None], axis=0)    # [BLOCK_SIZE]

    # Apply scaling
    out_vals *= scaling

    # ----------------------------
    # Accumulate into global output
    # ----------------------------
    out_row_base = actual_token_id * intermediate_dim
    out_ptrs = output_ptr + out_row_base + out_offs

    tl.atomic_add(
        out_ptrs,
        out_vals,
        mask=out_mask & has_rank,
    )

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
    Forward pass for per-expert LoRA computation using a 3D Triton grid:
        grid = (spatial, slices, loras)

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
    # Shapes
    num_tokens, hidden_dim = hidden_states.shape
    num_loras, _, intermediate_dim, max_rank = lora_b_weights.shape
    num_dispatched = token_ids.shape[0]

    # Make sure everything is on the same device and contiguous
    device = hidden_states.device
    hidden_states = hidden_states.contiguous()
    lora_a_weights = lora_a_weights.contiguous()
    lora_b_weights = lora_b_weights.contiguous()
    token_ids = token_ids.contiguous()
    expert_ids = expert_ids.contiguous()
    lora_ids = lora_ids.contiguous()
    lora_ranks = lora_ranks.contiguous()
    lora_scalings = lora_scalings.contiguous()

    # Initialize or reuse output tensor for in-place addition
    if base_output is None:
        # Use float32 for accumulation; you can cast back if needed
        output = torch.zeros(
            num_tokens,
            hidden_dim,
            dtype=torch.float32,
            device=device,
        )
    else:
        output = base_output
        assert output.shape == (num_tokens, hidden_dim) # TODO (jonahcb): check if this is correct
        assert output.device == device

    # Tile size for hidden and output dimensions
    BLOCK_SIZE = 64  # tune as needed

    # Number of output slices along intermediate_dim
    num_slices = (intermediate_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    # 3D grid: (spatial, slices, loras)
    grid = (num_dispatched, num_slices, num_loras)

    _per_expert_lora_kernel[grid](
        # Pointers
        hidden_states,          # hidden_states_ptr
        lora_a_weights,         # lora_a_weights_ptr
        lora_b_weights,         # lora_b_weights_ptr
        output,                 # output_ptr

        # Dispatch info
        token_ids,              # token_ids_ptr
        expert_ids,             # expert_ids_ptr
        lora_ids,               # lora_ids_ptr

        # Dimensions
        hidden_dim,             # hidden_dim
        intermediate_dim,       # intermediate_dim
        max_rank,               # max_rank
        num_experts,            # num_experts
        num_dispatched,         # num_dispatched (runtime scalar)

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

        # Rank & scaling
        lora_ranks,             # lora_ranks_ptr
        lora_scalings,          # lora_scalings_ptr

        # Block size (constexpr)
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
