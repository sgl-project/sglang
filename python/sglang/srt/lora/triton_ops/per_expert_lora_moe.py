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
    hidden_states_ptr,  # [num_total_tokens, input_dim]
    lora_a_weights_ptr,  # [num_loras, num_experts, max_rank, input_dim]
    lora_b_weights_ptr,  # [num_loras, num_experts, output_dim, max_rank]
    output_ptr,  # [num_total_tokens, output_dim] - base output (modified in-place)
    lora_output_ptr,  # [num_total_tokens, output_dim] - separate LoRA-only output
    # Dispatch info (length = num_dispatched)
    token_ids_ptr,  # [num_dispatched] -> index into hidden/output
    expert_ids_ptr,  # [num_dispatched]
    lora_ids_ptr,  # [num_dispatched]
    topk_weights_ptr,  # [num_dispatched] - Router weights for each dispatched token
    # Dimensions
    input_dim: tl.constexpr,
    output_dim: tl.constexpr,
    max_rank: tl.constexpr,
    num_experts: tl.constexpr,
    num_dispatched,
    # Strides for 4D LoRA A weights [num_loras, num_experts, max_rank, input_dim]
    lora_a_stride_lora: tl.constexpr,
    lora_a_stride_expert: tl.constexpr,
    lora_a_stride_rank: tl.constexpr,
    lora_a_stride_input: tl.constexpr,
    # Strides for 4D LoRA B weights [num_loras, num_experts, output_dim, max_rank]
    lora_b_stride_lora: tl.constexpr,
    lora_b_stride_expert: tl.constexpr,
    lora_b_stride_output: tl.constexpr,
    lora_b_stride_rank: tl.constexpr,
    # LoRA ranks per adapter [num_loras]
    lora_ranks_ptr,
    # Scaling factors per adapter [num_loras]
    lora_scalings_ptr,
    # Block size (used for input and output tiling; rank is not tiled)
    BLOCK_SIZE: tl.constexpr,
    # Whether to multiply by router weights
    MUL_ROUTED_WEIGHT: tl.constexpr,
    # Whether this is down_proj (affects stacking factor for rank calculation)
    IS_DOWN_PROJ: tl.constexpr,
):
    """
    Compute per-expert LoRA delta:

        delta[token, out_slice, lora] = B[out_slice, :] @ (A @ hidden_states[token])

    3D Grid: (spatial, slices, loras)
      - spatial = program_id(0): dispatched token index
      - slices  = program_id(1): tile index along output_dim
      - loras   = program_id(2): LoRA adapter index
    """

    # 3D grid indices
    spatial_id = tl.program_id(0)  # dispatched token index
    slice_id = tl.program_id(1)  # output slice index
    lora_id_grid = tl.program_id(2)  # LoRA adapter index

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

    # Adjust rank for stacked modules (gate_up_proj has stacking factor 2)
    effective_rank = rank
    if not IS_DOWN_PROJ:  # gate_up_proj case
        effective_rank = rank * 2

    has_rank = effective_rank > 0
    if not has_rank:
        return

    # ----------------------------
    # Base pointers
    # ----------------------------
    # hidden_states[actual_token_id, :]
    hidden_ptr = hidden_states_ptr + actual_token_id * input_dim

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
    r_offs = tl.arange(0, max_rank)  # [max_rank]
    rank_mask = r_offs < effective_rank  # [max_rank]

    # TODO (Jonahcb): check if it is better to allocate outside the kernel
    # Accumulator for intermediate: [max_rank]
    intermediate = tl.zeros((max_rank,), dtype=tl.float32)

    # Tile over input_dim in chunks of BLOCK_SIZE
    NUM_INPUT_TILES = (input_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    for input_tile_idx in range(NUM_INPUT_TILES):
        input_start = input_tile_idx * BLOCK_SIZE
        input_offs = input_start + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]
        input_mask = input_offs < input_dim  # [BLOCK_SIZE]

        # Load input values for this tile: [BLOCK_SIZE]
        h_vals = tl.load(
            hidden_ptr + input_offs,
            mask=input_mask,
            other=0.0,
        ).to(tl.float32)

        # Build [max_rank, BLOCK_SIZE] tile of A:
        #   rows: r_offs
        #   cols: input_offs
        # offset = base + r * stride_rank + h * stride_input
        a_ptrs = (
            lora_a_base
            + r_offs[:, None] * lora_a_stride_rank
            + input_offs[None, :]
            * lora_a_stride_input  # check if it is necessary to multiply by stride value as it should be contigious in this dimension
        )
        a_vals = tl.load(
            a_ptrs,
            mask=rank_mask[:, None] & input_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Dot over hidden axis: [max_rank]
        # intermediate[r] += sum_h A[r, h] * h_vals[h]
        intermediate += tl.sum(a_vals * h_vals[None, :], axis=1)

    # ----------------------------
    # Stage 2: y_slice = B[out_slice, :] @ intermediate
    # One output slice per program along output_dim.
    # ----------------------------
    out_start = slice_id * BLOCK_SIZE
    out_offs = out_start + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]
    out_mask = out_offs < output_dim  # [BLOCK_SIZE]

    # Build [max_rank, BLOCK_SIZE] tile of B:
    #   rows: r_offs (rank dimension)
    #   cols: out_offs (output dimension)
    # offset = base + out * stride_output + r * stride_rank
    b_ptrs = (
        lora_b_base
        + out_offs[None, :] * lora_b_stride_output
        + r_offs[:, None] * lora_b_stride_rank
    )
    b_vals = tl.load(
        b_ptrs,
        mask=rank_mask[:, None] & out_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    #   out_vals[j] = sum_r B[j, r] * intermediate[r]
    out_vals = tl.sum(b_vals * intermediate[:, None], axis=0)  # [BLOCK_SIZE]

    # Apply scaling
    out_vals *= scaling

    # Apply router weight if enabled (matches base MoE behavior)
    if MUL_ROUTED_WEIGHT:
        topk_weight = tl.load(
            topk_weights_ptr + spatial_id
        )  # I don't think this correctly understands how top_k weights is organized (now moe dispatch reorganizes topk weights to work w this)
        out_vals *= topk_weight

    # ----------------------------
    # Accumulate into global output (base_output) and store to lora_output
    # ----------------------------
    out_row_base = actual_token_id * output_dim
    out_ptrs = output_ptr + out_row_base + out_offs
    lora_out_ptrs = lora_output_ptr + out_row_base + out_offs

    # Convert to output dtype (matches hidden_states dtype, could be float16/bfloat16/float32)
    out_vals_typed = out_vals.to(output_ptr.dtype.element_ty)

    # Add to base_output in-place
    tl.atomic_add(out_ptrs, out_vals_typed, out_mask)

    # Also store to separate lora_output tensor (same dtype)
    tl.atomic_add(lora_out_ptrs, out_vals_typed, out_mask)


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
    is_down_proj: bool = False,
    topk_weights: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for per-expert LoRA computation using a 3D Triton grid:
        grid = (spatial, slices, loras)

    Args:
        hidden_states: [num_tokens, input_dim] where input_dim is hidden_dim for gate_up_proj
                      or intermediate_dim for down_proj
        lora_a_weights: [num_loras, num_experts, max_rank, input_dim]
        lora_b_weights: [num_loras, num_experts, output_dim, max_rank]
        token_ids: [num_dispatched] - Original token indices
        expert_ids: [num_dispatched] - Expert ID for each dispatched token
        lora_ids: [num_dispatched] - LoRA ID for each dispatched token
        lora_ranks: [num_loras] - Rank for each LoRA
        lora_scalings: [num_loras] - Scaling factor for each LoRA
        num_experts: Total number of experts
        base_output: [num_tokens, output_dim] - Base MoE output (modified in-place)
        is_down_proj: Whether this is for down_proj (intermediate_dim -> hidden_dim)
                     or gate_up_proj (hidden_dim -> intermediate_dim)
        topk_weights: [num_dispatched] - Router weights for each dispatched token.
                     Always multiplied by output (router weights are applied to final output).

    Returns:
        tuple of:
            output: [num_tokens, output_dim] - Base output + LoRA delta (in-place)
            lora_output: [num_tokens, output_dim] - Just the LoRA delta contribution
    """

    # Shapes
    num_tokens, input_dim = hidden_states.shape
    num_loras, _, output_dim, _ = lora_b_weights.shape
    num_dispatched = token_ids.shape[0]

    # Use fixed max_rank for consistent kernel compilation
    # Maximum stacking factor is 2 (for gate_up_proj), so max_rank = max_lora_rank * 2
    # We assume max_lora_rank is reasonably small (e.g., 64-128) so max_rank = 256 is safe
    max_rank = 256  # Conservative upper bound for max_lora_rank * 2

    # Make sure everything is on the same device and contiguous
    device = hidden_states.device

    # Use hidden_states dtype for consistency with model
    dtype = hidden_states.dtype
    hidden_states = hidden_states.contiguous()
    lora_a_weights = lora_a_weights.contiguous()
    lora_b_weights = lora_b_weights.contiguous()
    token_ids = token_ids.contiguous()
    expert_ids = expert_ids.contiguous()
    lora_ids = lora_ids.contiguous()
    lora_ranks = lora_ranks.contiguous()
    lora_scalings = lora_scalings.contiguous()

    # Handle topk_weights (always provided, but only applied for down_proj)
    mul_routed_weight = is_down_proj  # Apply router weights only to down_proj output
    if topk_weights is not None:
        topk_weights = topk_weights.contiguous()
    else:
        # Create dummy tensor if not provided
        topk_weights = torch.empty(0, device=device, dtype=dtype)

    # Initialize or reuse output tensor for in-place addition
    if base_output is None:
        # Use specified dtype for consistency with model
        output = torch.zeros(
            num_tokens,
            output_dim,
            dtype=dtype,
            device=device,
        )
    else:
        output = base_output
        assert output.shape == (
            num_tokens,
            output_dim,
        ), f"Expected shape ({num_tokens}, {output_dim}), got {output.shape}"
        assert output.device == device

    # Allocate separate tensor for just the LoRA contribution
    lora_output = torch.zeros(
        num_tokens,
        output_dim,
        dtype=dtype,
        device=device,
    )

    # Tile size for hidden and output dimensions
    BLOCK_SIZE = 64  # tune as needed

    # Number of output slices along output_dim
    num_slices = (output_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    # 3D grid: (spatial, slices, loras)
    grid = (num_dispatched, num_slices, num_loras)

    _per_expert_lora_kernel[grid](
        # Pointers
        hidden_states,  # hidden_states_ptr
        lora_a_weights,  # lora_a_weights_ptr
        lora_b_weights,  # lora_b_weights_ptr
        output,  # output_ptr (base output, modified in-place)
        lora_output,  # lora_output_ptr (separate LoRA-only output)
        # Dispatch info
        token_ids,  # token_ids_ptr
        expert_ids,  # expert_ids_ptr
        lora_ids,  # lora_ids_ptr
        topk_weights,  # topk_weights_ptr
        # Dimensions
        input_dim,  # input_dim (hidden_dim for gate_up_proj, intermediate_dim for down_proj)
        output_dim,  # output_dim (intermediate_dim for gate_up_proj, hidden_dim for down_proj)
        max_rank,  # max_rank
        num_experts,  # num_experts
        num_dispatched,  # num_dispatched (runtime scalar)
        # LoRA A strides: [num_loras, num_experts, max_rank, input_dim]
        lora_a_weights.stride(0),
        lora_a_weights.stride(1),
        lora_a_weights.stride(2),
        lora_a_weights.stride(3),
        # LoRA B strides: [num_loras, num_experts, output_dim, max_rank]
        lora_b_weights.stride(0),
        lora_b_weights.stride(1),
        lora_b_weights.stride(2),
        lora_b_weights.stride(3),
        # Rank & scaling
        lora_ranks,  # lora_ranks_ptr
        lora_scalings,  # lora_scalings_ptr
        # Block size (constexpr)
        BLOCK_SIZE=BLOCK_SIZE,
        # Router weight multiplication flag
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        # Whether this is down_proj
        IS_DOWN_PROJ=is_down_proj,
    )

    return output, lora_output
