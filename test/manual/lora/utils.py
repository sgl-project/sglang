from typing import Optional

import torch


def safe_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication with mixed precision handling for float16"""
    result = torch.matmul(a.float(), b.float())
    return result.to(a.dtype)


def reference_sgmv_shrink(
    x: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
    lora_ranks: torch.Tensor,
    lora_scalings: torch.Tensor,
    num_slices: int = 1,
) -> torch.Tensor:
    """
    Simple sequence-level reference implementation of SGMV shrink operation.

    Args:
        x: (total_seq_len, input_dim) - Input activations
        weights: (num_loras, num_slices * max_rank, input_dim) - LoRA A weights
        weight_indices: LoRA idx for each sequence
        seq_lengths: Length of each sequence
        lora_ranks: LoRA rank for each LoRA adapters
        lora_scalings: LoRA scaling for each LoRA adapters
        num_slices: Number of slices (3 for QKV, 2 for gate_up, 1 for others)

    Returns:
        output: (total_seq_len, num_slices * max_rank) - Intermediate activations
    """
    if weights.numel() == 0:
        total_seq_len = x.shape[0]
        return torch.zeros(total_seq_len, 0, dtype=x.dtype, device=x.device)

    total_seq_len, _ = x.shape
    _, weight_out_dim, _ = weights.shape
    max_rank = weight_out_dim // num_slices

    output = torch.zeros(
        total_seq_len, num_slices * max_rank, dtype=x.dtype, device=x.device
    )

    token_offset = 0
    for lora_idx, seq_len, rank, scaling in zip(
        weight_indices,
        seq_lengths,
        lora_ranks[weight_indices],
        lora_scalings[weight_indices],
    ):
        if seq_len == 0:
            continue

        if rank > 0:
            x_seq = x[token_offset : token_offset + seq_len, :]
            w_seq = weights[lora_idx, : num_slices * rank, :]

            result = safe_matmul(x_seq, w_seq.t())
            output[token_offset : token_offset + seq_len, : num_slices * rank] = (
                scaling * result
            )

        token_offset += seq_len

    return output


def reference_sgmv_expand(
    x: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
    lora_ranks: torch.Tensor,
    slice_offsets: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Simple sequence-level reference implementation of SGMV expand operation.

    Args:
        x: (total_seq_len, num_slices * max_rank) - Intermediate activations
        weights: (num_loras, output_dim, max_rank) - LoRA B weights
        weight_indices: LoRA idx for each sequence
        seq_lengths: Length of each sequence
        lora_ranks: LoRA rank for each LoRA adapters
        slice_offsets: Tensor defining slice boundaries
        base_output: Optional base output to accumulate into

    Returns:
        output: (total_seq_len, total_output_dim) - Final output
    """
    if weights.numel() == 0:
        total_seq_len = x.shape[0]
        total_output_dim = slice_offsets[-1].item() if len(slice_offsets) > 0 else 0
        return torch.zeros(
            total_seq_len, total_output_dim, dtype=x.dtype, device=x.device
        )

    total_seq_len, _ = x.shape

    num_slices = len(slice_offsets) - 1

    if base_output is not None:
        output = base_output.clone()
    else:
        total_output_dim = slice_offsets[-1].item()
        output = torch.zeros(
            total_seq_len, total_output_dim, dtype=x.dtype, device=x.device
        )

    token_offset = 0
    for lora_idx, seq_len, rank in zip(
        weight_indices, seq_lengths, lora_ranks[weight_indices]
    ):
        if seq_len == 0:
            continue

        if rank > 0:
            # Extract sequence intermediate activations
            x_seq = x[
                token_offset : token_offset + seq_len, : num_slices * rank
            ]  # (seq_len, num_slices * rank)

            for slice_idx in range(num_slices):
                slice_start_input = slice_idx * rank
                slice_end_input = (slice_idx + 1) * rank

                slice_start_output = slice_offsets[slice_idx].item()
                slice_end_output = slice_offsets[slice_idx + 1].item()

                x_slice = x_seq[:, slice_start_input:slice_end_input]  # (seq_len, rank)
                w_slice = weights[
                    lora_idx, slice_start_output:slice_end_output, :rank
                ]  # (slice_dim, rank)

                result = safe_matmul(x_slice, w_slice.t())  # (seq_len, slice_dim)
                output[
                    token_offset : token_offset + seq_len,
                    slice_start_output:slice_end_output,
                ] += result

        token_offset += seq_len

    return output
