from typing import Optional

import torch


def sgemm_lora_a_embedding_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    scaling_tensor: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    total_seq_len = inputs.shape[0]
    if weights.numel() == 0:
        return torch.zeros(total_seq_len, 0, dtype=weights.dtype, device=weights.device)

    num_loras, max_rank, _ = weights.shape

    output = torch.zeros(
        total_seq_len, max_rank, dtype=weights.dtype, device=weights.device
    )

    token_offset = 0
    for lora_idx, seq_len in zip(weight_indices, seg_len_tensor):
        if seq_len == 0:
            continue

        rank = lora_ranks[lora_idx]
        if rank > 0:

            x_seq = inputs[token_offset : token_offset + seq_len]
            w_seq = weights[lora_idx, :rank]

            result = torch.nn.functional.embedding(x_seq, w_seq.T)
            output[token_offset : token_offset + seq_len, :rank] = (
                scaling_tensor[lora_idx].item() * result
            )

        token_offset += seq_len

    return output


def sgemm_lora_a_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    scaling_tensor: torch.Tensor,
    num_slices: int = 1,
) -> torch.Tensor:
    total_seq_len, input_dim = inputs.shape
    if weights.numel() == 0:
        return torch.zeros(total_seq_len, 0, dtype=inputs.dtype, device=inputs.device)

    num_loras, weight_out_dim, _ = weights.shape
    max_rank = weight_out_dim // num_slices

    output = torch.zeros(
        total_seq_len, num_slices * max_rank, dtype=inputs.dtype, device=inputs.device
    )

    token_offset = 0
    for lora_idx, seq_len in zip(weight_indices, seg_len_tensor):
        if seq_len == 0:
            continue

        rank = lora_ranks[lora_idx]
        if rank > 0:

            x_seq = inputs[token_offset : token_offset + seq_len]
            w_seq = weights[lora_idx, : num_slices * rank]

            output[token_offset : token_offset + seq_len, : num_slices * rank].addmm_(
                x_seq,
                w_seq.T,
                beta=0,
                alpha=scaling_tensor[lora_idx].item(),
            )

        token_offset += seq_len

    return output


def sgemm_lora_b_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    slice_offsets: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    total_seq_len, _ = inputs.shape
    num_loras, weight_out_dim, _ = weights.shape
    total_output_dim = slice_offsets[-1].item() if len(slice_offsets) > 0 else 0

    if weights.numel() == 0:
        return torch.zeros(
            total_seq_len, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )

    num_slices = len(slice_offsets) - 1

    if base_output is not None:
        output = base_output
    else:
        output = torch.zeros(
            total_seq_len, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )

    token_offset = 0
    for lora_idx, seq_len in zip(weight_indices, seg_len_tensor):
        if seq_len == 0:
            continue

        rank = lora_ranks[lora_idx]
        if rank > 0:

            for slice_idx in range(num_slices):
                slice_start_input = slice_idx * rank
                slice_end_input = (slice_idx + 1) * rank

                slice_start_output = slice_offsets[slice_idx]
                slice_end_output = slice_offsets[slice_idx + 1]

                x_slice = inputs[
                    token_offset : token_offset + seq_len,
                    slice_start_input:slice_end_input,
                ]  # (seq_len, rank)
                w_slice = weights[
                    lora_idx, slice_start_output:slice_end_output, :rank
                ]  # (slice_dim, rank)

                output[
                    token_offset : token_offset + seq_len,
                    slice_start_output:slice_end_output,
                ].addmm_(x_slice, w_slice.T)

        token_offset += seq_len

    return output
