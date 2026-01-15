from typing import Optional

import torch

from sglang.srt.server_args import get_global_server_args

def sgemm_lora_a_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    scaling_tensor: torch.Tensor,
    num_slices: int = 1,
):
    total_seq_len, input_dim = inputs.shape
    if weights.numel() == 0:
        return torch.zeros(total_seq_len, 0, dtype=inputs.dtype, device=inputs.device)

    num_loras, weight_out_dim, _ = weights.shape
    
    if get_global_server_args().disable_overlap_schedule and torch.all(weight_indices == weight_indices[0]):
        idx = weight_indices[0]
        rank = lora_ranks[idx]
        scaling = scaling_tensor[idx]
        if rank > 0:
            output = torch.einsum("si, oi -> so", inputs, weights[idx, :, :]) * scaling
        else:
            output = torch.zeros(total_seq_len, weight_out_dim, dtype=inputs.dtype, device=inputs.device)
        return output

    max_rank = weight_out_dim // num_slices

    output = torch.zeros(
        total_seq_len, num_slices * max_rank, dtype=inputs.dtype, device=inputs.device
    )

    token_offset = 0
    for lora_idx, seq_len, rank in zip(
        weight_indices, seg_len_tensor, lora_ranks[weight_indices]
    ):
        if seq_len == 0:
            continue

        if rank > 0:

            x_seq = inputs[token_offset : token_offset + seq_len, :]
            w_seq = weights[lora_idx, : num_slices * rank, :]

            result = torch.einsum("si, oi -> so", x_seq, w_seq)
            output[token_offset : token_offset + seq_len, : num_slices * rank] = (
                scaling_tensor[lora_idx] * result
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
):
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
    if get_global_server_args().disable_overlap_schedule and torch.all(weight_indices==weight_indices[0]):
        idx= weight_indices[0]
        rank = lora_ranks[idx]
        if rank > 0:
            for slice_idx in range(num_slices):
                slice_start_input = slice_idx * rank
                slice_end_input = (slice_idx + 1) * rank

                slice_start_output = slice_offsets[slice_idx]
                slice_end_output = slice_offsets[slice_idx + 1]

                x_slice = inputs[
                    :,
                    slice_start_input:slice_end_input,
                ]  # (seq_len, rank)
                w_slice = weights[
                    idx, slice_start_output:slice_end_output, :rank
                ]  # (slice_dim, rank)

                result = torch.einsum("si, oi -> so", x_slice, w_slice)
                output[
                    :,
                    slice_start_output:slice_end_output,
                ] += result

        return output

    token_offset = 0
    for lora_idx, seq_len, rank in zip(
        weight_indices, seg_len_tensor, lora_ranks[weight_indices]
    ):
        if seq_len == 0:
            continue

        if rank == 0:
            token_offset += seq_len
            continue

        for slice_idx in range(num_slices):
            slice_start_input = slice_idx * rank
            slice_end_input = (slice_idx + 1) * rank

            slice_start_output = slice_offsets[slice_idx]
            slice_end_output = slice_offsets[slice_idx + 1]

            x_slice = inputs[
                token_offset : token_offset + seq_len :,
                slice_start_input:slice_end_input,
            ]  # (seq_len, rank)
            w_slice = weights[
                lora_idx, slice_start_output:slice_end_output, :rank
            ]  # (slice_dim, rank)

            result = torch.einsum("si, oi -> so", x_slice, w_slice)
            output[
                token_offset : token_offset + seq_len,
                slice_start_output:slice_end_output,
            ] += result

        token_offset += seq_len

    return output
