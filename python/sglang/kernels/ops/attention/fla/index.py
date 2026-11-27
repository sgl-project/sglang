# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from sglang.kernels.ops.attention.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    if torch.compiler.is_compiling():
        cu_seqlens = cu_seqlens.to(torch.int64)
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    max_num_chunks: int | None = None,
) -> torch.LongTensor:
    chunks_per_sequence = triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    index_dtype = torch.int64 if torch.compiler.is_compiling() else cu_seqlens.dtype
    chunk_offsets = torch.nn.functional.pad(
        chunks_per_sequence.cumsum(0, dtype=index_dtype), (1, 0)
    )
    if max_num_chunks is not None and torch.compiler.is_compiling():
        chunk_slots = torch.arange(
            max_num_chunks,
            dtype=index_dtype,
            device=cu_seqlens.device,
        )
        sequence_indices = (chunk_slots[:, None] >= chunk_offsets[None, 1:]).sum(dim=1)
        sequence_indices = sequence_indices.clamp(max=chunks_per_sequence.shape[0] - 1)
        chunk_indices = chunk_slots - chunk_offsets[sequence_indices]
        return torch.stack((sequence_indices, chunk_indices), dim=1).to(
            cu_seqlens.dtype
        )

    if max_num_chunks is not None:
        num_chunks = int(chunks_per_sequence.sum())
        torch._check_is_size(num_chunks)
        torch._check(num_chunks <= max_num_chunks)
        padding = max_num_chunks - num_chunks
        repeat_counts = chunks_per_sequence
        repeat_counts = torch.cat(
            (
                repeat_counts[:-1],
                repeat_counts[-1:] + padding,
            )
        )
        output_size = max_num_chunks
    else:
        num_chunks = int(chunks_per_sequence.sum())
        torch._check_is_size(num_chunks)
        repeat_counts = chunks_per_sequence
        output_size = num_chunks
    sequence_indices = torch.repeat_interleave(
        torch.arange(
            chunks_per_sequence.shape[0],
            dtype=cu_seqlens.dtype,
            device=cu_seqlens.device,
        ),
        repeat_counts,
        output_size=output_size,
    )
    chunk_indices = (
        torch.arange(output_size, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        - chunk_offsets[sequence_indices]
    )
    return torch.stack(
        (sequence_indices, chunk_indices),
        dim=1,
    ).to(cu_seqlens.dtype)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
