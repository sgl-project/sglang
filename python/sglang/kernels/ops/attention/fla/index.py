# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from sglang.kernels.ops.attention.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    chunk_counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    indices = torch.cat([torch.arange(n) for n in chunk_counts])
    # Number chunks by sequence, so zero-length sequences keep their ids.
    seq_ids = torch.repeat_interleave(
        torch.arange(len(chunk_counts)),
        torch.tensor(chunk_counts, dtype=torch.long),
        output_size=indices.numel(),
    )
    return torch.stack([seq_ids, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
